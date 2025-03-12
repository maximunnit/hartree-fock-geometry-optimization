import numpy as np
import scipy.special as sp
import copy
from numba import njit
from timeit import default_timer as timer

Zs = (                           # atomic symbols in order, used to get the nuclear charge from symbol
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", 
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", 
    "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "I", 
    "Te", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", 
    "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", 
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", 
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
)

cartesian_gaussian_ijk = {      # all the possible i,j,k permutations for Cartesian GTFs of each type
    "S": ((0,0,0),),
    "P": ((1,0,0),(0,1,0),(0,0,1)),
    "D": ((2,0,0),(0,2,0),(0,0,2),(1,1,0),(0,1,1),(1,0,1)),
    "F": ((3,0,0),(0,3,0),(0,0,3),(2,1,0),(2,0,1),(1,2,0),(0,2,1),(1,0,2),(0,1,2),(1,1,1))
}


class Atom:     # holds symbol and position of atom in structure
    def __init__(self, symbol, pos):
        self.pos = np.array(pos,dtype=float)    # position of the nucleus
        self.symbol = symbol                    # atomic symbol
        self.Z = Zs.index(symbol) + 1           # nuclear charge

    def __str__(self):
        return f"{self.symbol}: {self.pos}"


@njit
def JITValue(pos, atom_pos, ijk, orbexps, coefs):
    # each primitive gaussian has the form  g = N * x^i * y^j * z^k * e^-orbexp*r^2
    # the CGTF evaluates to a weighted sum of each primitive gaussian, with contraction coefs as weights
    # rel_pos is an array with an x, y, and z column for handling evaluation of a batch of points at once
    rel_pos = (pos-atom_pos).transpose()    # position relative to atomic nucleus
    i,j,k = ijk
    M = len(orbexps)
    exponential_factor = np.exp((rel_pos[0]**2+rel_pos[1]**2+rel_pos[2]**2).repeat(M).reshape(-1,M) * -orbexps)            #        
    unnormalized_primitive_gaussians_matrix = (rel_pos[0]**i * rel_pos[1]**j * rel_pos[2]**k).repeat(M).reshape(-1,M) * exponential_factor  # the repeat+reshape has the same effect as np.tile, but compatible with JIT
    return unnormalized_primitive_gaussians_matrix @ coefs  # matrix multiplication


class CGTF:     # (cartesian) contracted gaussian-type function
    def __init__(self, atom, ijk, Ns, orbexps, contcoefs):
        self.atom = atom                        # atom this basis function is from
        self.ijk = ijk                          # i, j, k values that determine the angular momentum of the orbital
        self.Ns = Ns                            # normalization constants of each of the functions
        self.orbital_exponents = orbexps        # orbital exponents of each of the functions
        self.contraction_coefs = contcoefs      # contraction coefficients of each of the functions
        self.combined_coefs = contcoefs*Ns
    
    def Value(self, pos):
        return JITValue(pos, self.atom.pos, self.ijk, self.orbital_exponents, self.combined_coefs)


def ReadBasisSetFile(f):        # basis set file from https://www.basissetexchange.org/ in Jaguar format
    print(f"Reading basis set file {f}")
    basis_set_name = ""         # name of the basis set
    basis_set = {}     # dictionary of all the basis functions per element

    bsf = open(f"./basissets/{f}", "r")
    bsftext = bsf.read()
    bsf.close()

    # prepare the text for looping through:
    element_text = bsftext[bsftext.find("BASIS"):].split("****")[:-1]   # each element's basis set data ends with the line ****, here we also trim out the last string because it's always empty
    firstline = element_text[0].splitlines()[0]
    basis_set_name = firstline.split(" ")[1]                            # the first element in the set also has the basis set name on the first line
    element_text[0] = element_text[0][len(firstline):]                  # after we read the name we trim out this first line, so the first element matches the rest of the elements

    for e in element_text:          # e is a string containing all the lines associated with one particular element
        e = e.strip().splitlines()
        element_symbol = e[0].strip()   # first line is its atomic symbol
        CGTFs = []
        for l in range(1,len(e)):   # the rest of the lines are data, looping over line index here so we can read lines in batches and then add to the index to skip the ones we've read
            if e[l][0] in cartesian_gaussian_ijk:               # if the line denotes an orbital type and number of primitive gaussians, then construct the contracted gaussian from the next lines
                num_gaussians = int(e[l].split(" ")[2])
                orbexps = np.zeros(num_gaussians)                       
                contcoefs = np.zeros(num_gaussians)

                for g in range(num_gaussians):                  # get the orbital exponents and contraction coefficients for the CGTFs 
                    orbexps[g], contcoefs[g] = map(float, e[l+g+1].replace("D", "E").split())   # replace D with E because scientific notation is written like 0000D00 instead of 0000E00 in these files

                for ijk in cartesian_gaussian_ijk[e[l][0]]:     # p-type basis functions come in sets of 3, d-type basis functions come in sets of 6, f-type in sets of 10, etc
                    Ns = np.zeros(num_gaussians)                # normalization constants depend on i,j,k
                    Ns = ((2*orbexps/np.pi)**0.75) * np.sqrt((8*orbexps)**sum(ijk) * sp.factorial(ijk[0])*sp.factorial(ijk[1])*sp.factorial(ijk[2])/(sp.factorial(2*ijk[0])*sp.factorial(2*ijk[1])*sp.factorial(2*ijk[2])))
                    CGTFs.append(CGTF(Atom("H", np.zeros(3, float)), ijk, Ns, orbexps, contcoefs))

                l += num_gaussians
        basis_set[element_symbol] = CGTFs.copy()
        
    print(f"Finished getting basis set: {basis_set_name}")
    return basis_set

def GetBasisFunctions(atoms, basis_set):    # takes in a basis set and a list of atoms (i.e. the structure of the molecule) and returns all the relevant basis functions
    basis_functions = []
    for atom in atoms:
        for _ in basis_set[atom.symbol]:
            bf = copy.copy(_)
            bf.atom = atom
            basis_functions.append(bf)
    return basis_functions
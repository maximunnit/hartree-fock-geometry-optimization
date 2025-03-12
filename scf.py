from pylebedev import PyLebedev
leb = PyLebedev()
import scipy.integrate as si
import itertools as it
from numba import njit
from timeit import default_timer as timer

from basisset import *

np.set_printoptions(precision=3)

epsilon = 1e-3      # small value used throughout the code
point_density = 4   # how dense the repulsion integral r1,r2 grid is
p,w = leb.get_points_and_weights(11)    # [x,y,z] points on a sphere, and their weights, for lebedev quadrature spherical integration

verbose = False     # set to True to have the program print out its steps as it goes

def GetOverlapIntegrals(basis_functions):
    if verbose: print("Calculating overlap integrals")
    B = len(basis_functions)
    overlaps = np.zeros((B,B))

    for i in range(B):
        for j in range(i+1):            # only run the loop over the top half echelon, it's a symmetric matrix
            bf1 = basis_functions[i]
            bf2 = basis_functions[j]
            avg_atom_pos = (bf1.atom.pos + bf2.atom.pos)/2  # evaluate the integrals starting from the average atom position rather than at (0,0,0)

            def integrand(pos):
                return bf1.Value(pos)*bf2.Value(pos)

            overlap = si.quad(lambda r: 4*np.pi*r*r*np.sum(w*integrand(r*p + avg_atom_pos)), 0, np.infty, limit=10000)[0]
            
            overlaps[i,j] = overlap
            overlaps[j,i] = overlap   # S_ab = S_ba  since our basis functions are real    
    return overlaps

@njit
def JIT_HcoreCoulombCoef(pos,Zs,atoms_pos):
    coef = np.zeros(len(pos))
    for i in range(len(Zs)):
        coef += Zs[i]/np.sqrt(np.sum((pos-atoms_pos[i])**2, axis=1) + epsilon)  # adding a tiny value to avoid dividing by zero, adds some negligible error
    return coef

def GetCoreHamiltonianIntegrals(basis_functions, atoms):
    if verbose: print("Calculating core Hamiltonian integrals")
    B = len(basis_functions)
    core_hamils = np.zeros((B,B))

    inv_eps2 = 1/epsilon**2

    for i in range(B):
        for j in range(i+1):                    # only run the loop over the top half echelon, it's a symmetric matrix
            start_time = timer()
            bf1 = basis_functions[i]
            bf2 = basis_functions[j]
            avg_atom_pos = (bf1.atom.pos + bf2.atom.pos)/2  # evaluate the integrals starting from the average atom position rather than at (0,0,0)

            rmax = np.sqrt(-np.log(epsilon)/np.min(np.append(bf1.orbital_exponents, bf2.orbital_exponents)))       # solves for r in e^-orbexp*r^2 = epsilon using the smallest orbital exponent, integrating past this radial distance contributes a negligible amount
            rmax += max(np.linalg.norm(bf1.atom.pos-avg_atom_pos), np.linalg.norm(bf2.atom.pos-avg_atom_pos))      # we're integrating around the avg atom position rather than the atom, so extend the radius to actually include the all of the space with significant contribution to the integral

            Zs = np.array([a.Z for a in atoms])
            atoms_pos = np.array([a.pos for a in atoms])
            def integrand(pos):
                laplacian_bf2  = (bf2.Value(pos - [epsilon,0,0]) + bf2.Value(pos + [epsilon,0,0]) - 6*bf2.Value(pos))   # central difference second derivatives added together
                laplacian_bf2 += (bf2.Value(pos - [0,epsilon,0]) + bf2.Value(pos + [0,epsilon,0]))
                laplacian_bf2 += (bf2.Value(pos - [0,0,epsilon]) + bf2.Value(pos + [0,0,epsilon]))
                laplacian_bf2 *= inv_eps2
                
                return bf1.Value(pos)*(-0.5*laplacian_bf2 - bf2.Value(pos)*JIT_HcoreCoulombCoef(pos, Zs, atoms_pos))
                
            hamil = si.quad(lambda r: 4*np.pi*r*r*np.sum(w*integrand(r*p + avg_atom_pos)), 0, rmax, limit=10000)[0]    # integration using scipy's gaussian quadrature and pylebedev's lebedev quadrature


            core_hamils[i,j] = hamil
            core_hamils[j,i] = hamil  # Hcore_ab = Hcore_ba  since our basis functions are real
            #print(f"Hcore integral {idx} took {timer()-start_time:.2f}s --- {basis_functions[idx[0]].atom} {basis_functions[idx[0]].ijk} and {basis_functions[idx[1]].atom} {basis_functions[idx[1]].ijk} --> evaluated to {core_hamils[idx]}")
    return core_hamils


@njit
def JIT_Pos1Pos2(r1, r2, atoms_pos, atom1):
    return r1*p + atoms_pos[atom1], r2*p + atoms_pos[atom1] # centered around the same atom

# Stratmann-Scuseria-Frisch partitioning, lets us integrate around each atom center so we can focus the integration sample points more closely around the locations of interest
# https://sci-hub.se/https://doi.org/10.1016/0009-2614(96)00600-8
num_lebedev_points = len(p)
SSF_a = 0.64    # the threshold quantity a, described in the paper, optimal value for partition curve threshold given to be 0.64
@njit
def JIT_SSFPartitionPolynomial(mu): # z(mu)
    return 0.0625 * (35*(mu/SSF_a) - 35*(mu/SSF_a)**3 + 21*(mu/SSF_a)**5 - 5*(mu/SSF_a)**7)

@njit
def JIT_SSFCellFunction(pos, atoms_pos):
    num_atom_points = len(atoms_pos)
    cell_function_matrix = np.ones((num_atom_points, num_atom_points, num_lebedev_points)) # matrix we do all the work in, ultimately to contain the cell function values s(mu)
    for i in range(num_atom_points):
        for j in range(i):    # only run the loop over the top half echelon since it's an antisymmetric matrix, also avoiding the diagonal because they are undefined
            r_ig = (atoms_pos[i]-pos)
            r_jg = (atoms_pos[j]-pos)
            R_ij = (atoms_pos[i]-atoms_pos[j])
            mu = (np.sqrt(r_ig[:,0]**2 + r_ig[:,1]**2 + r_ig[:,2]**2) - np.sqrt(r_jg[:,0]**2 + r_jg[:,1]**2 + r_jg[:,2]**2))/np.sqrt(R_ij[0]**2 + R_ij[1]**2 + R_ij[2]**2)
            matrix_element = np.array([(np.sign(_) if abs(_)>=SSF_a else JIT_SSFPartitionPolynomial(_)) for _ in mu]) # piecewise function, -1 when <= -a, z(mu) when between -a and a, +1 when >= a
            cell_function_matrix[i,j] = matrix_element      
            cell_function_matrix[j,i] = -matrix_element # antisymmetric matrix
    return 0.5*(1-cell_function_matrix)

# lebedev spherical surface double integral - integrating this leaves only an integral over r1,r2
@njit
def JIT_RepulsionIntegrand(pos1, pos2, r1, r2, wavefunctions_matrix):
    r_squared_matrix  = (pos1[:,0].repeat(num_lebedev_points).reshape(-1,num_lebedev_points) - pos2[:,0].repeat(num_lebedev_points).reshape(-1,num_lebedev_points).transpose())**2    # matrix of the distance^2 between every pair of points in the pos1 and pos2 arrays
    r_squared_matrix += (pos1[:,1].repeat(num_lebedev_points).reshape(-1,num_lebedev_points) - pos2[:,1].repeat(num_lebedev_points).reshape(-1,num_lebedev_points).transpose())**2
    r_squared_matrix += (pos1[:,2].repeat(num_lebedev_points).reshape(-1,num_lebedev_points) - pos2[:,2].repeat(num_lebedev_points).reshape(-1,num_lebedev_points).transpose())**2
    r_12_matrix = np.sqrt(r_squared_matrix)   # adding a tiny value to avoid dividing by zero, adds some negligible error
    return 16*np.pi*np.pi*r1*r1*r2*r2*np.sum(wavefunctions_matrix/r_12_matrix)


# two-electron repulsion integrals calculated using lebedev quadrature for the spherical parts and riemann quadrature for the radial parts
# not the best approach here performance-wise but with an adequate point density the results are still good
# see https://sci-hub.se/https://doi.org/10.1002/wcms.78 for approaches that could yield better performance
def GetRepulsionIntegrals(basis_functions, atoms):
    if verbose: print("Calculating two-electron repulsion integrals")
    B = len(basis_functions)
    repulsions = np.ones((B,B,B,B))*(np.nan)            # default value is NaN so we know which elements we haven't set yet

    atoms_pos = np.array([a.pos for a in atoms])
    num_atoms = len(atoms_pos)
    atom_SSF_radius_thresholds = np.zeros(num_atoms)    # if r < 0.5*(1-a)*threshold then we know instantly that every weight = 1, without having to compute anything
    for a in range(num_atoms):
        different_atoms_pos = np.append(atoms_pos[:a], atoms_pos[a+1:], axis=0)
        dist_to_nearest = np.min(np.linalg.norm(different_atoms_pos-atoms_pos[a], axis=1))
        atom_SSF_radius_thresholds[a] = 0.5*(1-SSF_a)*dist_to_nearest

    for idx, i in np.ndenumerate(repulsions):           # not the best way to do this for loop but it's not slowing it down at all so it's fine
        start_time = timer()
        if np.isnan(repulsions[idx]):
            bf1 = basis_functions[idx[0]]
            bf2 = basis_functions[idx[1]]
            bf3 = basis_functions[idx[2]]
            bf4 = basis_functions[idx[3]]

            rmax = np.sqrt(-np.log(epsilon)/np.min(np.append(np.append(bf1.orbital_exponents, bf2.orbital_exponents), np.append(bf3.orbital_exponents, bf4.orbital_exponents))))    # solves for r in e^-orbexp*r^2 = epsilon using the smallest orbital exponent, integrating past this radial distance contributes a negligible amount

            repulsion = 0
            num_points = int(rmax*point_density)
            differential = rmax*rmax/(num_points*num_points)    # dr1dr2

            for atom1 in range(len(atoms)):
                for atom2 in range(len(atoms)):
                    def integrand(r2, r1):
                        #int_time = timer()
                        pos1, pos2 = JIT_Pos1Pos2(r1, r2, atoms_pos, atom1)
                        #print(f"calculating pos1, pos2 took {(timer()-int_time)*1e6:.2f}μs")
                        #int_time = timer()
                        ssf_w1 = np.ones(num_lebedev_points)
                        ssf_w2 = np.ones(num_lebedev_points)
                        if r1 >= atom_SSF_radius_thresholds[atom1]: # only do the full weight calculation if the distance from the nucleus is greater than the threshold
                            cell_function_matrix = JIT_SSFCellFunction(pos1, atoms_pos)
                            ssf_w1 = np.prod(np.append(cell_function_matrix[atom1,:atom1], cell_function_matrix[atom1,atom1+1:], axis=0), axis=0)          # unnormalized weight
                            ssf_w1 = ssf_w1/np.sum([np.prod(np.append(cell_function_matrix[_,:_], cell_function_matrix[_,_+1:], axis=0), axis=0) for _ in range(num_atoms)], axis=0)  # normalize the weight
                        if r2 >= atom_SSF_radius_thresholds[atom2]:
                            cell_function_matrix = JIT_SSFCellFunction(pos2, atoms_pos)
                            ssf_w2 = np.prod(np.append(cell_function_matrix[atom2,:atom2], cell_function_matrix[atom2,atom2+1:], axis=0), axis=0)          # unnormalized weight
                            ssf_w2 = ssf_w2/np.sum([np.prod(np.append(cell_function_matrix[_,:_], cell_function_matrix[_,_+1:], axis=0), axis=0) for _ in range(num_atoms)], axis=0)  # normalize the weight

                        wavefunctions_matrix = np.outer(w*ssf_w1*bf1.Value(pos1)*bf2.Value(pos1), w*ssf_w2*bf3.Value(pos2)*bf4.Value(pos2))
                        #print(f"calculating wavefunctions took {(timer()-int_time)*1e6:.2f}μs")
                        #int_time = timer()
                        result = JIT_RepulsionIntegrand(pos1, pos2, r1, r2, wavefunctions_matrix)
                        #print(f"calculating result took {(timer()-int_time)*1e6:.2f}μs")
                        return result
                    # repulsion = si.nquad(integrand, [[epsilon, rmax], [epsilon, rmax]], opts = [{'limit':10000},{'limit':10000}])[0]
                    for r1 in np.linspace(epsilon, rmax, num_points):
                        for r2 in np.linspace(epsilon, rmax, num_points):
                            if abs(r1-r2)>1e-5: repulsion += integrand(r2,r1) * differential # double integral over r1,r2  (abs term is to avoid r1=r2)
                            

            # the following eight evaluate to the same result so we can avoid some unnecessary calculations
            repulsions[idx] = repulsion
            repulsions[idx[1],idx[0],idx[2],idx[3]] = repulsion
            repulsions[idx[0],idx[1],idx[3],idx[2]] = repulsion
            repulsions[idx[1],idx[0],idx[3],idx[2]] = repulsion
            repulsions[idx[2],idx[3],idx[0],idx[1]] = repulsion
            repulsions[idx[3],idx[2],idx[0],idx[1]] = repulsion
            repulsions[idx[2],idx[3],idx[1],idx[0]] = repulsion
            repulsions[idx[::-1]] = repulsion
        if verbose: print(f"repulsion integral {idx} took {timer()-start_time:.2f}s --- {basis_functions[idx[0]].atom} {basis_functions[idx[0]].ijk} and {basis_functions[idx[1]].atom} {basis_functions[idx[1]].ijk} and {basis_functions[idx[2]].atom} {basis_functions[idx[2]].ijk} and {basis_functions[idx[3]].atom} {basis_functions[idx[3]].ijk}")
        if verbose: print(f"--> evaluated to {repulsions[idx]}")
    return repulsions


def GetDensityMatrix(roothaan_coefs, half_num_electrons):
    B = len(roothaan_coefs)
    densities = np.zeros((B,B))

    for i in range(B):
        for j in range(i+1):                    # only run the loop over the top half echelon, it's a symmetric matrix
            densities[i,j] = 0
            for k in range(half_num_electrons):
                densities[i,j] += roothaan_coefs[i, k]*roothaan_coefs[j, k]

            densities[j,i] = densities[i,j]     # P_ab = P_ba since the roothaan coefs are real
    
    return 2*densities  # factor of 2

def GetFockMatrix(core_hamils, densities, repulsions):
    fock_matrix = np.zeros_like(core_hamils)

    for idx, i in np.ndenumerate(fock_matrix):
        double_sum = 0
        for idy, j in np.ndenumerate(densities):
                double_sum += densities[idy] * (repulsions[idx[0], idx[1], idy[0], idy[1]] - 0.5 * repulsions[idx[0], idy[1], idy[0], idx[1]])
        fock_matrix[idx] = core_hamils[idx] + double_sum
    
    return fock_matrix

def GetNuclearRepulsionEnergy(atoms):
    V_NN = 0
    for atoms in it.combinations(atoms, 2):
        V_NN += atoms[0].Z*atoms[1].Z/np.linalg.norm(atoms[0].pos - atoms[1].pos)
    return V_NN


def GetSCFEnergy(atoms, basis_set, num_electrons):
    if verbose: print("Starting SCF energy calculation")
    
    if num_electrons % 2 != 0:
        print("Not a closed shell molecule: number of electrons must be even")
        return 0

    basis_functions = GetBasisFunctions(atoms, basis_set)

    overlaps = GetOverlapIntegrals(basis_functions)             # electron overlap integrals
    overlap_eigvals, overlap_eigvecs = np.linalg.eigh(overlaps)
    orthonormalization_matrix = overlap_eigvecs @ (np.diag(overlap_eigvals**-0.5) @ overlap_eigvecs.transpose())  # orthonormalization transformation matrix

    core_hamils = GetCoreHamiltonianIntegrals(basis_functions, atoms)   # core hamiltonian integrals
    repulsions = GetRepulsionIntegrals(basis_functions, atoms)          # two-electron repulsion integrals

    # estimate initial values of roothan equation coefficient matrix, with the approximation that fock matrix F = core hamiltonian matrix H
    orbital_energies, roothaan_coefs = np.linalg.eigh(np.linalg.inv(overlaps)*core_hamils)

    densities = GetDensityMatrix(roothaan_coefs, num_electrons//2)

    if verbose: print("Calculating optimized electron density matrix")
    iterations = 0
    while True:
        iterations += 1
        fock_matrix = GetFockMatrix(core_hamils, densities, repulsions)

        fock_orthonormalized = orthonormalization_matrix.transpose() @ fock_matrix @ orthonormalization_matrix
        orbital_energies, coefs_orthonormalized = np.linalg.eigh(fock_orthonormalized)
        roothaan_coefs = orthonormalization_matrix @ coefs_orthonormalized

        new_densities = GetDensityMatrix(roothaan_coefs, num_electrons//2)
        diff = np.linalg.norm(densities - new_densities)
        densities = new_densities.copy()

        if diff < epsilon: break
        if iterations > 1000:   # sometimes the densities matrix fails to converge and just alternates between two matrices instead, otherwise it definitely should have converged before hitting a thousand iterations
            print("SCF didn't converge, retry with better starting geometry")
            break
    
    V_NN = GetNuclearRepulsionEnergy(atoms)
    SCF_energy = V_NN + 0.5*np.sum(densities * (fock_matrix + core_hamils))

    if verbose: print("Finished SCF energy calculation")
    return SCF_energy


def GetSCFEnergyGradient(atoms, basis_set, num_electrons):  # analytic gradient would be better but the references describing it are horribly unreadable
    if verbose: print("Starting SCF gradient calculation")  # besides, what I did find seemed like it would require almost this many computations anyway
    grad = np.zeros(3*len(atoms))                           

    for i, atom in enumerate(atoms):
        atom.pos += [epsilon/2, 0, 0]
        e1 = GetSCFEnergy(atoms, basis_set, num_electrons)
        if verbose: print("SCF energy at x+dx", e1)
        atom.pos -= [epsilon, 0, 0]
        e2 = GetSCFEnergy(atoms, basis_set, num_electrons)
        if verbose: print("SCF energy at x-dx", e2)
        grad[3*i] = (e1 - e2)/epsilon    # central difference numerical derivative
        atom.pos += [epsilon/2, 0, 0]

        atom.pos += [0, epsilon/2, 0]
        e1 = GetSCFEnergy(atoms, basis_set, num_electrons)
        if verbose: print("SCF energy at y+dy", e1)
        atom.pos -= [0, epsilon, 0]
        e2 = GetSCFEnergy(atoms, basis_set, num_electrons)
        if verbose: print("SCF energy at y-dy", e2)
        grad[3*i+1] = (e1 - e2)/epsilon
        atom.pos += [0, epsilon/2, 0]

        atom.pos += [0, 0, epsilon/2]
        e1 = GetSCFEnergy(atoms, basis_set, num_electrons)
        if verbose: print("SCF energy at z+dz", e1)
        atom.pos -= [0, 0, epsilon]
        e2 = GetSCFEnergy(atoms, basis_set, num_electrons)
        if verbose: print("SCF energy at z-dz", e2)
        grad[3*i+2] = (e1 - e2)/epsilon
        atom.pos += [0, 0, epsilon/2]

    return grad

def OptimizeGeometry(atoms, basis_set, num_electrons):
    print("Starting geometry optimization")

    geom = np.zeros(3*len(atoms))   # geometry of the molecule   [x0, y0, z0, x1, y1, z1, x2, y2, z2, ...]
    for i, atom in enumerate(atoms):
        geom[3*i : 3*i+3] = atom.pos

    print("Calculating initial energy gradient")
    grad = GetSCFEnergyGradient(atoms, basis_set, num_electrons)
    new_grad = grad.copy()
    I = np.eye(3*len(atoms))                    # identity matrix
    inv_hessian = I/np.linalg.norm(grad)        # initial inverse hessian, diagonal matrix of just 1/||gradient||

    print("Initial geometry, gradient, and Hessian prepared")

    geometries = [geom.copy()]  # lists of previous and current geometries, gradients, and error vectors to be used for GDIIS
    gradients = [grad.copy()]
    error_vectors = []

    restrict_step_size = True   # set to True to cap the magnitude of the geometry step depending on how big the previous step was
    step_size_limit = 3         # step size can be no longer than step_size_limit * prev_step_size, prevents any one single step from blowing up
    prev_step_size = np.infty

    update_hessian = False      # set to True to update the inverse hessian according to the BFGS scheme, otherwise it just sticks with the initial diagonal matrix

    forget_old_steps = True     # set to True to only use the most recent steps to predict next step, instead of keeping track of every step since the beginning
    saved_steps_limit = 10      # now many recent steps to remember

    geom_convergence_epsilon = 0.05    # if the energy gradient magnitude is smaller than this, say we're done
    iterations = 0
    # modified GDIIS method to iteratively try to find geometry that minimizes SCF energy
    # https://schlegelgroup.wayne.edu/Pub_folder/251.pdf
    while np.linalg.norm(grad) > geom_convergence_epsilon:   # when gradient is (close to) 0, we've found a local minimum (or a saddle point)
        iterations += 1

        error_vectors.append(-inv_hessian@grad)  # new error vector

        error_dots_matrix = np.ones((len(geometries)+1,len(geometries)+1))            # matrix of inner products of every pair of previous error vectors, except the final row and final column are all ones with a zero on the final diagonal
        for idx, i in np.ndenumerate(error_dots_matrix[:-1,:-1]):
            error_dots_matrix[idx] = error_vectors[idx[0]] @ error_vectors[idx[1]]
        error_dots_matrix[-1,-1] = 0

        rhs_vector = np.zeros(len(geometries)+1)    # vector on the right hand side of the GDIIS coefficient matrix equation
        rhs_vector[-1] = 1                          # all zeros except the final element is a one

        GDIIS_coefs = np.linalg.solve(error_dots_matrix, rhs_vector)[:-1]    # coefficients that give the error vector of our current best geometry, last element is a lagrangian multiplier so we slice it out

        intermediate_geom = GDIIS_coefs @ geometries
        intermediate_grad = GDIIS_coefs @ gradients

        new_geom = intermediate_geom - inv_hessian @ intermediate_grad    # GDIIS step size
        delta_geom = new_geom - geom
        if restrict_step_size and np.linalg.norm(delta_geom) > step_size_limit * prev_step_size:               # cap out the step size at step_size_limit * prev_step_size
            delta_geom *= step_size_limit * prev_step_size/np.linalg.norm(delta_geom)
        prev_step_size = np.linalg.norm(delta_geom)
        geom += delta_geom

        for i, atom in enumerate(atoms):            # update the positions
            atom.pos = geom[3*i : 3*i+3]
        geometries.append(geom.copy())              # add this geometry to the history

        print(f"Iteration {iterations}: energy gradient magnitude {np.linalg.norm(grad):.2f}")
        print(f"gradient: {grad}")
        print(f"geometry: {geom}")

        new_grad = GetSCFEnergyGradient(atoms, basis_set, num_electrons)
        delta_grad = new_grad - grad
        grad = new_grad.copy()          # update the gradient
        gradients.append(grad.copy())   # add this gradient to the history

        if update_hessian:
            inv_hessian = ((I - np.outer(delta_geom, delta_grad)/(delta_grad@delta_geom))@inv_hessian@(I - np.outer(delta_grad, delta_geom)/(delta_grad@delta_geom))) + np.outer(delta_geom, delta_geom)/(delta_grad@delta_geom)        # bfgs update formula to update the inverse hessian
        if forget_old_steps and len(geometries)>=saved_steps_limit:    # only use the most recent steps to get next step (but it still uses the cumulative hessian)
            geometries.pop(0)
            gradients.pop(0)
            error_vectors.pop(0)

    return geom
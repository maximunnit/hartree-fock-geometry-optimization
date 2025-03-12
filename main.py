from basisset import *
from scf import *


# implementation of the closed-shell hartree-fock method to find energy-minimizing molecular geometries
# largely based off the process described in Ira Levine's book Quantum Chemistry, and Attila Szabo and Neil Ostlund's book Modern Quantum Chemistry


#basis_set = ReadBasisSetFile("STO-3G(H,He).txt")   # the ReadBasisSetFile function cannot read the combined s and p orbitals that some older basis sets like STO-3G have, but since H and He only have s-type basis functions it works fine reading those
#basis_set = ReadBasisSetFile("mini.txt")
basis_set = ReadBasisSetFile("def2-SV(P).txt")
#basis_set = ReadBasisSetFile("def2-TZVP.txt")

atoms_H2O = [Atom("H", [1.,0,-1.]), Atom("O", [0.,0.,0.]), Atom("H", [1.,0.,1.])]
atoms_H2 = [Atom("H", [-1.,0.,0.]), Atom("H", [1.,0.,0.])]
atoms_H3 = [Atom("H", [-1.,0.,0.]), Atom("H", [1.,0.,0.]), Atom("H", [0.,0.,1.])]
atoms_HeH = [Atom("He", [-1.,0.,0.]), Atom("H", [1.,0.,0.])]


start_time = timer()

num_electrons = sum([_.Z for _ in atoms_H2])
OptimizeGeometry(atoms_H3, basis_set, num_electrons)    # num_electrons can also be set manually if the molecule is an ion
print(f"optimized geometry:")
for a in atoms_H3:
    print(a)

print(f"total time elapsed: {timer()-start_time:.0f}s")
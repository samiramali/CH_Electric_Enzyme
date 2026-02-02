from prody import *
import os
from Bio.PDB import PDBParser, PDBIO

# ===========================
# Load structure
# ===========================
structure = parsePDB('1OGX.pdb')

# ===========================
# Build ANM model
# ===========================
anm = ANM('KSI')
anm.buildHessian(structure)
anm.calcModes(n_modes=20)

# First 3 lowest-frequency modes
modes = anm[:3]

# ===========================
# Convert ProDy AtomGroup → Biopython structure
# Needed to write PDBs
# ===========================
parser = PDBParser(QUIET=True)
bio = parser.get_structure("ksi", "1OGX.pdb")
atoms = [a for a in bio.get_atoms()]

# Output folder
os.makedirs("ensemble", exist_ok=True)
io = PDBIO()

print("Generating snapshots with ProDy sampleModes (ProDy 2.5.0)...")

# ===========================
# Generate 100 snapshots
# ===========================
for i in range(100):

    # Correct signature for ProDy 2.5.0:
    # sampleModes(modes, atoms=None, n_confs=1000, rmsd=1.0)
    # We pass:
    #   - modes   = selected ModeSet
    #   - atoms   = structure (ProDy AtomGroup)
    #   - n_confs = 1 conformation each iteration
    ens = sampleModes(modes, atoms=structure, n_confs=1, rmsd=1.0)

    # Get coordinates
    coords = ens.getCoordsets()[0]

    # Apply coordinates to Biopython atoms
    for atom, xyz in zip(atoms, coords):
        atom.set_coord(xyz)

    # Save as PDB
    io.set_structure(bio)
    io.save(f"ensemble/frame_{i:03d}.pdb")

print("Done. 100 snapshots saved to ./ensemble/")

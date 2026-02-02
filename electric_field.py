#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 19:19:40 2025

@author: samir
"""

import numpy as np
import os
import csv
from Bio.PDB import PDBParser

# ---------------------------
# Coulomb constant (kJ·Å·mol⁻¹·e⁻²)
# ---------------------------
k = 1389.354846

# ---------------------------
# Step 1: Identify the ligand and its C=O atoms in the reference PDB
# ---------------------------

parser = PDBParser(QUIET=True)
ref = parser.get_structure("ref", "1OGX.pdb")

ligand = None
for residue in ref.get_residues():
    hetflag = residue.id[0]
    if hetflag.strip():    # Hetero: ligand
        ligand = residue
        break

if ligand is None:
    raise RuntimeError("No ligand found in 1OGX.pdb!")

print("Found ligand:", ligand.get_resname())

# Identify carbonyl C–O bond in ligand
C_atom = None
O_atom = None

for atom in ligand.get_atoms():
    name = atom.get_name().upper()
    elem = atom.element

    if elem == "C" and "O" not in name:
        if C_atom is None:           # first carbon = carbonyl C
            C_atom = atom

    if elem == "O":
        if O_atom is None:           # first oxygen = carbonyl O
            O_atom = atom

print("Carbonyl atoms identified:", C_atom.get_name(), O_atom.get_name())

# ---------------------------
# Step 2: Define partial charges
# ---------------------------
def get_charge(atom):
    """Simple Amber-like partial charges"""
    e = atom.element

    if e == "O":
        return -0.55      # carbonyl-like O
    if e == "N":
        return -0.60
    if e == "C":
        return +0.55      # carbonyl C
    if e == "H":
        return +0.10
    if e == "S":
        return -0.50
    return 0.0            # fallback

# ---------------------------
# Step 3: Electric field function
# ---------------------------
def electric_field(q, r_charge, r_point):
    r = r_point - r_charge
    dist = np.linalg.norm(r)
    return k * q * r / (dist**3 + 1e-12)

# ---------------------------
# Step 4: Process ensemble snapshots
# ---------------------------

results = []

for fname in sorted(os.listdir("ensemble")):
    if not fname.endswith(".pdb"):
        continue

    path = f"ensemble/{fname}"
    structure = parser.get_structure("frame", path)

    # --- find ligand in each frame ---
    lig = None
    for residue in structure.get_residues():
        if residue.id[0].strip():
            lig = residue
            break
    if lig is None:
        print("Warning: no ligand in", fname)
        continue

    # --- find carbonyl C and O ---
    C = None
    O = None
    for atom in lig.get_atoms():
        ename = atom.element
        if ename == "C" and C is None:
            C = atom
        if ename == "O" and O is None:
            O = atom

    # Midpoint of C=O
    p = 0.5 * (C.coord + O.coord)

    # Compute total electric field vector at p
    E = np.zeros(3)

    atoms = [a for a in structure.get_atoms()]
    for atom in atoms:
        q = get_charge(atom)
        E += electric_field(q, atom.coord, p)

    # Project onto C→O bond direction
    bond_vec = O.coord - C.coord
    bond_unit = bond_vec / np.linalg.norm(bond_vec)
    E_proj = np.dot(E, bond_unit)

    results.append([fname, E_proj])

    print("Processed:", fname, "  E_proj =", E_proj)

# ---------------------------
# Step 5: Save results
# ---------------------------

with open("electric_field_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "E_proj"])
    w.writerows(results)

print("\nSaved electric_field_results.csv")

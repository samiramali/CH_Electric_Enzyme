#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 19:31:15 2025

@author: samir
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
from Bio.PDB import PDBParser

def get_ca_coords(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_path)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
    return np.array(coords)

def compute_Rs(ca_coords):
    N = len(ca_coords)
    Rs = []
    for s in range(1, N):
        dists = [np.linalg.norm(ca_coords[i] - ca_coords[i + s]) for i in range(N - s)]
        Rs.append(np.mean(dists))
    return Rs

def compute_CH_for_ensemble(ensemble_dir, output_file):

    print(f"\n=== Processing ensemble: {ensemble_dir} ===")
    
    pdb_files = sorted([f for f in os.listdir(ensemble_dir) if f.endswith(".pdb")])
    print(f"Found {len(pdb_files)} PDB snapshots.")

    if len(pdb_files) == 0:
        raise RuntimeError("No PDB files found in this ensemble directory!")

    Rs_matrix = []

    # Step 1: Extract Rs(s) from all snapshots
    for pdb_file in pdb_files:
        coords = get_ca_coords(os.path.join(ensemble_dir, pdb_file))
        Rs = compute_Rs(coords)
        Rs_matrix.append(Rs)

    # Align lengths
    min_len = min(len(r) for r in Rs_matrix)
    Rs_matrix = np.array([r[:min_len] for r in Rs_matrix])

    # Compute mean ⟨R(s)⟩
    mean_Rs = np.mean(Rs_matrix, axis=0)
    rel_Rs_matrix = Rs_matrix / mean_Rs  # relative R(s)

    # Step 2: Conformational Heterogeneity (CH) and relative CH
    CH = np.std(Rs_matrix, axis=0)
    rel_CH = np.std(rel_Rs_matrix, axis=0)

    s_vals = np.arange(1, len(mean_Rs)+1)

    # Step 3: Save to CSV
    df = pd.DataFrame({'s': s_vals, 'mean_Rs': mean_Rs})
    for i, fname in enumerate(pdb_files):
        df[f'snapshot_{i+1}'] = Rs_matrix[i]

    df['CH'] = CH
    df['rel_CH'] = rel_CH

    df.to_csv(output_file, index=False)
    print(f"Saved C.H. results → {output_file}")

    return s_vals, rel_CH


# --------------------
# RUN AS SCRIPT
# --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Cα-based C.H. for ensemble.")
    parser.add_argument("ensemble_dir", help="Path to folder with PDB snapshots")
    parser.add_argument("output_file", help="Output CSV file")

    args = parser.parse_args()

    compute_CH_for_ensemble(args.ensemble_dir, args.output_file)

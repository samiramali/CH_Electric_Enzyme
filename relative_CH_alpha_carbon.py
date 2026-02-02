#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 15:10:42 2025

@author: samir
"""

import numpy as np
import os
import pandas as pd
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt

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

# === SETTINGS ===
ensemble_dir = "protein_ensemble/"  # directory with PDB files (e.g., conf1.pdb, conf2.pdb...)
output_file = "Rs_data_IDP_relative_CH.csv"

pdb_files = sorted([f for f in os.listdir(ensemble_dir) if f.endswith(".pdb")])
print(f"Found {len(pdb_files)} structures.")

Rs_matrix = []

# Step 1–3: Extract ⟨R(s)⟩ from all conformers
for pdb_file in pdb_files:
    coords = get_ca_coords(os.path.join(ensemble_dir, pdb_file))
    Rs = compute_Rs(coords)
    Rs_matrix.append(Rs)

min_len = min(len(r) for r in Rs_matrix)
Rs_matrix = np.array([r[:min_len] for r in Rs_matrix])
 # shape: (n_structures, s_values)
mean_Rs = np.mean(Rs_matrix, axis=0)
rel_Rs_matrix = Rs_matrix / mean_Rs  # relative ⟨R(s)⟩

# Step 4: Compute standard deviations (C.H. and relative C.H.)
CH = np.std(Rs_matrix, axis=0)
rel_CH = np.std(rel_Rs_matrix, axis=0)

# Step 5: Save to CSV
s_vals = np.arange(1, len(mean_Rs)+1)
df = pd.DataFrame({'s': s_vals, 'mean_Rs': mean_Rs})
for i, fname in enumerate(pdb_files):
    df[f'snapshot_{i+1}'] = Rs_matrix[i]
df['CH'] = CH
df['rel_CH'] = rel_CH
df.to_csv(output_file, index=False)
print(f"Saved output to {output_file}")

# Step 6: Print relative CH values
print("\n=== Relative C.H. per sequence separation ===")
for s, val in zip(s_vals, rel_CH):
    print(f"s = {s:2d}: Relative C.H. = {val:.4f}")
    
    
    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Compute average distances over snapshots for each s
Rs_avg = np.mean(Rs_matrix, axis=0)  # shape: (num_s,)

# Compute pairwise differences between each s
pairwise_distances = np.abs(Rs_avg[:, None] - Rs_avg[None, :])  # shape: (num_s, num_s)

# Plot heatmap with longer distances darker
plt.figure(figsize=(8,6))
sns.heatmap(pairwise_distances, cmap="magma_r", cbar_kws={'label': 'Distance between s'})
plt.xlabel("s")
plt.ylabel("s")
plt.title("Pairwise distance heatmap (longer distance darker)", fontsize=16)
plt.tight_layout()
plt.show()

    
    

# Optional: Plot
plt.plot(s_vals, rel_CH, label="Relative C.H.", color='darkred')
plt.xlabel("Sequence separation (s)")
plt.ylabel("Relative C.H.")
plt.title("Relative Conformational Heterogeneity for IDP Ensemble")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

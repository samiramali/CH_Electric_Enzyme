#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 19:57:31 2025

@author: samir
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# Load data
df_low = pd.read_csv("CH_low.csv")
df_high = pd.read_csv("CH_high.csv")

# Extract only columns with snapshot data
low_Rs = df_low.filter(regex="snapshot_").values   # shape (30, num_s)
high_Rs = df_high.filter(regex="snapshot_").values # shape (30, num_s)

num_s = low_Rs.shape[1]
p_values = []

# Compute Mann–Whitney U p-value for each sequence separation s
for s in range(num_s):
    stat, p = mannwhitneyu(high_Rs[:, s], low_Rs[:, s], alternative='two-sided')
    p_values.append(p)

s_vals = df_low["s"].values

# Save p-values
pd.DataFrame({"s": s_vals, "p_value": p_values}).to_csv("CH_pvalues.csv", index=False)
print("Saved p-values → CH_pvalues.csv")

# Plot p-value vs s
plt.figure(figsize=(9,5))
plt.plot(s_vals, p_values, color="black")
plt.axhline(0.05, color="red", linestyle="--", label="p = 0.05")
plt.xlabel("Sequence separation (s)")
plt.ylabel("p-value")
plt.title("Statistical Significance of CH Differences (High vs Low EF)")
plt.yscale("log")  # better visualization
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("CH_pvalue_curve.png", dpi=300)
plt.show()

print("Saved p-value plot → CH_pvalue_curve.png")

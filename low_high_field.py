#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 19:23:05 2025

@author: samir
"""

import pandas as pd
import os
import shutil

# Load electric field results
df = pd.read_csv("electric_field_results.csv")

# Sort by field strength
df_sorted = df.sort_values("E_proj")

# Define fraction for top and bottom groups
fraction = 0.30   # top 30%, bottom 30%

n = len(df_sorted)
low_n = int(fraction * n)
high_n = int(fraction * n)

low = df_sorted.head(low_n)
high = df_sorted.tail(high_n)

# Create folders
os.makedirs("ensemble_low", exist_ok=True)
os.makedirs("ensemble_high", exist_ok=True)

# Copy files
for f in low["frame"]:
    shutil.copy(f"ensemble/{f}", f"ensemble_low/{f}")

for f in high["frame"]:
    shutil.copy(f"ensemble/{f}", f"ensemble_high/{f}")

print("Done!")
print(f"Low-field snapshots:  {len(low)}")
print(f"High-field snapshots: {len(high)}")

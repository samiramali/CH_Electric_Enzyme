#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 19:36:40 2025

@author: samir
"""

import pandas as pd
import matplotlib.pyplot as plt

df_low = pd.read_csv("CH_low.csv")
df_high = pd.read_csv("CH_high.csv")

plt.figure(figsize=(8,5))
plt.plot(df_low["s"], df_low["rel_CH"], label="Low-field ensemble", color='blue')
plt.plot(df_high["s"], df_high["rel_CH"], label="High-field ensemble", color='red')

plt.xlabel("Sequence separation (s)")
plt.ylabel("Relative C.H.")
plt.title("High-field vs Low-field Conformational Heterogeneity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("CH_field_comparison.png", dpi=300)
plt.show()

print("Saved plot → CH_field_comparison.png")

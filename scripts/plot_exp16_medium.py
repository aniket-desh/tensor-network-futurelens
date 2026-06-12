#!/usr/bin/env python
r"""Exp 16E figure: shared-recipe MPS gap vs GPT-2 medium layer."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs" / "16_matrix_product_physics_sprint" / "figures"
gaps = json.load(open(ROOT / "results/runs/gpt2med_exp16_layers/layer_gaps.json"))

fig, ax = plt.subplots(figsize=(6.2, 4.0))
Ls = sorted(int(k) for k in gaps)
mu = [gaps[str(L)]["gap_mean"] for L in Ls]
sd = [gaps[str(L)]["gap_sd"] for L in Ls]
ax.errorbar(Ls, mu, yerr=sd, fmt="-o", color="tab:red", capsize=4, lw=2)
for L in Ls:
    for g in gaps[str(L)]["per_seed"]:
        ax.plot([L], [g], ".", color="tab:red", alpha=0.45, ms=5)
ax.axhline(0, color="gray", lw=1)
ax.set_xticks(Ls)
ax.set_xlabel("GPT-2 medium layer (of 24)")
ax.set_ylabel("MPS − best baseline (test top-1, n=8)")
ax.set_title("Medium-scale gap vs layer (shared recipe, 4 seeds ± sd)\n"
             "largest early (L8), fading to tie deeper — and recipe-conditional")
fig.tight_layout()
fig.savefig(FIGDIR / "fig_medium_layers.png", dpi=150)
print(f"wrote {FIGDIR/'fig_medium_layers.png'}")

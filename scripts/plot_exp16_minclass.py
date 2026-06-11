#!/usr/bin/env python
r"""Exp 16A figure: minimal matrix-product function class — gap vs trained MPS-D16."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs" / "16_matrix_product_physics_sprint" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
rows = json.load(open(ROOT / "results/runs/gpt2_exp14_seeds/mech_table_n8.json"))

ORDER = [
    ("mps_D16", "MPS D16 (trained, ref)", "tab:red"),
    ("mps_D16_frozen", "FROZEN random cores", "darkred"),
    ("mps_D16_rank2", "rank-2 cores", "firebrick"),
    ("mps_D16_shuf", "sites shuffled", "indianred"),
    ("mps_D8", "D8 (482k params)", "lightcoral"),
    ("mps_D16_frozenorth", "frozen orthogonal cores", "salmon"),
    ("mps_D16_fixedphi", "frozen PCA φ", "rosybrown"),
    ("mps_D16_sym4", "symmetrized (4 orders)", "sienna"),
    ("mps_diag_D16", "diagonal (commuting) D16", "peru"),
    ("multpool", "bond-free product ×256", "tab:brown"),
    ("mlp", "MLP", "tab:blue"),
    ("bilinear", "bilinear", "tab:green"),
    ("conv1d", "conv1d", "tab:orange"),
    ("dilatedconv", "dilated conv (multiscale)", "gold"),
    ("treepool", "tree pooling (multiscale)", "olive"),
    ("attention", "attention", "tab:purple"),
]

fig, ax = plt.subplots(figsize=(9.5, 5.2))
ys = np.arange(len(ORDER))[::-1]
for y, (key, label, color) in zip(ys, ORDER):
    r = rows[key]
    d, (lo, hi) = r["delta_vs_mps"], r["boot95"]
    ax.barh(y, d, xerr=[[d - lo], [hi - d]], color=color, height=0.65, capsize=3)
ax.axvline(0, color="k", lw=1)
ax.axvspan(-0.0005, 0.0005, color="tab:red", alpha=0.08)
ax.set_yticks(ys)
ax.set_yticklabels([label for _, label, _ in ORDER], fontsize=8.5)
ax.set_xlim(-0.009, 0.002)
ax.set_xlabel("test top-1 difference vs trained MPS-D16 (n=8, 4 seeds, paired cluster-boot 95% CI)")
ax.set_title("Minimal structure: frozen random non-commuting cores + learned φ suffice;\n"
             "commuting / fixed-φ / order-mixing / multiscale variants all fail")
fig.tight_layout()
fig.savefig(FIGDIR / "fig_minclass.png", dpi=150)
print(f"wrote {FIGDIR/'fig_minclass.png'}")

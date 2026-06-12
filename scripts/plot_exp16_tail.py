#!/usr/bin/env python
r"""Exp 16C figure: tail-weighted training (alpha=1, n=32) — shared lr vs tuned MLP."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs" / "16_matrix_product_physics_sprint" / "figures"

# (label, overall, tail-half, color)  — from results/runs/gpt2_exp16_tail
DATA = [
    ("bilinear\n@shared lr", 0.0780, 0.0767, "tab:green"),
    ("MLP\n@shared lr", 0.0811, 0.0786, "tab:blue"),
    ("MPS-D16\n@shared lr", 0.0877, 0.0870, "tab:red"),
    ("MLP\n@3e-4", 0.0877, 0.0866, "royalblue"),
    ("MLP\n@1e-4 (tuned)", 0.0897, 0.0890, "navy"),
]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(len(DATA))
ax.bar(x - 0.18, [d[1] for d in DATA], width=0.36, color=[d[3] for d in DATA])
ax.bar(x + 0.18, [d[2] for d in DATA], width=0.36,
       color=[d[3] for d in DATA], alpha=0.55)
ax.set_xticks(x)
ax.set_xticklabels([d[0] for d in DATA], fontsize=8.5)
ax.set_ylim(0.070, 0.0915)
ax.set_ylabel("test top-1 (n=32, tail-weighted KL, 2 seeds)")
ax.set_title("Tail-weighted objective: the MPS 'tail advantage' also vanishes\n"
             "once the MLP gets its learning rate")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="gray", label="all 32 positions"),
                   Patch(color="gray", alpha=0.55, label="tail half (s=17–32)")],
          fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "fig_tail.png", dpi=150)
print(f"wrote {FIGDIR/'fig_tail.png'}")

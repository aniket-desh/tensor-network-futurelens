#!/usr/bin/env python
r"""Exp 16D figure: de-persisted bulk correlation decay — power-law vs exponential."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "runs" / "gpt2_exp16_powerlaw"
FIGDIR = ROOT / "docs" / "16_matrix_product_physics_sprint" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

data = json.load(open(OUTDIR / "powerlaw_vs_exp.json"))
rows = {(r["layer"], r["b"]): r for r in data["rows"]}
curves = data["curves"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
cmap = plt.cm.viridis(np.linspace(0, 0.85, 4))
for (ax, scale, title) in ((axes[0], "loglog", "log–log: straight ⇒ power law"),
                           (axes[1], "semilogy", "log–linear: straight ⇒ exponential")):
    for c, b in zip(cmap, (1, 2, 4, 8)):
        key = f"L6_b{b}"
        y = np.array(curves[key])
        ds = np.arange(len(y))
        m = ds >= 2
        r = rows[(6, b)]
        lab = f"b={b} (powR²={r['pow']['r2']:.2f}, expR²={r['exp']['r2']:.2f})"
        if scale == "loglog":
            ax.loglog(ds[m], y[m], "-o", ms=3, color=c, label=lab)
        else:
            ax.semilogy(ds[m], y[m], "-o", ms=3, color=c, label=lab)
    ax.set_xlabel("lag Δ (block units)")
    ax.set_ylabel("‖P⊥ Ĉ(Δ) P⊥‖ (normalized)")
    ax.set_title(title)
    ax.legend(fontsize=7)
fig.suptitle("GPT-2 small layer 6 — bulk correlations after persistent-subspace removal", y=1.02)
fig.tight_layout()
fig.savefig(FIGDIR / "fig_powerlaw.png", dpi=150, bbox_inches="tight")
print(f"wrote {FIGDIR/'fig_powerlaw.png'}")

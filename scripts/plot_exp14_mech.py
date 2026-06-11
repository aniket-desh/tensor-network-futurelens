#!/usr/bin/env python
r"""Exp 14 mechanism figure (fig4): shuffle/const ablation bars + bond-dimension curve."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs" / "12_mythos_sprint" / "figures"
rows = json.load(open(ROOT / "results/runs/gpt2_exp14_seeds/mech_table_n8.json"))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

# ---- panel A: ablations + baselines, sorted bars ---------------------------
order = ["mps_D16", "mps_D16_shuf", "mps_D16_noconst", "mlp", "bilinear", "mlp_shuf",
         "conv1d", "attention"]
labels = {"mps_D16": "MPS D16", "mps_D16_shuf": "MPS D16\nsites SHUFFLED",
          "mps_D16_noconst": "MPS D16\nno const", "mlp_shuf": "MLP\nsites shuffled",
          "mlp": "MLP", "bilinear": "bilinear", "conv1d": "conv1d",
          "attention": "attention"}
xs = np.arange(len(order))
means = [rows[m]["mean"] for m in order]
sds = [rows[m]["seed_sd"] for m in order]
cols = ["tab:red", "tab:red", "salmon", "tab:blue", "tab:green", "tab:blue",
        "tab:orange", "tab:purple"]
alphas = [1, 0.55, 0.4, 0.9, 0.9, 0.45, 0.9, 0.9]
for x, mu, sd, c, a in zip(xs, means, sds, cols, alphas):
    axes[0].bar(x, mu, yerr=sd, color=c, alpha=a, width=0.7, capsize=3)
axes[0].set_xticks(xs)
axes[0].set_xticklabels([labels[m].replace("\n", " ") for m in order], fontsize=7.5,
                        rotation=18, ha="right")
axes[0].set_ylim(0.088, 0.101)
axes[0].set_ylabel("test top-1 (n=8, mean over 4 seeds ± sd)")
axes[0].set_title("Destroying chain order does NOT remove the MPS edge")
axes[0].axhline(rows["mps_D16"]["mean"], color="tab:red", lw=0.8, ls=":")

# ---- panel B: bond-dimension curve with params -----------------------------
Ds = [2, 4, 8, 16, 32]
mu = [rows[f"mps_D{D}"]["mean"] for D in Ds]
sd = [rows[f"mps_D{D}"]["seed_sd"] for D in Ds]
params = {2: "82k", 4: "162k", 8: "482k", 16: "1.76M", 32: "6.88M"}
axes[1].errorbar(Ds, mu, yerr=sd, fmt="-o", color="tab:red", capsize=3, label="MPS(D)")
for D, m in zip(Ds, mu):
    axes[1].annotate(params[D], (D, m), textcoords="offset points", xytext=(2, -14),
                     fontsize=7, color="gray")
best_base = max(rows[b]["mean"] for b in ("mlp", "conv1d", "bilinear", "attention"))
axes[1].axhline(best_base, color="tab:blue", ls="--", lw=1.2,
                label=f"best baseline (MLP, 1.83M params)")
axes[1].set_xscale("log", base=2)
axes[1].set_xticks(Ds)
axes[1].set_xticklabels([str(D) for D in Ds])
axes[1].set_xlabel("bond dimension D")
axes[1].set_ylabel("test top-1 (n=8)")
axes[1].set_title("Saturates at D≈8 (482k params); D=32 overfits")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "fig4_mechanism.png", dpi=150)
print(f"wrote {FIGDIR/'fig4_mechanism.png'}")

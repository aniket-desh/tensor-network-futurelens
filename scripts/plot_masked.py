#!/usr/bin/env python
r"""Plot Exp 04 (B5 masked-MPS completion) vs B4 readout and MLP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tn_futurelens.utils.plotting import save_fig, set_style

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs" / "gpt2_masked_mps"
FIGDIR = ROOT / "docs" / "04_masked_mps" / "figures"


def best(results, prefix, metric):
    cand = [r for r in results if r["name"] == prefix or r["name"].startswith(prefix + "_D")]
    return min(cand, key=lambda r: r[metric]) if cand else None


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    layers = {}
    for p in sorted(RUNS.glob("layer_*.json")):
        d = json.load(open(p)); layers[d["layer"]] = d

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    conds = ["mlp", "b4_readout", "b5_masked"]
    pretty = {"mlp": "MLP", "b4_readout": "B4 readout", "b5_masked": "B5 masked"}
    colors = {"mlp": "#f57c00", "b4_readout": "#1f77b4", "b5_masked": "#08306b"}
    metrics = [("nmse_mean", "NMSE", (0.78, 0.95)), ("kl_mean", "teacher KL", (3.3, 3.95)),
               ("top1_mean", "top-1 agreement", (0.08, 0.12))]
    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        width = 0.38
        for li, (L, d) in enumerate(layers.items()):
            vals = [best(d["results"], c, "nmse_mean")[metric] for c in conds]
            x = np.arange(len(conds)) + (li - 0.5) * width
            ax.bar(x, vals, width=width, color=[colors[c] for c in conds],
                   alpha=0.6 if li else 1.0, edgecolor="k", linewidth=0.4,
                   label=f"layer {L}")
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels([pretty[c] for c in conds], fontsize=9)
        ax.set_ylabel(ylabel); ax.set_ylim(*ylim); ax.set_title(ylabel)
    axes[0].legend(fontsize=9, title="solid=L6, faded=L12", title_fontsize=8)
    fig.suptitle("B5 masked-MPS completion ≈ B4 readout (marginally better KL/top-1); "
                 "both beat MLP+frozen-PCA", y=1.03, fontsize=11)
    save_fig(fig, FIGDIR / "fig1_b4_vs_b5.png")
    print(f"wrote {FIGDIR}")


if __name__ == "__main__":
    main()

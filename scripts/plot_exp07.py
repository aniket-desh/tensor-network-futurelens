#!/usr/bin/env python
r"""Plot Exp 07: strong baselines x {MSE, KL} objective."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tn_futurelens.utils.plotting import save_fig, set_style
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs" / "gpt2_exp07"
FIGDIR = ROOT / "docs" / "07_strong_baselines_kl" / "figures"
LAYER = 6
ORDER = ["mlp", "attention", "conv1d", "bilinear", "mps_const_D16"]
PRETTY = {"mlp": "MLP", "attention": "attention", "conv1d": "conv1d",
          "bilinear": "bilinear", "mps_const_D16": "MPS+const"}


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    data = {obj: {r["name"]: r for r in json.load(open(RUNS / f"layer_{LAYER}_{obj}.json"))["results"]}
            for obj in ["mse", "kl"]}
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    metrics = [("nmse_mean", "validation NMSE", "lower=better"),
               ("kl_mean", "teacher KL", "lower=better"),
               ("top1_mean", "top-1 agreement", "higher=better")]
    x = np.arange(len(ORDER)); w = 0.38
    for ax, (metric, title, note) in zip(axes, metrics):
        for k, (obj, color) in enumerate([("mse", "#90caf9"), ("kl", "#08306b")]):
            vals = [data[obj][m][metric] for m in ORDER]
            ax.bar(x + (k - 0.5) * w, vals, w, color=color, edgecolor="k", lw=0.4,
                   label=f"{obj} objective")
        ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in ORDER], rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{title}\n({note})")
    axes[0].legend(fontsize=9)
    fig.suptitle("Layer 6: training objective matters more than architecture — KL training "
                 "improves top-1/KL across the board; no model decisively wins", y=1.03, fontsize=11)
    save_fig(fig, FIGDIR / "fig1_objective_x_model.png")
    print(f"wrote {FIGDIR}")


if __name__ == "__main__":
    main()

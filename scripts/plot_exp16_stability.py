#!/usr/bin/env python
r"""Exp 16B figures: lr-sensitivity curves + seed scatter, and data-size curves."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs" / "16_matrix_product_physics_sprint" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
COLORS = {"mlp": "tab:blue", "conv1d": "tab:orange", "bilinear": "tab:green",
          "mps_D8": "lightcoral", "mps_D16": "tab:red"}

table = json.load(open(ROOT / "results/runs/gpt2_exp16_lrgrid/stability_table.json"))

# ---- fig A: top-1 vs lr (mean over seeds, ± min/max), per model ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axes[0]
for m, rec in table.items():
    by_lr = defaultdict(list)
    for k, v in rec["cells"].items():
        lr = float(k.split("|")[0])
        by_lr[lr].append(v)
    lrs = sorted(by_lr)
    mu = [np.mean(by_lr[lr]) for lr in lrs]
    lo = [np.min(by_lr[lr]) for lr in lrs]
    hi = [np.max(by_lr[lr]) for lr in lrs]
    kw = dict(lw=2.5, zorder=5) if m == "mps_D16" else dict(lw=1.3, alpha=0.9)
    ax.plot(lrs, mu, "-o", color=COLORS[m], label=m, **kw)
    ax.fill_between(lrs, lo, hi, color=COLORS[m], alpha=0.13)
ax.set_xscale("log")
ax.set_xlabel("learning rate")
ax.set_ylabel("test top-1 (n=8, mean over seeds ± range)")
ax.set_title("lr response: dense baselines peak ABOVE the MPS plateau at low lr;\n"
             "the shared lr 1.5e-3 sat in the MPS sweet spot")
ax.legend(fontsize=8)

# ---- fig B: seed scatter at the shared lr (all seeds incl. 4-7) ----------------
ax = axes[1]
order = ["mps_D16", "mps_D8", "mlp", "bilinear", "conv1d"]
for i, m in enumerate(order):
    vals = [v for k, v in table[m]["cells"].items()
            if abs(float(k.split("|")[0]) - 1.5e-3) < 1e-9]
    x = np.full(len(vals), i) + np.random.default_rng(0).normal(0, 0.04, len(vals))
    ax.plot(x, vals, "o", color=COLORS[m], alpha=0.75, ms=6)
    ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color=COLORS[m], lw=2.5)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, fontsize=9)
ax.set_ylabel("test top-1 (n=8)")
ax.set_title("seed scatter at shared lr 1.5e-3 (8 seeds where available)")
fig.tight_layout()
fig.savefig(FIGDIR / "fig_stability.png", dpi=150)
print(f"wrote {FIGDIR/'fig_stability.png'}")

# ---- fig C: data-size curves ----------------------------------------------------
ds_dir = ROOT / "results/runs/gpt2_exp16_datasize"
if any(ds_dir.glob("results_*.json")):
    by = defaultdict(lambda: defaultdict(list))      # model -> n_train -> [vals]
    for f in sorted(ds_dir.glob("results_*.json")):
        d = json.load(open(f))
        sz = d.get("n_train_used")
        for r in d["runs"]:
            by[r["model"]][sz].append(r["test_top1_mean"])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m, dd in by.items():
        xs = sorted(dd)
        mu = [np.mean(dd[x]) for x in xs]
        lo = [np.min(dd[x]) for x in xs]
        hi = [np.max(dd[x]) for x in xs]
        kw = dict(lw=2.5, zorder=5) if m == "mps_D16" else dict(lw=1.3)
        ax.plot(xs, mu, "-o", color=COLORS.get(m, "gray"),
                label=f"{m} (its best lr)", **kw)
        ax.fill_between(xs, lo, hi, color=COLORS.get(m, "gray"), alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("training windows")
    ax.set_ylabel("test top-1 (n=8)")
    ax.set_title("Data-size scaling (same 50k-window test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_datasize.png", dpi=150)
    print(f"wrote {FIGDIR/'fig_datasize.png'}")

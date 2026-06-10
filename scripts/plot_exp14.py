#!/usr/bin/env python
r"""Exp 14 figures.

fig1: MPS − baseline test top-1 gap vs horizon n (per-seed points, seed-mean line,
      cluster-bootstrap 95% CI band vs each baseline).
fig2: absolute test top-1 vs n, all models, seed mean ± min/max.

Usage: python scripts/exp14_stats.py && python scripts/plot_exp14.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs" / "12_mythos_sprint" / "figures"
BASE_COLORS = {"mlp": "tab:blue", "conv1d": "tab:orange", "bilinear": "tab:green",
               "attention": "tab:purple", "mps_D16": "tab:red"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="gpt2_exp14_seeds")
    ap.add_argument("--mps", default="mps_D16")
    args = ap.parse_args()
    OUTDIR = ROOT / "results" / "runs" / args.outdir
    FIGDIR.mkdir(parents=True, exist_ok=True)
    stats = json.load(open(OUTDIR / "stats_summary.json"))
    runs = []
    for f in sorted(OUTDIR.glob("results_*.json")):
        runs += json.load(open(f))["runs"]

    ns = sorted(int(n) for n in stats)
    baselines = [b for b in ("mlp", "conv1d", "bilinear", "attention")
                 if b in stats[str(ns[0])]["rows"]]

    # ---- fig 1: gap vs n --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for b in baselines:
        mu = [stats[str(n)]["rows"][b]["gap_mean"] for n in ns]
        lo = [stats[str(n)]["rows"][b]["boot95"][0] for n in ns]
        hi = [stats[str(n)]["rows"][b]["boot95"][1] for n in ns]
        ax.plot(ns, mu, "-o", color=BASE_COLORS[b], label=f"vs {b}")
        ax.fill_between(ns, lo, hi, color=BASE_COLORS[b], alpha=0.15)
        for n in ns:
            for g in stats[str(n)]["rows"][b]["per_seed"]:
                ax.plot([n], [g], ".", color=BASE_COLORS[b], alpha=0.5, ms=4)
    gb = [stats[str(n)]["gap_vs_best_test"] for n in ns]
    ax.plot(ns, gb, "k--s", lw=2, label="vs best baseline (test)")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("horizon n (future sites)")
    ax.set_ylabel(f"{args.mps} − baseline   (test top-1)")
    ax.set_title("Held-out-test MPS gap vs horizon (multi-seed; band = cluster-boot 95% CI)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_gap_vs_horizon.png", dpi=150)

    # ---- fig 2: absolute top-1 vs n ---------------------------------------
    by = defaultdict(lambda: defaultdict(list))   # model -> n -> [seed means]
    for r in runs:
        by[r["model"]][r["n"]].append(r["test_top1_mean"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mdl, d in by.items():
        xs = sorted(d)
        mu = np.array([np.mean(d[n]) for n in xs])
        lo = np.array([np.min(d[n]) for n in xs])
        hi = np.array([np.max(d[n]) for n in xs])
        kw = dict(lw=2.5, zorder=5) if mdl == args.mps else dict(lw=1.2, alpha=0.85)
        ax.plot(xs, mu, "-o", color=BASE_COLORS.get(mdl, "gray"), label=mdl, **kw)
        ax.fill_between(xs, lo, hi, color=BASE_COLORS.get(mdl, "gray"), alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("horizon n (future sites)")
    ax.set_ylabel("test top-1 agreement (mean over horizons)")
    ax.set_title("Absolute completion accuracy vs horizon (seed mean ± range)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_top1_vs_horizon.png", dpi=150)
    print(f"wrote {FIGDIR}/fig1_gap_vs_horizon.png, fig2_top1_vs_horizon.png")


if __name__ == "__main__":
    main()

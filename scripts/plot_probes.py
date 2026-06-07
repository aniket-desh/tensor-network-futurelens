#!/usr/bin/env python
r"""Plot GPT-2 completion-probe results (briefing §7 scaling diagnostics).

Reads results/runs/gpt2_probes/layer_*.json and produces the key scientific plots:
NMSE/KL vs bond dimension D, and KL vs horizon s, with parameter-matched baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tn_futurelens.utils.plotting import save_fig, set_style

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs" / "gpt2_probes"
FIGDIR = ROOT / "docs" / "02_gpt2_baselines_mps" / "figures"
TABLE = ROOT / "results" / "tables" / "probe_results_gpt2.csv"


def load_layers():
    out = {}
    for p in sorted(RUNS.glob("layer_*.json")):
        d = json.load(open(p))
        out[d["layer"]] = d
    return out


def mps_curve(res, key):
    pts = [(int(r["name"].split("D")[-1]), r[key]) for r in res if r["name"].startswith("mps")]
    pts.sort()
    return [d for d, _ in pts], [v for _, v in pts]


def baseline(res, name, key):
    for r in res:
        if r["name"] == name:
            return r[key]
    return None


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    layers = load_layers()
    set_style()
    colors = {6: "#1f77b4", 12: "#d62728", 8: "#2ca02c"}

    # CSV
    import csv
    with open(TABLE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "model", "n_params", "nmse_mean", "kl_mean", "top1_mean"])
        for L, d in layers.items():
            for r in d["results"]:
                w.writerow([L, r["name"], r["n_params"], r["nmse_mean"], r["kl_mean"], r["top1_mean"]])

    # Fig 1: NMSE and KL vs D
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for L, d in layers.items():
        col = colors.get(L, "k")
        Ds, nmse = mps_curve(d["results"], "nmse_mean")
        _, kl = mps_curve(d["results"], "kl_mean")
        axes[0].plot(Ds, nmse, "o-", color=col, label=f"MPS, layer {L}")
        axes[1].plot(Ds, kl, "o-", color=col, label=f"MPS, layer {L}")
        for bl, ls in [("multisite_linear", "--"), ("mlp_h256", ":")]:
            axes[0].axhline(baseline(d["results"], bl, "nmse_mean"), color=col, ls=ls, alpha=0.55)
            axes[1].axhline(baseline(d["results"], bl, "kl_mean"), color=col, ls=ls, alpha=0.55)
    axes[0].set_xlabel("bond dimension $D$"); axes[0].set_ylabel("validation NMSE (mean over horizons)")
    axes[0].set_title("Completion NMSE vs $D$")
    axes[1].set_xlabel("bond dimension $D$"); axes[1].set_ylabel("teacher KL (mean over horizons)")
    axes[1].set_title("Teacher KL vs $D$")
    axes[0].legend(fontsize=8, title="dashed=linear, dotted=MLP", title_fontsize=7)
    save_fig(fig, FIGDIR / "fig1_nmse_kl_vs_D.png")

    # Fig 2: KL and top-1 vs horizon (best MPS vs baselines), per layer
    fig, axes = plt.subplots(1, len(layers), figsize=(6.2 * len(layers), 4.4), squeeze=False)
    for ax, (L, d) in zip(axes[0], layers.items()):
        res = d["results"]
        n = d["n"]
        hs = list(range(1, n + 1))
        # best MPS by nmse
        mps = [r for r in res if r["name"].startswith("mps")]
        best = min(mps, key=lambda r: r["nmse_mean"])
        for r, c, mk in [(baseline_rec(res, "multisite_linear"), "#7f7f7f", "s"),
                         (baseline_rec(res, "mlp_h256"), "#ff7f0e", "^"),
                         (best, colors.get(L, "k"), "o")]:
            ax.plot(hs, r["kl_per_h"], mk + "-", color=c, label=r["name"])
        ax.set_xlabel("horizon $s$ (tokens ahead)"); ax.set_ylabel("teacher KL")
        ax.set_title(f"Layer {L}: KL vs horizon")
        ax.set_xticks(hs); ax.legend(fontsize=8)
    save_fig(fig, FIGDIR / "fig2_kl_vs_horizon.png")
    print(f"wrote figures to {FIGDIR} and table {TABLE}")


def baseline_rec(res, name):
    for r in res:
        if r["name"] == name:
            return r
    return res[0]


if __name__ == "__main__":
    main()

#!/usr/bin/env python
r"""Plot Exp 11: GPT-2 small vs medium structure replication + predictive tie."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tn_futurelens.utils.plotting import save_fig, set_style
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MED = ROOT / "results" / "runs" / "gpt2_medium"
SMALL_CORR = ROOT / "results" / "runs" / "gpt2_correlations" / "results.json"
SMALL_MODES = ROOT / "results" / "runs" / "gpt2_exp06" / "gpt2.json"
FIGDIR = ROOT / "docs" / "11_scale_gpt2_medium" / "figures"


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    med = json.load(open(MED / "corr.json"))
    med_rows = med["rows"]; med_L = med["n_layers"]
    small_corr = {r["layer"]: r for r in json.load(open(SMALL_CORR))["rows"]}
    small_modes = {r["layer"]: r for r in json.load(open(SMALL_MODES))["rows"]}
    pred = json.load(open(MED / "predict_layer_12.json"))

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # eff modes vs relative depth
    sm_x = [l / 12 for l in small_modes]; sm_y = [small_modes[l]["eff_modes_sv"] for l in small_modes]
    md_x = [r["layer"] / med_L for r in med_rows]; md_y = [r["eff_modes"] for r in med_rows]
    axes[0].plot(sm_x, sm_y, "o-", color="#1f77b4", label="GPT-2 small")
    axes[0].plot(md_x, md_y, "s-", color="#d62728", label="GPT-2 medium")
    axes[0].set_xlabel("relative depth (layer / n_layers)"); axes[0].set_ylabel("effective modes")
    axes[0].set_title("Correlation mode count grows with depth\n(replicates across scale)")
    axes[0].legend(fontsize=9)

    # bulk xi + persistent frac vs relative depth (medium)
    axes[1].plot([r["layer"] / med_L for r in med_rows], [r["bulk_xi"] for r in med_rows],
                 "o-", color="#08306b", label="bulk ξ")
    axes[1].set_xlabel("relative depth"); axes[1].set_ylabel("bulk ξ (tokens)", color="#08306b")
    axes[1].tick_params(axis="y", labelcolor="#08306b")
    ax1b = axes[1].twinx()
    ax1b.plot([r["layer"] / med_L for r in med_rows], [r["persistent_frac"] for r in med_rows],
              "s--", color="#d62728", label="persistent frac")
    ax1b.set_ylabel("persistent fraction", color="#d62728"); ax1b.tick_params(axis="y", labelcolor="#d62728")
    axes[1].set_title("GPT-2 medium: finite-ξ bulk +\npersistent subspace grows with depth")

    # predict bars at L12
    names = [r["name"] for r in pred["results"]]
    nmse = [r["nmse_mean"] for r in pred["results"]]
    top1 = [r["top1_mean"] for r in pred["results"]]
    x = np.arange(len(names)); w = 0.35
    axes[2].bar(x - w / 2, nmse, w, color="#1f77b4", edgecolor="k", lw=0.4, label="NMSE")
    ax2b = axes[2].twinx()
    ax2b.bar(x + w / 2, top1, w, color="#ff7f0e", edgecolor="k", lw=0.4, label="top-1")
    axes[2].set_xticks(x); axes[2].set_xticklabels(["MLP", "bilinear", "MPS+const"], fontsize=8)
    axes[2].set_ylabel("NMSE", color="#1f77b4"); axes[2].set_ylim(0.85, 0.92)
    ax2b.set_ylabel("top-1", color="#ff7f0e"); ax2b.set_ylim(0.08, 0.11)
    axes[2].set_title("GPT-2 medium L12 completion:\nMPS ties baselines (as in small)")
    save_fig(fig, FIGDIR / "fig1_scale.png")
    print(f"wrote {FIGDIR}")


if __name__ == "__main__":
    main()

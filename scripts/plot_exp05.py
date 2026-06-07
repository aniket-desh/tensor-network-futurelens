#!/usr/bin/env python
r"""Plot Exp 05: connected-only MPS-vs-MLP gap, and learned vs empirical correlation length."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tn_futurelens.utils.plotting import save_fig, set_style

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs" / "gpt2_exp05"
CORR = ROOT / "results" / "runs" / "gpt2_correlations" / "results.json"
FIGDIR = ROOT / "docs" / "05_connected_transfer" / "figures"


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    layers = [6, 12]
    conn = {L: json.load(open(RUNS / f"connected_layer_{L}.json")) for L in layers}
    trans = {L: json.load(open(RUNS / f"transfer_layer_{L}.json")) for L in layers}
    corr = {r["layer"]: r for r in json.load(open(CORR))["rows"]}

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # (a) connected-only: MPS-minus-MLP gap, full vs connected
    ax = axes[0]
    width = 0.35
    x = np.arange(len(layers))
    full = [conn[L]["conditions"]["full"]["mps_minus_mlp"] for L in layers]
    co = [conn[L]["conditions"]["connected_only"]["mps_minus_mlp"] for L in layers]
    ax.bar(x - width / 2, full, width, color="#90caf9", edgecolor="k", lw=0.4, label="full task")
    ax.bar(x + width / 2, co, width, color="#08306b", edgecolor="k", lw=0.4, label="connected-only")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"layer {L}" for L in layers])
    ax.set_ylabel("NMSE(MPS) − NMSE(MLP)   (>0 = MPS worse)")
    ax.set_title("Removing the persistent subspace does NOT\nhelp the MPS (gap stays >0, widens)")
    ax.legend(fontsize=9)

    # (b) learned transfer xi vs empirical bulk xi
    ax = axes[1]
    emp = [corr[L]["bulk_xi"] for L in layers]
    learned = [float(np.mean(trans[L]["learned_xi_top"][:4])) for L in layers]
    ax.bar(x - width / 2, emp, width, color="#d62728", edgecolor="k", lw=0.4,
           label="empirical bulk ξ (Exp 01)")
    ax.bar(x + width / 2, learned, width, color="#08306b", edgecolor="k", lw=0.4,
           label="learned MPS transfer ξ")
    ax.set_xticks(x); ax.set_xticklabels([f"layer {L}" for L in layers])
    ax.set_ylabel("correlation length ξ (tokens)")
    ax.set_title("Trained MPS transfer ξ ≈ 2 regardless of layer;\ndoes not track empirical ξ")
    ax.legend(fontsize=9)
    save_fig(fig, FIGDIR / "fig1_connected_and_transfer.png")
    print(f"wrote {FIGDIR}")


if __name__ == "__main__":
    main()

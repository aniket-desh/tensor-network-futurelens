#!/usr/bin/env python
r"""Plot Exp 09: correlation mode count in PCA vs learned-phi space."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tn_futurelens.utils.plotting import save_fig, set_style
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs" / "gpt2_exp09"
FIGDIR = ROOT / "docs" / "09_learned_phi_bridge" / "figures"


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    layers = []
    for p in sorted(RUNS.glob("layer_*.json")):
        layers.append(json.load(open(p)))
    set_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(layers)); w = 0.35
    pca = [d["pca_space"]["eff_modes_sv"] for d in layers]
    phi = [d["learned_phi_space"]["eff_modes_sv"] for d in layers]
    ax.bar(x - w / 2, pca, w, color="#1f77b4", edgecolor="k", lw=0.4, label="PCA space")
    ax.bar(x + w / 2, phi, w, color="#08306b", edgecolor="k", lw=0.4, label="learned-φ space")
    ax.set_xticks(x); ax.set_xticklabels([f"layer {d['layer']}" for d in layers])
    ax.set_ylabel("effective correlation modes (Hankel SV > 5%)")
    ax.set_title("Learned φ does NOT simplify the correlation structure\n"
                 "(more modes, not fewer) — no MPS-friendly low-mode space")
    ax.legend(fontsize=9)
    save_fig(fig, FIGDIR / "fig1_pca_vs_learned_phi.png")
    print(f"wrote {FIGDIR}")


if __name__ == "__main__":
    main()

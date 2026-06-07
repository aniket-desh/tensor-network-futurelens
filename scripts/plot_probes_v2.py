#!/usr/bin/env python
r"""Plot Exp 03 (constant channel + learned phi) results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tn_futurelens.utils.plotting import save_fig, set_style

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs" / "gpt2_probes_v2"
FIGDIR = ROOT / "docs" / "03_const_learned_phi" / "figures"
TABLE = ROOT / "results" / "tables" / "probe_results_v2.csv"

# the "ladder" we want to show, best D picked per MPS variant
LADDER = ["linear", "linear_learned", "mlp", "mlp_learned",
          "mps_pca", "mps_pca_const", "mps_learned_const"]
PRETTY = {"linear": "linear (PCA)", "linear_learned": "linear (learned φ)",
          "mlp": "MLP (PCA)", "mlp_learned": "MLP (learned φ)",
          "mps_pca": "MPS", "mps_pca_const": "MPS +const", "mps_learned_const": "MPS +const +learned φ"}
COLORS = {"linear": "#9e9e9e", "linear_learned": "#616161", "mlp": "#ffb74d",
          "mlp_learned": "#f57c00", "mps_pca": "#90caf9", "mps_pca_const": "#1f77b4",
          "mps_learned_const": "#08306b"}


def load():
    out = {}
    for p in sorted(RUNS.glob("layer_*.json")):
        d = json.load(open(p)); out[d["layer"]] = d
    return out


def best_of(results, kind, metric="nmse_mean"):
    """Best (min) value for a condition kind across its D values."""
    cand = [r for r in results if r.get("kind", r["name"].rsplit("_D", 1)[0]) == kind
            or r["name"] == kind or r["name"].startswith(kind + "_D")]
    cand = [r for r in results if r["name"] == kind or r["name"].startswith(kind + "_D")]
    return min(cand, key=lambda r: r[metric]) if cand else None


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    layers = load()

    with open(TABLE, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "name", "n_params", "nmse_mean", "kl_mean", "top1_mean"])
        for L, d in layers.items():
            for r in d["results"]:
                w.writerow([L, r["name"], r["n_params"], r["nmse_mean"], r["kl_mean"], r["top1_mean"]])

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    ylims = {"nmse_mean": (0.78, 0.98), "kl_mean": (3.3, 4.05)}
    for ax, metric, ylabel in [(axes[0], "nmse_mean", "validation NMSE"),
                               (axes[1], "kl_mean", "teacher KL")]:
        ax.set_ylim(*ylims[metric])
        width = 0.38
        for li, (L, d) in enumerate(layers.items()):
            vals, labels, cols = [], [], []
            for kind in LADDER:
                r = best_of(d["results"], kind, metric)
                if r is None:
                    continue
                vals.append(r[metric]); labels.append(PRETTY[kind]); cols.append(COLORS[kind])
            x = np.arange(len(vals)) + (li - 0.5) * width
            bars = ax.bar(x, vals, width=width, color=cols, alpha=0.65 if li else 1.0,
                          edgecolor="k", linewidth=0.4, label=f"layer {L}")
            # mark MLP (learned) reference line per layer
            mlpL = best_of(d["results"], "mlp_learned", metric)
            if mlpL:
                ax.axhline(mlpL[metric], color="#f57c00", ls=":" if li else "--", alpha=0.6, lw=1)
        ax.set_xticks(np.arange(len(LADDER)))
        ax.set_xticklabels([PRETTY[k] for k in LADDER], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by condition (best $D$)\nsolid=layer 6, faded=layer 12; "
                     "orange line = MLP+learned-φ")
    axes[0].legend(fontsize=9)
    fig.suptitle("Constant channel lifts the MPS to the linear baseline; learned-φ helps "
                 "every model; MPS+const+learned-φ ties the best baseline (note zoomed y-axis)",
                 y=1.02, fontsize=11)
    save_fig(fig, FIGDIR / "fig1_ladder.png")

    # D-saturation for mps_learned_const
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for L, d in layers.items():
        pts = sorted((r["D"] if "D" in r and r["D"] else int(r["name"].split("D")[-1]), r["nmse_mean"])
                     for r in d["results"] if r["name"].startswith("mps_learned_const_D"))
        if pts:
            ax.plot([x for x, _ in pts], [y for _, y in pts], "o-", label=f"layer {L}")
        ml = best_of(d["results"], "mlp_learned", "nmse_mean")
        if ml:
            ax.axhline(ml["nmse_mean"], ls="--", alpha=0.5)
    ax.set_xlabel("bond dimension $D$"); ax.set_ylabel("NMSE")
    ax.set_title("MPS (+const +learned φ) vs $D$\n(dashed = MLP+learned-φ)")
    ax.legend(fontsize=9)
    save_fig(fig, FIGDIR / "fig2_mps_vs_D.png")
    print(f"wrote {FIGDIR} and {TABLE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
r"""Plot/merge Exp 12 (GPT-J causal intervention) results across the two GPU runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tn_futurelens.utils.plotting import save_fig, set_style
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs" / "gptj_causal"
FIGDIR = ROOT / "docs" / "12_gptj_causal_intervention" / "figures"


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    merged = {}
    meta = {}
    for p in sorted(RUNS.glob("results_*.json")):
        d = json.load(open(p))
        meta = {k: d[k] for k in ("layer", "m", "n", "prompt_len", "n_windows")}
        merged.update(d["results"])
    json.dump({**meta, "results": merged}, open(RUNS / "results_merged.json", "w"), indent=2)

    n = meta.get("n", 3)
    order = [k for k in ["unigram", "readout_mps", "interv_single", "interv_mlp", "interv_mps"]
             if k in merged]
    pretty = {"unigram": "unigram floor", "readout_mps": "readout (no interv)",
              "interv_single": "interv: single-state", "interv_mlp": "interv: MLP",
              "interv_mps": "interv: MPS"}
    colors = {"unigram": "#bbbbbb", "readout_mps": "#2ca02c", "interv_single": "#ff7f0e",
              "interv_mlp": "#1f77b4", "interv_mps": "#08306b"}

    set_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(1, n + 1)
    for k in order:
        top1 = merged[k]["top1"][:n]
        ax.plot(x, top1, "o-", color=colors.get(k, "k"), label=pretty.get(k, k))
    ax.set_xlabel("future horizon (tokens ahead of observed window)")
    ax.set_ylabel("top-1 agreement with GPT-J's own future token")
    ax.set_title(f"GPT-J causal-intervention FutureLens (layer {meta.get('layer')})\n"
                 "does causal intervention beat the readout; does the TN map beat single/MLP?")
    ax.set_xticks(x); ax.legend(fontsize=9)
    save_fig(fig, FIGDIR / "fig1_causal_topk.png")
    print("merged:", {k: merged[k]["top1"] for k in order})


if __name__ == "__main__":
    main()

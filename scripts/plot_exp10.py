#!/usr/bin/env python
import json
from pathlib import Path
import numpy as np
from tn_futurelens.utils.plotting import save_fig, set_style
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs" / "gpt2_born"
FIGDIR = ROOT / "docs" / "10_born_machine" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
layers = [json.load(open(p)) for p in sorted(RUNS.glob("layer_*.json"))]
set_style()
fig, ax = plt.subplots(figsize=(7.5, 4.6))
order = ["unigram", "bigram", "mlp", "born"]
colors = {"unigram": "#bbbbbb", "bigram": "#2ca02c", "mlp": "#ff7f0e", "born": "#08306b"}
x = np.arange(len(layers)); w = 0.2
for k, name in enumerate(order):
    vals = [d["next_symbol_acc"][name] for d in layers]
    ax.bar(x + (k - 1.5) * w, vals, w, color=colors[name], edgecolor="k", lw=0.4, label=name)
ax.set_xticks(x); ax.set_xticklabels([f"layer {d['layer']}" for d in layers])
ax.set_ylabel("next-symbol accuracy (quantized, K=256)")
ax.set_title("Born-machine MPS conditional completion vs baselines\n"
             "generative TN does NOT beat a bigram")
ax.legend(fontsize=9)
save_fig(fig, FIGDIR / "fig1_born_vs_baselines.png")
print("wrote", FIGDIR)

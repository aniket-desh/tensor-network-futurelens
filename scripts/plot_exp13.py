#!/usr/bin/env python
import json
from pathlib import Path
import numpy as np
from tn_futurelens.utils.plotting import save_fig, set_style
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs" / "gpt2_longhorizon"
FIGDIR = ROOT / "docs" / "13_long_horizon_kl" / "figures"; FIGDIR.mkdir(parents=True, exist_ok=True)
by_n = {}
for p in sorted(RUNS.glob("results_*.json")):
    by_n.update(json.load(open(p))["by_n"])
ns = sorted(int(k) for k in by_n)
set_style()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
models = ["mlp", "conv1d", "bilinear", "mps_D16"]
colors = {"mlp": "#f57c00", "conv1d": "#9467bd", "bilinear": "#2ca02c", "mps_D16": "#08306b"}
for mname in models:
    axes[0].plot(ns, [by_n[str(n)][mname]["top1_mean"] for n in ns], "o-",
                 color=colors[mname], label=mname)
axes[0].set_xlabel("horizon n (tokens completed)"); axes[0].set_ylabel("top-1 agreement (mean over horizons)")
axes[0].set_title("Completion top-1 vs horizon (KL objective)"); axes[0].set_xscale("log", base=2)
axes[0].set_xticks(ns); axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter()); axes[0].legend(fontsize=9)
gap = [by_n[str(n)]["mps_minus_best_baseline"] for n in ns]
axes[1].plot(ns, gap, "ks-")
axes[1].axhline(0, color="r", ls=":", lw=1)
axes[1].set_xlabel("horizon n"); axes[1].set_ylabel("MPS − best baseline (top-1)")
axes[1].set_title("Does the MPS gap turn positive at long horizon?")
axes[1].set_xscale("log", base=2); axes[1].set_xticks(ns)
axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
save_fig(fig, FIGDIR / "fig1_long_horizon.png"); print("wrote", FIGDIR)

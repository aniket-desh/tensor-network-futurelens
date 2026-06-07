#!/usr/bin/env python
r"""Plot Exp 08: horizon sweep (MPS - best baseline vs n) and bond sweep (NMSE vs D)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tn_futurelens.utils.plotting import save_fig, set_style
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs" / "gpt2_exp08"
EXP06 = ROOT / "results" / "runs" / "gpt2_exp06" / "gpt2.json"
FIGDIR = ROOT / "docs" / "08_sweep" / "figures"


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    h = json.load(open(RUNS / "horizon.json"))
    ns = sorted(int(k) for k in h["by_n"])
    gap = [h["by_n"][str(n)]["mps_minus_best_baseline"] for n in ns]
    mps = [h["by_n"][str(n)]["mps"] for n in ns]
    mlp = [h["by_n"][str(n)]["mlp"] for n in ns]
    bil = [h["by_n"][str(n)]["bilinear"] for n in ns]
    axes[0].plot(ns, mlp, "s--", color="#f57c00", label="MLP")
    axes[0].plot(ns, bil, "^--", color="#2ca02c", label="bilinear")
    axes[0].plot(ns, mps, "o-", color="#08306b", label="MPS+const")
    axes[0].set_xlabel("horizon $n$ (future sites)"); axes[0].set_ylabel("validation NMSE")
    axes[0].set_title("Completion NMSE vs horizon (layer 6)")
    axes[0].set_xticks(ns); axes[0].legend(fontsize=9)
    ax0b = axes[0].twinx()
    ax0b.plot(ns, gap, "d:", color="#d62728", alpha=0.7)
    ax0b.set_ylabel("MPS − best baseline", color="#d62728")
    ax0b.axhline(0, color="#d62728", lw=0.8, ls=":")
    ax0b.tick_params(axis="y", labelcolor="#d62728")

    b = json.load(open(RUNS / "bond.json"))
    Ds = sorted(int(k) for k in b["by_D"])
    nmse = [b["by_D"][str(D)] for D in Ds]
    axes[1].plot(Ds, nmse, "o-", color="#08306b", label="MPS+const")
    axes[1].axhline(b["mlp"], ls="--", color="#f57c00", label="MLP")
    axes[1].axhline(b["bilinear"], ls="--", color="#2ca02c", label="bilinear")
    try:
        impliedD = {r["layer"]: r["implied_D"] for r in json.load(open(EXP06))["rows"]}[b["layer"]]
        axes[1].axvline(impliedD, ls=":", color="grey", label=f"Exp06 implied $D$={impliedD}")
    except Exception:
        pass
    axes[1].set_xlabel("bond dimension $D$"); axes[1].set_ylabel("validation NMSE")
    axes[1].set_title("MPS NMSE vs bond dimension (layer 6, $n=4$)")
    axes[1].set_xscale("log", base=2); axes[1].set_xticks(Ds)
    axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].legend(fontsize=9)
    save_fig(fig, FIGDIR / "fig1_sweeps.png")
    print(f"wrote {FIGDIR}")


if __name__ == "__main__":
    main()

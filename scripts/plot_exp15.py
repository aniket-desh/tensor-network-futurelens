#!/usr/bin/env python
r"""Exp 15 figure: effective mode count and block-ξ vs coarse-graining block size."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "runs" / "gpt2_exp15_block"
FIGDIR = ROOT / "docs" / "12_mythos_sprint" / "figures"

rows = json.load(open(OUTDIR / "modes_vs_block.json"))["rows"]
layers = sorted({r["layer"] for r in rows})
FIGDIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colors = {layers[0]: "tab:blue", layers[-1]: "tab:red"}
for L in layers:
    rs = sorted([r for r in rows if r["layer"] == L], key=lambda r: r["b"])
    bs = [r["b"] for r in rs]
    axes[0].plot(bs, [r["eff_modes_sv"] for r in rs], "-o", color=colors[L], label=f"layer {L}")
    axes[1].plot(bs, [r["bulk_xi_block"][0] for r in rs], "-o", color=colors[L],
                 label=f"layer {L} (block units)")
    axes[1].plot(bs, [r["bulk_xi_tokens"][0] for r in rs], "--s", color=colors[L],
                 alpha=0.5, label=f"layer {L} (token units)")
axes[0].set_xscale("log", base=2); axes[1].set_xscale("log", base=2)
for ax in axes:
    ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel("block size b (tokens per coarse-grained site)")
axes[0].set_ylabel("effective modes (Hankel SV > 5% max)")
axes[0].set_title("Coarse-graining INCREASES the mode count")
axes[0].legend()
axes[1].set_ylabel("leading bulk correlation length ξ")
axes[1].set_title("Block-ξ is scale-invariant (~8 blocks at every b)")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "fig3_block_coarse_graining.png", dpi=150)
print(f"wrote {FIGDIR/'fig3_block_coarse_graining.png'}")

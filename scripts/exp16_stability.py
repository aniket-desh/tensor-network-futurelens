#!/usr/bin/env python
r"""Exp 16B analysis — stability of probe families across seeds × learning rates.

Merges n=8 runs from sprint-1 and sprint-2 outdirs (identical prep/split/test set):
  gpt2_exp14_seeds   (5 models, lr 1.5e-3, seeds 0-3)
  gpt2_exp14_mech_a  (mps_D8/D32..., lr 1.5e-3, seeds 0-3)
  gpt2_exp14_lr      (mlp/bilinear/mps_D16 at 5e-4 & 3e-3; tags encode lr)
  gpt2_exp16_lrgrid  (missing lr cells + seeds 4-7 at 1.5e-3)

Per model: mean, seed sd (at shared lr, all seeds), lr sd (of per-lr means),
worst/best cell, regret of the shared lr vs the model's best lr, and the full
(lr, seed) -> top1 table. Writes stability_table.json + prints a summary.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRS = ["gpt2_exp14_seeds", "gpt2_exp14_mech_a", "gpt2_exp14_lr", "gpt2_exp16_lrgrid"]
MODELS = ["mlp", "conv1d", "bilinear", "mps_D8", "mps_D16"]
SHARED_LR = 1.5e-3


def lr_of(path, data):
    if "lr" in data:
        return float(data["lr"])
    t = path.stem
    if "5e4" in t:
        return 5e-4
    if "3e3" in t:
        return 3e-3
    return SHARED_LR


def main():
    cells = defaultdict(dict)        # model -> (lr, seed) -> top1
    for d in DIRS:
        for f in sorted((ROOT / "results" / "runs" / d).glob("results_*.json")):
            data = json.load(open(f))
            lr = lr_of(f, data)
            if data.get("n_train_used") not in (None, 40000):
                continue                                  # data-size runs excluded here
            if float(data.get("tail_alpha", 0)) != 0:
                continue
            for r in data["runs"]:
                if r["n"] != 8 or r["model"] not in MODELS:
                    continue
                cells[r["model"]][(lr, r["seed"])] = r["test_top1_mean"]

    table = {}
    print(f"{'model':9s} {'mean@1.5e-3':>12s} {'sd_seed(8)':>11s} {'sd_lr':>7s} "
          f"{'worst':>7s} {'best':>7s} {'best_lr':>8s} {'regret':>7s}")
    for m in MODELS:
        c = cells[m]
        lrs = sorted({k[0] for k in c})
        shared = [v for (lr, s), v in c.items() if lr == SHARED_LR]
        per_lr_mean = {lr: np.mean([v for (l2, s), v in c.items() if l2 == lr]) for lr in lrs}
        best_lr = max(per_lr_mean, key=per_lr_mean.get)
        rec = {
            "n_cells": len(c), "lrs": lrs,
            "mean_shared": round(float(np.mean(shared)), 5),
            "sd_seed_shared": round(float(np.std(shared, ddof=1)), 5),
            "per_lr_mean": {f"{lr:g}": round(float(v), 5) for lr, v in per_lr_mean.items()},
            "sd_lr": round(float(np.std(list(per_lr_mean.values()), ddof=1)), 5),
            "worst": round(float(min(c.values())), 5),
            "best": round(float(max(c.values())), 5),
            "best_lr": f"{best_lr:g}",
            "regret_shared": round(float(per_lr_mean[best_lr] - per_lr_mean[SHARED_LR]), 5),
            "cells": {f"{lr:g}|s{s}": round(v, 5) for (lr, s), v in sorted(c.items())},
        }
        table[m] = rec
        print(f"{m:9s} {rec['mean_shared']:>12.4f} {rec['sd_seed_shared']:>11.4f} "
              f"{rec['sd_lr']:>7.4f} {rec['worst']:>7.4f} {rec['best']:>7.4f} "
              f"{rec['best_lr']:>8s} {rec['regret_shared']:>7.4f}")
    out = ROOT / "results" / "runs" / "gpt2_exp16_lrgrid" / "stability_table.json"
    json.dump(table, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

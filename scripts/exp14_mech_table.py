#!/usr/bin/env python
r"""Merge main grid + mechanism runs at n=8 into one paired table.

All runs share the same prep split, so per-window correctness files are directly
pairable across outdirs. Reports seed-mean test top-1, seed sd, and the paired
cluster-bootstrap 95% CI of (variant − mps_D16) for the ablations and
(mps_D16 − variant) only as raw gaps for baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DIRS = ["gpt2_exp14_seeds", "gpt2_exp14_mech_a", "gpt2_exp14_mech_b", "gpt2_exp14_multpool",
        "gpt2_exp16_minclass"]
N = 8
TEST_START = 50000          # 40k train + 10k select
SEQ = 216


def cluster_boot(d, clusters, n_boot, rng):
    uniq = np.unique(clusters)
    sums = np.array([d[clusters == c].sum() for c in uniq])
    sizes = np.array([(clusters == c).sum() for c in uniq])
    idx = rng.integers(0, uniq.size, (n_boot, uniq.size))
    return sums[idx].sum(1) / sizes[idx].sum(1)


def main():
    rng = np.random.default_rng(0)
    per = {}                       # model -> [S, N_test] seed-stacked per-window means
    for dname in DIRS:
        outdir = ROOT / "results" / "runs" / dname
        files = sorted(outdir.glob(f"correct_n{N}_seed*.pt"))
        by_seed = {}
        for f in files:
            s = int(f.stem.split("seed")[1])
            by_seed[s] = torch.load(f)
        if not by_seed:
            continue
        seeds = sorted(by_seed)
        for mdl in by_seed[seeds[0]]:
            per[mdl] = np.stack([by_seed[s][mdl].float().mean(1).numpy() for s in seeds])
    ref = per["mps_D16"]
    clusters = (np.arange(ref.shape[1]) + TEST_START) // SEQ
    print(f"n={N}, {ref.shape[0]} seeds, {ref.shape[1]} test windows, "
          f"{len(np.unique(clusters))} sequences\n")
    print(f"{'model':18s} {'mean':>7s} {'seed sd':>8s} {'Δ vs mps_D16':>13s} {'boot95(Δ)':>20s}")
    rows = {}
    for mdl in sorted(per, key=lambda m: -per[m].mean()):
        x = per[mdl]
        d = (x - ref[:x.shape[0]]).mean(0)
        bs = cluster_boot(d, clusters, 10000, rng)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows[mdl] = {"mean": round(float(x.mean()), 5),
                     "seed_sd": round(float(x.mean(1).std(ddof=1)), 5),
                     "delta_vs_mps": round(float(d.mean()), 5),
                     "boot95": [round(float(lo), 5), round(float(hi), 5)]}
        print(f"{mdl:18s} {x.mean():7.4f} {x.mean(1).std(ddof=1):8.4f} "
              f"{d.mean():+13.4f}  [{lo:+.4f},{hi:+.4f}]")
    out = ROOT / "results" / "runs" / "gpt2_exp14_seeds" / "mech_table_n8.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

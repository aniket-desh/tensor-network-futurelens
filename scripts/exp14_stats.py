#!/usr/bin/env python
r"""Exp 14 statistics — paired, multi-seed analysis of the MPS-vs-baseline gap.

Inputs: results/runs/gpt2_exp14_seeds/correct_n{n}_seed{s}.pt ({model: bool [N_test, n]}).

Views reported per horizon n:
  1. per-seed test-mean gap MPS − each baseline, and across-seed mean ± sd + t-stat;
  2. gap vs *best* baseline, best chosen two ways: on test means (Exp 13 convention,
     biased against MPS) and on the select set (clean);
  3. paired CLUSTER bootstrap: stride-1 windows overlap heavily (216 windows per
     256-token sequence), so windows are resampled by sequence (cluster id =
     global_index // 216), averaging correctness over seeds per window first;
  4. per-horizon-position breakdown for the n where the gap is largest.

Usage: python scripts/exp14_stats.py [--mps mps_D16]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "runs" / "gpt2_exp14_seeds"
BASELINES = ["mlp", "conv1d", "bilinear", "attention"]


def load_runs():
    runs, split = [], None
    for f in sorted(OUTDIR.glob("results_*.json")):
        d = json.load(open(f))
        runs += d["runs"]
        split = d["split"]
    return runs, split


def cluster_boot(d, clusters, n_boot, rng):
    """Paired cluster bootstrap of mean(d): resample sequences with replacement."""
    uniq = np.unique(clusters)
    by_c = [d[clusters == c] for c in uniq]
    sums = np.array([x.sum() for x in by_c])
    sizes = np.array([x.size for x in by_c])
    idx = rng.integers(0, uniq.size, (n_boot, uniq.size))
    return sums[idx].sum(1) / sizes[idx].sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mps", default="mps_D16")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    runs, split = load_runs()
    test_start = split[0] + split[1]            # global index of first test window
    sel = {(r["n"], r["seed"], r["model"]): r["sel_top1_best"] for r in runs}

    files = sorted(OUTDIR.glob("correct_n*_seed*.pt"))
    by_n: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    for f in files:
        n, s = (int(x) for x in f.stem.replace("correct_n", "").split("_seed"))
        by_n.setdefault(n, {})[s] = torch.load(f)

    summary = {}
    for n in sorted(by_n):
        seeds = sorted(by_n[n])
        models = list(by_n[n][seeds[0]].keys())
        baselines = [b for b in BASELINES if b in models]
        # per-window correctness averaged over horizons: [S, N] per model
        per = {mdl: np.stack([by_n[n][s][mdl].float().mean(1).numpy() for s in seeds])
               for mdl in models}
        mps = per[args.mps]
        print(f"\n=== n={n} (seeds {seeds}, test windows {mps.shape[1]}) ===")
        rows = {}
        for b in baselines:
            gaps = mps.mean(1) - per[b].mean(1)               # per-seed gap
            mu, sd = gaps.mean(), gaps.std(ddof=1) if len(gaps) > 1 else 0.0
            t = mu / (sd / np.sqrt(len(gaps))) if sd > 0 else np.inf
            # paired cluster bootstrap (by sequence) on seed-averaged correctness
            d = (mps - per[b]).mean(0)                         # [N] paired diffs
            clusters = (np.arange(d.size) + test_start) // 216
            bs = cluster_boot(d, clusters, args.boot, rng)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows[b] = {"gap_mean": round(float(mu), 5), "gap_sd": round(float(sd), 5),
                       "t": round(float(t), 2), "boot95": [round(float(lo), 5), round(float(hi), 5)],
                       "per_seed": [round(float(g), 5) for g in gaps]}
            print(f"  {args.mps} - {b:9s}: {mu:+.4f} ± {sd:.4f} (seeds), t={t:+.2f}, "
                  f"boot95=[{lo:+.4f},{hi:+.4f}] per-seed={[f'{g:+.4f}' for g in gaps]}")
        # best baseline two ways
        test_means = {b: per[b].mean() for b in baselines}
        best_test = max(test_means, key=test_means.get)
        sel_means = {b: np.mean([sel.get((n, s, b), np.nan) for s in seeds]) for b in baselines}
        best_sel = max(sel_means, key=sel_means.get)
        gap_vs_best_test = float(mps.mean() - test_means[best_test])
        gap_vs_best_sel = float(mps.mean() - per[best_sel].mean())
        print(f"  best baseline on test: {best_test} (gap {gap_vs_best_test:+.4f}); "
              f"on select: {best_sel} (gap {gap_vs_best_sel:+.4f})")
        summary[n] = {"rows": rows, "best_test": best_test,
                      "gap_vs_best_test": round(gap_vs_best_test, 5),
                      "best_sel": best_sel, "gap_vs_best_sel": round(gap_vs_best_sel, 5),
                      "means": {m: round(float(per[m].mean()), 5) for m in models}}
    json.dump(summary, open(OUTDIR / "stats_summary.json", "w"), indent=2)
    print(f"\nwrote {OUTDIR/'stats_summary.json'}")


if __name__ == "__main__":
    main()

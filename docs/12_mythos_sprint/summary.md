# Mythos sprint — Stress-testing the missing MPS advantage · Summary

> **STATUS: DRAFT SKELETON — numbers pending Exp 14 completion.**

## Executive summary (≤600 words)

[PLACEHOLDER — written last]

## Finding 1 — The Exp 13 "intermediate-horizon MPS edge" [VERDICT PENDING]

Exp 13 reported the project's only positive: under the KL objective, MPS-D16 beat the
best of three baselines at n=8 (+0.7%) and n=16 (+0.5%), single seed. We identified
five methodological loopholes (single seed; no attention baseline; epoch selection on
the reported set; unpaired statistics; ~21 effective independent eval texts due to
stride-1 window overlap) and re-ran the comparison with all five fixed (Exp 14:
4 seeds, 5 models, 80/10/10 train/select/test with 50k-window ≈ 231-sequence test set,
paired cluster bootstrap).

![fig1](figures/fig1_gap_vs_horizon.png)

[TABLE + verdict]

## Finding 2 — RG-style coarse-graining makes the chain LESS MPS-friendly, not more

The last untested structural hope (TASK Experiment D) was that block variables
v̄_I = mean(v_{Ib..Ib+b-1}) might have a lower effective mode count, creating the
few-mode regime an MPS is uniquely efficient at. The opposite holds (Exp 15):

![fig3](figures/fig3_block_coarse_graining.png)

- Effective mode count (block-Hankel SVs > 5% of max, p=64 PCA basis) **rises
  monotonically** with block size: layer 6: 27→33→43→45; layer 8: 34→41→48→49
  (b = 1→2→4→8).
- The leading bulk correlation length is **scale-invariant in block units** (ξ ≈ 8–9
  blocks at every b; linear growth in token units). The residual stream is
  approximately self-similar — closer to a critical/long-range many-mode system than
  to the gapped, few-mode chain the original MPS argument assumed.
- b=1 reproduces Exp 06's mode count exactly (27 at layer 6) — pipeline sanity.

Together with Exp 09 (learned φ increases modes 27→58), every representation change
tried moves the structure *away* from MPS-friendliness: the high-rank property is
robust to basis (φ) and to scale (blocking) — it is not an artifact of the token-level
description.

## Finding 3 — [Phase 3 result placeholder]

## What would have changed our mind

[PLACEHOLDER]

## Research map

- `scripts/exp14_seeds.py` / `exp14_stats.py` / `plot_exp14.py` — multi-seed stress
  test of the Exp 13 edge; results in `results/runs/gpt2_exp14_seeds/`.
- `scripts/exp15_block.py` / `plot_exp15.py` — block coarse-graining mode counts;
  results in `results/runs/gpt2_exp15_block/modes_vs_block.json`.
- `docs/12_mythos_sprint/{plan.md,research_log.md,figures/}`.

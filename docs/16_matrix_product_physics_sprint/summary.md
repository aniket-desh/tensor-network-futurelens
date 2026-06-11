# Matrix-product physics sprint — Summary

> **STATUS: DRAFT — 16B/16E/16C numbers pending.**

**Sprint:** 2026-06-11 19:30 UTC → 2026-06-12 (≈12 h) · 2× A40 48GB.
**Question (TASK_TWO.md):** *Is the MPS useful because it is a tensor network, or
because it is a stable non-commuting multiplicative feature map?*

## Executive summary

[FINAL — written last]

## Finding 1 — The MPS is a frozen random feature map: training the cores adds nothing

![fig_minclass](figures/fig_minclass.png)

[Settled: frozen random near-identity cores +0.0002 vs trained; rank-2 ≡; shuffle ≡;
commuting fails; fixed-φ fails (−0.46%); symmetrized fails (−0.48%); multiscale
baselines fail.]

## Finding 2 — Fair lr tuning kills the mean edge; robustness is what survives

[PENDING 16B completion — MLP@3e-4 mean .1002 > MPS .0991 (4 seeds); 1e-4 cells and
8-seed variance pending]

## Finding 3 — The de-persisted bulk is power-law, not exponential

![fig_powerlaw](figures/fig_powerlaw.png)

[Settled: AIC winner pow 8/8 across layers/blocks; α≈0.4–0.75; robust to persistent
removal rank 4/8/16. Reinterprets Claim A. Multiscale architectures still fail to
exploit it (dilatedconv/treepool at conv1d level).]

## Finding 4 — Scale and tail [PENDING 16E medium layers, 16C tail losses]

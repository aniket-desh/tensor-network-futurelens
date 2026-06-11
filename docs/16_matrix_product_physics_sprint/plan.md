# Sprint 2 plan — matrix-product mechanism, robustness, and new physics

Written 2026-06-11 19:40 UTC (T+0:10). Hardware: 2× A40 48GB (TASK_TWO says H100s;
actual pods are the same A40s as sprint 1). 93 GiB cgroup memory cap — prep-once
discipline throughout.

## What the repo has shown (post-Mythos)

- Claim A: finite-ξ bulk + persistent subspace; bulk is many-mode and *self-similar*
  under blocking (modes rise 27→45, block-ξ ≈ 8 at every scale).
- Claim B: MPS-D16 beats mlp/conv1d/bilinear/attention at every n∈{4..32} under a
  shared recipe (4 seeds, held-out test, cluster CIs > 0); per-model lr tuning
  shrinks the n=8 edge from +0.38% to +0.07–0.28%; ties bilinear at GPT-2 medium L12.
  MPS seed/lr variance is 3–6× smaller than the MLP's.
- Claim C: causally falsified (site shuffle costs zero) — but the bond-free diagonal
  product probe falls to MLP level, so the non-commuting D×D algebra is load-bearing.

## Remaining loopholes / open axes

1. **Is training the cores necessary?** (random-feature hypothesis untested)
2. Non-commutativity tested only via one diagonal probe (256×D=1); a same-D diagonal
   control and low-rank-core controls are missing.
3. Robustness claim rests on 4 seeds × 3 lrs — too thin to headline.
4. The per-position tail advantage was never targeted by a loss, only observed.
5. Medium scale tested at one layer (12) — "unlucky layer" unresolved.
6. Self-similarity is descriptive so far: no power-law-vs-exponential fit, no
   multiscale predictive baseline.

## Sprint design (deep on 16A + 16B; 16C folded into eval; 16E/16D as fillers)

### 16A — minimal matrix-product class (GPU0, ~2.5h)
Variants at L6, n=8, 4 seeds, existing prep (directly comparable to sprint-1 grid):
`mps_D16_frozen` (random near-identity cores frozen; train φ+head),
`mps_D16_frozenorth` (frozen random orthogonal cores),
`mps_D16_fixedphi` (frozen PCA φ; isolates the φ contribution),
`mps_diag_D16` (diagonal trainable cores at same D),
`mps_D16_rank2/4` (cores constrained to rank r),
`mps_D16_sym4` (averaged over 4 fixed site permutations),
plus 16D-flavored predictive baselines: `dilatedconv`, `treepool`.
References: sprint-1 mps_D16/D8/multpool/mlp/bilinear at identical protocol.
Decision table per TASK_TWO §4A interpretation key.

### 16B — stability grid (GPU1, ~6-7h)
- lr×seed grid: {mlp, bilinear, conv1d, mps_D8, mps_D16} × lr {3e-4,5e-4,1e-3,
  1.5e-3,3e-3} × seeds {0,1,2,3}, n=8, existing prep (40k train).
- data-size axis: new 80k-train prep (exp16 prep: 80k/10k/50k); sizes {10k,20k,40k,
  80k} × {mlp, bilinear, mps_D16} × 2 lrs {5e-4, 1.5e-3} × seeds {0,1}.
- Outputs: μ, σ_seed, σ_lr, worst/best case, regret vs per-model best lr.

### 16C — folded in + one targeted run
Per-position top-1(s) curves come free from saved correctness tensors. One
tail-weighted KL run (w_s ∝ s^α, α∈{1,2}) at n=32 for {mps_D16, bilinear, mlp},
2 seeds — does tail-weighting flip the ranking or amplify the MPS tail edge?

### 16E — medium layer sweep (GPU0 after 16A, ~3h incl. caching)
Cache gpt2-medium layers {8,16,20,24}; preps; grid {mlp, conv1d, bilinear,
attention, mps_D16} × 4 seeds × layers {8,16,20}, n=8. Key graph: gap vs layer
(with sprint-1 L12 point).

### 16D-lite — physics diagnostics (CPU, anytime)
Power-law vs exponential fits of whitened connected correlations across block scales
b∈{1,2,4,8}, after projecting out the persistent subspace (|λ|>0.9 modes);
report AIC/fit quality per scale. Predictive side covered by dilatedconv/treepool in
16A.

## Timeline
- T+0:00–0:45 setup, runner extensions, smoke.
- T+0:45–3:30 16A on GPU0; 16B lr-grid starts GPU1.
- T+3:30–6:30 16E on GPU0 (cache → prep → grid); 16B continues; 16D on CPU.
- T+6:30–8:30 16B data-size axis; 16C tail runs; analysis + figures.
- T+8:30–10:00 buffer / follow-ups from results.
- T+10:00–12:00 writing + red-team (≥90 min reserved).

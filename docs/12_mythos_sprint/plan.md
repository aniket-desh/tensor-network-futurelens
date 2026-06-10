# Mythos sprint — plan (written at T+0:45, 2026-06-10 21:45 UTC)

## Where the repo stands (from FINDINGS.md + Exp 12/13)

- Claim A (finite-ξ structure): supported, scale-robust (GPT-2 small/medium, GPT-J).
- Claim B (predictive advantage): weak/tie everywhere **except** Exp 13, which found a
  single-seed +0.7% (n=8) / +0.5% (n=16) top-1 edge for MPS-D16 over the best of three
  baselines under the KL objective at intermediate horizons — exactly the regime the
  TASK brief flags as the most plausible remaining opening (Experiment A).
- Claim C (transfer-matrix mechanism): not supported across readout / masked /
  generative / causal-intervention probes (Exp 12 closed the FutureLens-aligned lever:
  MPS is the *worst* donor map at GPT-J scale).

TASK.md's two priority experiments (A: long-horizon KL, B: causal intervention) have
both already been run (Exp 13, Exp 12). The single highest-information thing this
sprint can do is **decide whether the Exp 13 edge is real** — it is the only surviving
positive for the tensor network in the whole project.

## Identified loopholes in Exp 13 (what makes the edge suspect)

1. **Single seed** — model init and batch order both fixed at seed 0.
2. **No attention baseline** — TASK.md §3 requires attention/state-space among strong
   baselines; Exp 13 compared only MLP / conv1d / bilinear. `AttentionPool` (B3)
   exists in the repo but was not used.
3. **Selection on the reported metric** — early stopping picks the best epoch by val
   top-1 *on the same set that is reported*. With ~4.5k val windows the per-horizon
   binomial SE is ≈0.5%, the same size as the claimed edge. Max-over-epochs bias can
   differ between architectures (different epoch-to-epoch variance).
4. **No paired statistics** — gap reported as difference of means, not a paired
   per-window test, which throws away most of the statistical power.

## Sprint plan

### Phase 1 (hours 0–1.5): setup + reproduce
- uv sync, unit tests (36 passed), rebuild GPT-2 residual cache (6000 seqs, same
  config as prior experiments), smoke-run one exp13 config to reproduce the n=8
  numbers approximately and measure runtime.

### Phase 2 (hours 1.5–7): Exp 14 — the verdict experiment on the Exp 13 edge
`scripts/exp14_seeds.py`, both GPUs, seeds in parallel:
- **Design fixes:** 3-way split (80/10/10 train/select/test; epoch chosen on select,
  reported on held-out test), 50k windows (vs 30k), per-window per-horizon correctness
  saved for paired bootstrap/McNemar stats.
- **Models:** MLP, conv1d, bilinear, **attention (B3)**, MPS-D16(+const, learned φ) —
  identical learned-φ treatment for all, parameter counts recorded.
- **Grid:** n ∈ {4, 8, 16, 32}, m=8, layer 6, p=64, KL objective — exp13's exact
  regime. Seeds {0,1,2,3} (5th if time).
- **Decision rule:** the edge is "real" if the paired, multi-seed MPS − best-baseline
  gap at n=8/16 is positive with CI excluding 0 against *all* baselines including
  attention, on held-out test. Otherwise the verdict is "selection/seed noise" and the
  no-go picture is complete.

### Phase 3 (hours 7–10, conditional)
- **If edge survives:** mechanism probe — bond ablation D ∈ {1, 2, 4, 8, 16} at n=8
  (does the edge *need* the chain?), layer sweep (3/6/9), m=16. Explain why the regime
  is MPS-friendly.
- **If edge dies:** complete the no-go cleanly; spend remaining GPU time on TASK
  Experiment D (block coarse-graining: does RG-style blocking lower mode count?) as
  the one untested new direction.

### Phase 4 (final ~1.5–2 hours): writing + red-team
`docs/12_mythos_sprint/summary.md` per TASK.md §7–8.

## Compute budget
2× A40 48GB. Cache build ~15 min. Exp 14: measure per-config cost in smoke test;
seeds parallelized across GPUs. Reserve ≥1.5 h wall-clock for writing.

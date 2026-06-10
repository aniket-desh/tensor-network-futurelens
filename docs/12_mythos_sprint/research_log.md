# Mythos sprint — research log

Sprint start: **2026-06-10 21:01 UTC**. Hardware: 2× NVIDIA A40 48GB, 96 CPU, 503 GB RAM.

## T+0:00 – T+1:00 — orientation, environment, plan

- Read TASK.md, README, FINDINGS.md, Exp 12/13 summaries, and the core library code
  (`mps.py`, `baselines.py`, `probes.py`, `phi.py`, `eval.py`, `exp13_long_horizon.py`).
- **Key situational fact:** TASK.md's two priority experiments are already done in this
  repo — Exp 12 = Experiment B (causal intervention, negative: MPS is the worst donor
  map at GPT-J), Exp 13 = Experiment A (long-horizon KL, *small single-seed positive*:
  MPS−best-baseline = +0.7% top-1 at n=8, +0.5% at n=16, tie at n=32).
- Therefore the sprint's center of gravity: **stress-test the Exp 13 edge** — the only
  surviving positive for the TN hypothesis. Loopholes found by code reading:
  1. single seed (init + batch order),
  2. no attention baseline (TASK.md §3 requires one; `AttentionPool` exists unused),
  3. early-stop epoch selected on the *reported* val set (selection bias ~ same order
     of magnitude as the claimed edge: binomial SE ≈0.5%/horizon on 4.5k windows),
  4. unpaired statistics.
- Environment: fresh clone; `uv sync --extra interp` done; **36/36 unit tests pass**;
  no residual cache on disk → rebuilding (6000 wikitext-103 seqs × 256 tokens, layers
  0–12 even, same config as prior experiments).
- Wrote `plan.md`. Decision: Exp 14 = multi-seed, attention-baseline, 3-way-split
  replication with paired per-window stats; conditional Phase 3 (mechanism ablation if
  edge survives, block coarse-graining if it dies).

## T+1:00 – T+1:30 — Exp 14 designed, smoke-tested, launched

- Rebuilt cache: 6000 seqs × 256 tokens, layers {0,2,4,…,12}, logit-lens sanity exact
  (max|manual−model| = 0). ~16 GB.
- **A fifth loophole found while re-reading `windows.py`:** stride-1 windows overlap
  almost completely — each 256-token sequence yields 216 windows, so Exp 13's 4.5k
  val windows were only **~21 independent texts**. Any window-level error bar wildly
  overstates precision. Exp 14 therefore (a) uses a 50k-window test set (~231
  sequences) and (b) the stats script does a **paired cluster bootstrap by sequence**.
- `scripts/exp14_seeds.py`: 40k train / 10k select / 50k test (split fixed across
  seeds); models {mlp, conv1d, bilinear, attention(d_model=256), mps_D16(+const,
  learned φ)}; KL objective, lr 1.5e-3, bs 160, ≤15 epochs, patience 3 on select;
  per-window correctness saved for paired stats; teacher tokens fp32; training decode
  in bf16 autocast (same for all models). Param counts at n=8: MPS 1.76M ≤ MLP 1.83M
  ≤ bilinear 3.66M; conv 0.92M, attention 0.53M (smaller baselines are acceptable —
  they only strengthen a negative, and the MPS stays under the MLP ceiling).
- Smoke tests pass (all 5 models). Full grid launched 21:25 UTC, 4 procs / 2 GPUs:
  n∈{4,8,16}×seeds{0,1,2,3}, n=32×seeds{0,1} (n=32 costs 4× n=8; the decision regime
  is n=8/16). ETA ~01:00 UTC.

## T+0:30 – T+1:00 — ops + first numbers

- **Ops lesson:** container has a 93 GiB cgroup memory cap (host shows 503 GB). Both
  n=32 jobs were OOM-killed twice during the simultaneous dataset-build peak (~30 GB
  transient per proc); relaunched staggered (second job waits for the first to pass
  its build phase). The n∈{4,8,16} jobs were unaffected.
- Launched Exp 15 part 1 (block coarse-graining mode count, CPU, free) early since it
  is branch-independent. **First result:** layer 6, b=1 → 27 effective modes
  (replicates Exp 06's ≈27 exactly — good pipeline sanity); b=2 → **33 modes**, i.e.
  block-averaging *increases* the mode count so far, like learned-φ did in Exp 09.
- Exp 14 n=4 results trickling in (seed 0/2): mlp 0.110, conv 0.106, bilinear 0.111,
  attention 0.106 — bilinear leads at short horizon as in Exp 13.

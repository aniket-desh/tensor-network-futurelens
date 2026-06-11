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

## T+1:00 – T+1:45 — OOM fight → prep-once refactor; Exp 15 complete

- The 93 GiB cgroup cap kept biting: the relaunched n=32 build peaks OOM-killed the
  two *main* grid jobs (largest RSS at that moment). Root cause: every process
  re-built the 100k-window dataset (~28 GB transient).
- **Fix:** `scripts/exp14_prep.py` builds the dataset once and saves fp16 tensors +
  teacher tokens + train-split mean/std + train-fit PCA (6.2 GB, lossless — the cache
  shards are already fp16). Runner now loads prep (~30 s startup, ~8 GB steady) and
  converts fp16→fp32 per batch. Grid relaunched 22:28 UTC; per-model time dropped
  ~10× without contention (conv1d epoch: 17 s).
- Salvaged from the killed runs: complete n=32 seed-0 quintet — **MPS 0.0881 vs best
  baseline conv1d 0.0849 (+0.32%)**, bilinear collapses (0.0809) as in Exp 13, even
  with 14.3M params (2.2× MPS).
## T+1:45 – T+2:45 — Exp 14 verdict forming: the edge is real and broader than Exp 13

- Relaunched grid reproduces pre-OOM numbers exactly (seed-deterministic even with
  bf16 decode) — good.
- **n=8 (the Exp 13 headline regime), 4/4 seeds positive on held-out test with the
  attention baseline included:** MPS 0.0999/0.0998/0.0978/0.0990 (mean 0.0991, sd
  0.001) vs best-baseline-per-seed mean 0.0953 → **gap +0.38%** (Exp 13 claimed +0.7%
  on its leaky protocol; the clean number is about half, but solidly positive).
- **n=4:** MPS mean gap +0.18% (3/4 seeds positive) — Exp 13 had MPS *behind* at n=4;
  with 40k train windows (vs 25.5k) the MPS edges ahead even at short horizon.
- **n=32:** MPS mean gap vs best ≈ +0.18% (3/4 seeds positive; attention wins seed 1).
- **Per-horizon profile (n=32):** bilinear dominates h1 (0.148) then collapses by h8
  (0.072); the MPS is never best at h1 but has the FLATTEST decay — best at every
  position h4–h24. The aggregate edge is long-range robustness, not short-range fit.
- Phase 3 mechanism probes armed to fire when the main grid exits: bond ablation
  (D=2/4/8/32) + **site-shuffle control** (fixed permutation before the MPS — if the
  edge survives order destruction the chain mechanism is falsified) + no-const
  ablation + mlp_shuf harness check, all at n=8 × 4 seeds.

## T+2:45 – T+4:00 — main grid verdict + mechanism probes complete

- **Main grid (00:00 UTC): the edge is real.** Paired cluster-bootstrap stats: MPS-D16
  beats each of the four baselines individually at every n∈{4,8,16,32}; all 16 boot95
  CIs exclude zero; 61/64 per-seed comparisons positive. Gap vs best baseline per
  seed: +0.18%/+0.38%/+0.21%/+0.18%. Figures 1–2 generated and committed.
- **Mechanism probes (00:47 UTC), n=8, 4 seeds, paired:**
  - `mps_D16_shuf` (fixed site permutation): **Δ = −0.0000, CI [−0.0005,+0.0004]** —
    destroying the 1D chain order costs *nothing*. The transfer-matrix/chain mechanism
    (Claim C) is positively falsified even though the MPS wins; `mlp_shuf` ≈ `mlp`
    validates the harness.
  - Bond curve: D2 .0928 (82k) < D4 .0944 (162k) < **D8 .0989 (482k) ≈ D16 .0991
    (1.76M)** > D32 .0981 (6.88M, slight overfit). D8 beats every baseline at ~26% of
    the MLP's parameters.
  - `mps_D16_noconst`: −0.12% (CI excludes 0) — const channel helps slightly; not the
    driver.
- **Reframe:** what wins is the *order-insensitive multilinear product structure with
  a moderate (D≈8–16) bottleneck*, not tensor-network chain geometry. Claim B turns
  positive; Claim C stays dead, now with direct causal evidence (shuffle) rather than
  only correlational evidence (mode counts).
- Launched layer-8 replication (n=8, 5 models, 4 seeds) to rule out a layer-6 quirk.

## T+4:00 – T+4:45 — layer-8 replication + medium pipeline + full draft

- **Layer 8 (n=8, 4 seeds): edge replicates** — MPS .0989 vs best-per-seed .0976
  (+0.13%, 4/4 seeds, all per-baseline CIs > 0). Smaller than layer 6 (+0.38%) but
  unambiguous. Not a layer-6 quirk.
- Launched GPT-2 medium pipeline (cache layers 12/24 → prep → n=8 grid, 4 seeds).
  Ops notes: two cache jobs raced on building tokens.pt (one EOFError, relaunched);
  background `( until ...; cmd ) &` watcher subshells do NOT survive tool-session
  cleanup — only direct long-running processes do; switched to a single tracked
  background task for the whole chain.
- Full summary.md drafted (exec summary ≈560 words, findings 1–4, figures 1–4);
  medium numbers pending.

## T+4:45 – T+5:45 — GPT-2 medium replication + red-team runs

- **GPT-2 medium (345M, layer 12, n=8, 4 seeds): the edge attenuates.** MPS .0934 vs
  bilinear .0925 → +0.09%, CI [−0.01, +0.20] — a tie with the strongest baseline
  (2/4 seeds positive), while still beating MLP/conv1d/attention with CIs > 0.
  Reported plainly: advantage solid at 124M, not established at 345M.
- Red-team self-objections converted into runs: (a) `attention_big` (d_model=512,
  ~1.5M params — parameter-matched attention) at n=8 × 4 seeds; (b) learning-rate
  robustness {5e-4, 3e-3} × {mlp, mps_D16} × 2 seeds (shared lr=1.5e-3 could favor
  either family).
- Ops: medium token-build race (two cache jobs both built tokens.pt; one EOFError'd
  loading a half-written file) — relaunched; future scripts should build tokens
  explicitly first.

- **Exp 15 (block coarse-graining) complete — clean negative for Experiment D:**
  effective modes RISE with block size (L6 27→45, L8 34→49 at b=1→8) while block-ξ is
  scale-invariant (≈8 blocks at every b). The chain is self-similar and many-mode at
  every scale; no RG-revealed MPS-friendly regime. Figure committed
  (`figures/fig3_block_coarse_graining.png`).

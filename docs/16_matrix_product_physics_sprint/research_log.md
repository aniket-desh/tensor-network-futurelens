# Sprint 2 research log — matrix-product mechanism & robustness

Sprint start: **2026-06-11 19:30 UTC** (12 h budget → ~07:30 UTC). 2× A40 48GB
(TASK_TWO assumes H100s; pods are the sprint-1 A40s). 93 GiB cgroup cap — prep-once
discipline from sprint 1 retained.

## T+0:00 – T+0:45 — orientation, plan, runner extensions, launch

1. **What the repo has shown:** Claim A solid (+ self-similar under blocking);
   Claim B small horizon-broad edge under shared recipe, +0.07–0.28% under per-model
   lr tuning, 3–6× lower seed/lr variance, tie vs bilinear at medium L12; Claim C
   causally falsified (shuffle) but bond matrices load-bearing (bond-free product
   probe = MLP level).
2. **Remaining loopholes:** training-the-cores never ablated (random-feature
   hypothesis); non-commutativity tested only via one diagonal probe; robustness
   claim rests on 4 seeds × 3 lrs; tail advantage observed but never targeted;
   medium tested at one layer; self-similarity descriptive only.
3. **Priorities:** deep on 16A (minimal class) + 16B (stability); 16C folded into
   per-position eval (+ one tail-α run); 16E medium layers 8/16/20 as GPU filler;
   16D-lite power-law fits on CPU and dilatedconv/treepool as predictive baselines
   inside 16A.

- Tests 36/36 pass; preps from sprint 1 intact (L6/L8 small, L12 medium).
- Extended `exp14_seeds.py` (same runner for provenance): `mps_D16_frozen` (random
  near-identity cores frozen), `mps_D16_frozenorth` (frozen random orthogonal
  slices), `mps_D16_fixedphi` (frozen PCA φ), `mps_diag_D16` (diagonal/commuting at
  matched D — hidden width D vs D² confound noted; sprint-1 `multpool` is the
  matched-params diagonal point), `mps_D16_rank{2,4}` (low-rank core slices; no
  near-identity init exists at rank<D — intrinsic to the constraint),
  `mps_D16_sym4` (env averaged over 4 fixed site permutations), `dilatedconv`,
  `treepool` (16D predictive baselines); `--max-train` (data-size axis),
  `--tail-alpha` (w_s ∝ s^α KL weights); optimizer now skips frozen params.
- Smoke: all variants run; at 1 ep × 4k windows everything sits at the
  predict-the-mean floor (0.0929) — expected; differentiates by 4 ep × 20k
  (frozen 0.0955 vs diag 0.0929).
- **Launched 19:45 UTC:** GPU0 = 16A grid (9 variants × 4 seeds, n=8, sprint-1 prep
  → directly comparable to sprint-1 references). GPU1 = 16B lr-grid chain (only the
  56 cells missing after reusing sprint-1 runs: all 5 models at lr {3e-4, 1e-3};
  conv1d/mps_D8 at {5e-4, 3e-3}; then seeds 4–7 at 1.5e-3 for an 8-seed variance
  estimate).

## T+0:45 – T+1:15 — first wave of results (seed 0–1, partial)

- **16A (mechanism), seed 0:** frozen random near-identity cores **0.0995** ≈ trained
  D16 (0.0999, sprint 1); seed 1 frozen = **0.1002**. Rank-2 cores 0.0992, rank-4
  0.0990 — full edge retained. Frozen *orthogonal* cores 0.0967 (worse but >MLP).
  Frozen-PCA-φ 0.0928 (φ is essential). Diagonal-at-D16 0.0929 = predict-the-mean
  floor. **Symmetrizing over 4 permutations HURTS (0.0952)** — any one fixed order
  works; mixing orders blurs the features. dilatedconv 0.0926 / treepool 0.0926 —
  multiscale predictive baselines fail.
  → Emerging mechanism: the MPS acts as a **random non-commuting multiplicative
  feature expansion** (cores need not be trained; effective rank ≤2), with learned φ
  as the adapter.
- **16B early:** MLP keeps improving at lower lr — 3e-4 gives 0.1002/0.0995 (s0/s1),
  *above* the MPS reference. Extended the grid to lr=1e-4 (chained). The tuned-mean
  edge may die entirely; robustness would then be the whole Claim-B story.
- **16D (power-law vs exponential): decisive.** After persistent-subspace removal,
  the bulk correlation decay is **power-law at every layer and block scale** (AIC
  winner 8/8; pow R² 0.92–0.98 vs exp 0.77–0.87; α≈0.4–0.75 falling with b; robust
  to n_persist ∈ {4,8,16}). Sprint-1's "scale-invariant block-ξ" is explained:
  fitting exponentials to a power law gives ξ ∝ window. **Claim A's "finite
  correlation length" was partly a fit artifact — the de-persisted bulk is
  scale-free.**
- Armed chains: GPU0 → medium layers {8,16,20} after 16A; GPU1 → lr 1e-4 cells +
  data-size axis (80k prep built).

## T+1:15 – T+2:15 — 16A COMPLETE: the MPS is a frozen random feature map

Final paired table (4 seeds, 50k test windows, cluster-boot CIs vs trained MPS-D16
.0991; full table in `tables/mech_table_n8_full.json`, figure `fig_minclass.png`):

- **Indistinguishable from trained:** frozen random near-identity cores **.0994**
  (+0.0002 [−0.0004,+0.0009] — training the cores adds NOTHING), rank-2 cores .0993,
  shuffled .0991, D8 .0989.
- Slightly below: rank-4 .0986, D32 .0981, no-const .0979, frozen orthogonal .0976
  (the near-identity component matters mildly; pure random rotations lose 0.15%).
- **Fail tier (≈MLP .0955 or below):** commuting variants (multpool .0956, diagonal
  D16 .0937), frozen-PCA-φ .0945 (−0.46% — φ is the essential learned component),
  symmetrized-over-4-orders .0944 (−0.48% — each single order works, MIXING orders
  hurts), dilatedconv .0923 / treepool .0921 (multiscale baselines fail → the
  power-law structure is descriptive, not exploitable by these architectures),
  conv1d .0922, attention .0914.

**Mechanism settled (Success mode 1):** the minimal sufficient structure is a
*frozen random near-identity non-commuting matrix product at D≈8–16 + learned linear
φ + linear head*. The MPS functions as a matrix-product **random kernel/feature
expansion**; the only learning that matters below the head is φ. Non-commutativity
necessary (commuting fails), training unnecessary, order arbitrary-but-fixed.

## T+3:30 – T+4:15 — 16B COMPLETE: under fair tuning the MPS LOSES; 16E medium curve

- **Full lr grid (6 lrs × 4–8 seeds), per-lr means:** MLP rises monotonically toward
  low lr (3e-3 .0953 → 1e-4 **.1009**); bilinear same shape (→ **.1001**); mps_D16 is
  an inverted-U (peak **.0993** at 5e-4, plateau .0984–.0993 over [3e-4,1.5e-3],
  collapse .0931 at 1e-4); mps_D8 peaks .0981. → **Tuned ranking: MLP > bilinear >
  MPS-D16 > MPS-D8, with MLP +0.16% over the MPS's own best; 4/4 seeds.**
  Claim B is not merely shrunk — it is INVERTED under fair tuning. The sprint-1 edge
  was real only under the shared lr 1.5e-3 (MPS sweet spot, baselines' bad spot).
- What survives: the lr-robustness contrast is real (MPS spread 0.0032 over the
  10× range [3e-4,3e-3] vs MLP 0.0049 and a much deeper fall at 1e-4 for MPS), and
  the shared-lr regret is 0.46% for MLP vs 0.09% for MPS. But "flat response with a
  lower ceiling" is weak as a practical contribution — anyone with a small tuning
  budget should use the MLP. Honest verdict: Success mode 5 (clean negative for the
  predictive claim) + mode 1 (mechanism) + the physics finding.
- lr 3e-5 bracket run queued (does the MLP keep rising?).
- **16E medium (shared recipe):** L8 gap vs best baseline +0.21% (3/4 seeds, 1 tie);
  L16 ≈ tie (+0.02%); with sprint-1 L12 +0.09% → the medium edge decays with depth;
  L20 running. All shared-recipe numbers — given the tuning result, read these as
  recipe-conditional, not architecture-superiority claims.

## T+4:15 – T+5:00 — medium curve done; lr bracket; figures

- **16E complete (medium, shared recipe, 4 seeds):** gap vs best baseline
  L8 +0.21%±0.18, L12 +0.09%±0.30, L16 +0.02%±0.17, L20 +0.06%±0.11 — largest in the
  early-middle, fading to tie with depth; and all recipe-conditional given 16B.
- **lr 3e-5 bracket (seed 0):** MLP 0.1015 (hits the 15-epoch cap), bilinear 0.0993,
  MPS 0.0931 → MLP's protocol-constrained optimum is a broad plateau ≈[3e-5, 3e-4]
  at ~0.100–0.102; bracketed.
- Figures rendered: fig_minclass, fig_powerlaw (log-log straight vs log-linear
  curved), fig_medium_layers. Datasize 10k/20k done; 40k/80k + tail-α queued.

## T+2:15 — 16B interim: tuned-mean edge INVERTS (superseded by T+4:15 above)

- MLP at lr 3e-4 (4 seeds): .1002/.0995/.0992/.1020 → **mean .1002 > MPS .0991**.
  MLP@1e-3 .0974 — its optimum is ≤3e-4 (1e-4 cells queued). With a finer lr grid
  than sprint 1, the *tuned-mean* MPS edge disappears and likely flips negative;
  the robustness/no-tuning property becomes the entire Claim-B content.

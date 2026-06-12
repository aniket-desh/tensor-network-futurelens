# Tensor Network FutureLens — Consolidated Findings (Exp 00–11)

A single-page synthesis for handoff. Per-experiment detail in `docs/<NN>/summary.md`.

> **2026-06-11 update — Mythos sprint (Exp 14–15, [full report](12_mythos_sprint/summary.md)):**
> the Exp 13 edge **replicated** under a strict protocol (4 seeds, attention baseline,
> held-out test, cluster bootstraps): MPS > every baseline at every n ∈ {4..32} under a
> shared recipe; per-model lr tuning shrinks it to +0.07–0.28% with 3–6× lower
> seed/lr variance; ties bilinear at GPT-2 medium. **Mechanism: site-shuffling the MPS
> costs nothing → Claim C causally falsified** — the win is order-insensitive
> multilinearity with a D≈8 bottleneck, not chain structure. Block coarse-graining
> *raises* the effective mode count (27→45) with scale-invariant block-ξ ≈ 8.
> A bond-free product probe (256 × D=1 channels, matched params) falls to MLP level —
> the non-commuting bond matrices are load-bearing; the token-chain order is not.
> Verdict table below reflects Exp 00–13; see the sprint report for the update.

> **2026-06-12 update — Matrix-product physics sprint (Exp 16,
> [full report](16_matrix_product_physics_sprint/summary.md)): the resolution.**
> (i) Frozen random cores ≡ trained cores: the MPS is a **random multiplicative
> feature map**; only φ and the head learn. (ii) Under per-model lr tuning the
> ranking inverts — MLP .1013 > bilinear .1003 > MPS .0993 — and the tail advantage
> flips too; **Claim B is closed as a clean negative** (the MPS wins only below
> ~20k training windows, the random-features trade-off). (iii) **Claim A revised:**
> after persistent-subspace removal the bulk decays as a power law at every block
> scale (AIC 8/8) — the residual stream is scale-free, not finite-ξ.
> Positive-outlook threads that emerged from these negatives are logged in
> [OPEN_DIRECTIONS.md](OPEN_DIRECTIONS.md).

## The question, split into three claims

- **Claim A — Structure:** transformer residual streams, read along token position at a
  fixed layer, have **finite correlation length** (a 1D-many-body-like chain).
- **Claim B — Predictive advantage:** an MPS probe exploits that structure to predict
  future residuals/tokens **better than parameter-matched baselines**.
- **Claim C — Mechanism:** the advantage is specifically due to the **MPS transfer-matrix
  connected modes** capturing the residual correlations.

## Verdict

| Claim | Status | Key evidence |
|---|---|---|
| **A — finite-ξ structure** | ✅ **Supported, scale-robust** | Exp 01 (finite-ξ bulk, ξ≈3–8, + persistent subspace growing with depth); Exp 06 (decaying-bulk ξ≈5–9 in the clean fixed basis); Exp 11 (replicates at GPT-2 medium) |
| **B — predictive advantage** | ⚠️ **Weak / tie, with one small regime-specific positive** | Exp 02 (readout loses); Exp 03 (const+learned φ → ties best baseline); Exp 07 (no edge under MSE/KL at n=4); Exp 12 (causal: MPS donor *worst*). **But Exp 13:** under the KL objective at *intermediate horizons* the MPS edges the best baseline (n=8 +0.7%, n=16 +0.5%, single-seed) and is the most horizon-robust probe — small but consistent |
| **C — transfer-matrix mechanism** | ❎ **Not supported (full method space)** | Exp 05 (removing persistent subspace doesn't help; learned ξ≠empirical); Exp 06 (correlations are high-rank/many-mode, not few-mode); Exp 09 (learned φ gives *more* modes); Exp 10 (generative Born MPS beaten by a bigram); Exp 12 (causal intervention: TN is the worst donor map) |

Probe families tested (all negative for the TN *mechanism*): readout (B4, Exp 02–03),
masked completion (B5, Exp 04), generative Born machine (B6, Exp 10), and causal
intervention (FutureLens's strong method, Exp 12). Scales: GPT-2 124M / 345M / GPT-J 6B.

## The refined physics picture

$$V_i \;\approx\; \underbrace{G_i}_{\text{low-rank persistent / global}} \;+\; \underbrace{B_i}_{\text{finite-}\xi\text{ bulk, many modes}}$$

- The **bulk is finite-range** (ξ a few tokens, shortest mid-network) — Claim A holds.
- But it is **high-rank**: ≈17 modes at layer 0 → ≈48 at layer 12 (of 64), growing with
  depth (Exp 06). Implied bond $D\sim4$–7, not 2–3. The transfer-matrix *parsimony*
  argument (great for *few* modes) therefore doesn't bite.
- The **persistent subspace grows with depth** (fraction 0.11→0.49; a linear probe
  captures it trivially), so the predictive signal that exists isn't in the connected
  modes the MPS is uniquely good at.

## What actually moved the needle (for a predictor)

1. **Constant channel** (`ṽ=(1,v)`): lifts the MPS from below-linear to the linear
   baseline (it couldn't represent additive structure before) — Exp 03.
2. **Learned φ:** the single biggest lever — but it helps *every* model equally, and it
   does **not** create an MPS-friendly low-mode space (Exp 09).
3. **KL / logit training objective** (vs residual MSE): lifts top-1 from ~0.10 to ~0.13
   for *all* probes — the biggest single improvement found, and architecture-agnostic
   (Exp 07). **Use this going forward.**

None of these is MPS-specific.

## Methodological assets built

- Activation cache + correlation diagnostics (matrix two-point, whitening, exp/Prony/
  Hankel fits) — `src/tn_futurelens/{data,analysis}`.
- **Ho-Kalman/ERA realization** of a matrix correlation sequence (clean, fixed-basis
  mechanism probe) — `analysis/realization.py`.
- MPS family: readout (B4, ±const channel), masked completion (B5), Born machine (B6);
  feature maps; strong baselines (attention/conv/bilinear) — `src/tn_futurelens/models`.
- 36 unit tests; every experiment reproducible from `scripts/` + `docs/<NN>/plan.md`.

## Honest open levers (mostly closed)

1. **Causal-intervention probe class at GPT-J scale** — **TESTED (Exp 12).** Causal
   intervention beats the readout (replicates FutureLens), but the single-state donor is
   best and the MPS is the *worst* donor map — the TN doesn't help here either. The one
   remaining matched-to-FutureLens lever is now closed.
2. **Higher-order (non-Gaussian) structure**: Exp 06's realization is second-order; a
   nonlinear MPS could in principle exploit non-Gaussian structure the linear realization
   misses — but Exp 07/10/12 (nonlinear MPS, Born MPS, MPS donor) found no edge, so this
   is a long shot.
3. **Continuous/Gaussian-emission Born MPS** to avoid Exp 10's quantization loss (for
   Gaussian data this reduces to the Exp 06 realization).
4. **Maximal-fidelity FutureLens intervention** (raw-state transplant + KL distillation +
   long prompts) would raise *absolute* numbers but, per Exp 12's stronger-tuning re-run,
   not the single-vs-MPS ordering.

## One-sentence takeaway

> Transformer residual streams **do** have finite-correlation-length structure (and it
> replicates from GPT-2 to GPT-J), but it is **high-rank and not carried by the connected
> modes an MPS is uniquely efficient at — so a tensor-network probe, in readout,
> masked-completion, generative, or causal-intervention form, is at best competitive with,
> rarely better than, a learned feature map plus a generic predictor (and for causal
> interventions the single most-recent state beats the whole trajectory). The one place a
> small, consistent MPS edge does appear is completion under the KL objective at
> *intermediate* horizons (Exp 13: +0.5–0.7% top-1 at n=8–16, single-seed), exactly the
> narrow regime the physics analysis predicted. The analogy was real; the predicted
> *clean* computational advantage was not — only a marginal, regime-specific one.**

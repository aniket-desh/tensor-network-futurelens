# Tensor Network FutureLens — Consolidated Findings (Exp 00–11)

A single-page synthesis for handoff. Per-experiment detail in `docs/<NN>/summary.md`.

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
| **B — predictive advantage** | ⚠️ **Weak / tie** | Exp 02 (readout loses); Exp 03 (with const channel + learned φ, ties best baseline); Exp 07 (no edge vs attention/conv/bilinear under MSE *or* KL); Exp 08 (gap→0 with horizon, never positive) |
| **C — transfer-matrix mechanism** | ❎ **Not supported** | Exp 05 (removing persistent subspace doesn't help; learned ξ≠empirical); Exp 06 (correlations are high-rank/many-mode, not few-mode); Exp 09 (learned φ gives *more* modes); Exp 10 (generative Born MPS beaten by a bigram) |

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

## Honest open levers (not yet tested)

1. **Causal-intervention probe class** (FutureLens's stronger method) at **GPT-J scale** —
   a *different family* from the readouts/completions here; the one place a TN map could
   still matter (map observed trajectory → soft prompt / residual intervention).
2. **Higher-order (non-Gaussian) structure**: Exp 06's realization is second-order; a
   nonlinear MPS could in principle exploit non-Gaussian residual structure the linear
   realization misses — but Exp 07/10 (nonlinear MPS, Born MPS) found no edge, so this is
   a long shot.
3. **Continuous/Gaussian-emission Born MPS** to avoid Exp 10's quantization loss (for
   Gaussian data this reduces to the Exp 06 realization).

## One-sentence takeaway

> Transformer residual streams **do** have finite-correlation-length structure (and it
> replicates with scale), but it is **high-rank and not carried by the connected modes an
> MPS is uniquely efficient at — so a tensor-network probe, in readout, masked-completion,
> or generative form, is at best competitive with, never mechanistically better than,
> a learned feature map plus a generic predictor; the training objective matters more
> than the architecture.**

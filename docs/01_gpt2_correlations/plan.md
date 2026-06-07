# Experiment 01 — GPT-2 Residual Correlation Diagnostics · Plan

**This is the experiment that tests the project's core hypothesis** (briefing §5, §11):
do GPT-2 residual-stream features, read along token position at a fixed layer, have
finite correlation length? If yes (and few modes), an MPS is the right inductive
bias and we proceed to completion experiments; if power-law / no clean decay, an MPS
is a poor bias and that is itself informative.

## Question
For each source layer $\ell$, how does the residual two-point correlation
$\hat C^\ell(\Delta)$ decay with token lag $\Delta$? Is it exponential (finite
correlation length), and how many modes?

## Observable
Whitened matrix two-point function $\hat C^\ell(\Delta)=(C(0))^{-1/2}C(\Delta)(C(0))^{-1/2}$
in PCA-whitened feature space ($p=64$), summarised by operator norm, Frobenius norm,
and trace; single-exponential and floor+exponential fits; Hankel mode count.

## Data
GPT-2 small, WikiText-103, 8000 article-aware sequences × 256 tokens
($\approx 2\times10^6$ positions), layers $\{0,2,4,6,8,10,12\}$, cached fp16.
BOS (position 0) dropped; correlations computed within sequences only (clean doc
boundaries). Logit-lens sanity (folded LN) verified: $\max|\,$manual$-$model logits$|=0$.

## Method
1. Fit PCA-whitening $\phi$ ($768\to64$) per layer.
2. Stream all shards through a streaming correlation accumulator (single pass,
   stationary identity $C(\Delta)=\mathbb E[V_iV_{i+\Delta}^\top]-\mu\mu^\top$), on GPU.
3. Fit `floor + amp·exp(-Δ/ξ)` to the trace → bulk $\xi$ + long-range floor.
4. Count "persistent" directions: singular values of $\hat C(\Delta{=}32)$ above 0.5.

## Predictions / decision rule (briefing §5.4)
- Exponential + low-mode ⇒ MPS plausible ⇒ run completion (Exp 02) on those layers.
- Power-law / no decay ⇒ MPS poor bias ⇒ interpret negative result seriously.
- Exponential only in some layers ⇒ focus MPS there; use others as controls.

## Reproduce
```bash
# (after caching: scripts/cache_residuals.py)
.venv/bin/python scripts/compute_correlations.py
```
Outputs: `results/tables/correlation_fits_gpt2.csv`, `figures/`, `results/runs/gpt2_correlations/`.

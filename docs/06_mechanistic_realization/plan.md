# Experiment 06 — Mechanistic realization in the fixed PCA basis · Plan

**Goal (reviewer next-step #1, the clean mechanistic test).** Ask whether an MPS
transfer matrix *can represent* the measured residual correlations, with **no learned-φ
and no readout confound** — by working entirely in the fixed PCA-whitened basis where
the correlations were measured (Exp 01).

**Why not a trained MPS?** A trained autoregressive TI MPS with a fixed linear head
cannot extract the most-recent site from the cumulative product environment (the
coefficient of $v_i$ in $H_i=H_{i-1}M_i$ depends on the varying history). It fails to fit
even AR(1) (NMSE≈1.0). So a trained-MPS transfer spectrum is an unreliable probe.

**Instead — direct realization (Ho-Kalman / ERA).** A linear-Gaussian MPS that reproduces
$C(\Delta)$ is $C(\Delta)=H A^{\Delta-1}G$. We recover it directly from the empirical
matrix correlation sequence:
- block-Hankel SVD → singular spectrum; #significant SVs = number of correlation modes
  (state dim); an MPS needs bond $D$ with $D^2-1\ge$ that.
- eigenvalues of the realized $A$ = the decay modes $\lambda_\mu$; $\xi_\mu=-1/\ln|\lambda_\mu|$.

Both live in the same fixed PCA basis as the Exp 01 measurement — the clean comparison.

## Validation
Ho-Kalman on *measured* (noisy) synthetic correlations must recover planted ξ
(AR(1)→6.15; multi-mode→{2,8,30}).

## GPT-2
Layers {0,2,4,6,8,10,12}, PCA-whitened $p=64$, $\Delta\le 40$. Report effective mode
count, realized bulk ξ vs Exp 01, implied bond $D$.

## Reproduce
```bash
python scripts/exp06_mechanistic.py --mode both --device cuda:0
```

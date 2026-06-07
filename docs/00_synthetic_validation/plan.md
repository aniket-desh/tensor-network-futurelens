# Experiment 00 — Synthetic Validation · Plan

**Goal.** Before touching transformer activations, validate the parts of the
pipeline that the briefing's theory actually constrains, on processes with
*known* answers (briefing §13–14). This de-risks the correlation diagnostics and
the MPS machinery so that anything we later see on GPT-2 reflects the data, not a
code bug.

## Questions
1. Does the correlation-diagnostic code recover the true correlation length $\xi$
   and mode count $M$ of a known process? Does it flag a non-exponential (power-law)
   process as the negative control?
2. Does the transfer-matrix **counting law** hold — i.e. does representing a
   correlation function with $M$ exponential modes require bond dimension
   $D$ with $D^2-1\ge M$ (so $D\sim\sqrt{M}$)?
3. Is the trainable MPS layer correct, and does it capture multiplicative
   cross-site structure that a linear probe cannot?

## Method
- **Processes** (`tn_futurelens.data.synthetic`): AR(1) (single mode), sum-of-AR
  with well-separated $\xi$ (multi-mode), power-law $(1+\Delta)^{-\alpha}$ (control).
- **Diagnostics**: whitened matrix two-point function $\hat C(\Delta)$ → operator
  norm → single-exponential log-linear fit ($\xi$, $R^2$); Hankel-rank mode count.
- **Counting test**: measure the whitened trace correlation of $M$-mode processes
  ($M=1,2,4,7$); fit it with a $(D^2-1)$-mode model for $D=1..4$; locate the $D$ at
  which the reconstruction error hits the noise floor; compare to $\sqrt{M+1}$.
- **MPS correctness**: train linear / MLP / MPS probes on a synthetic target that is
  a *product* of per-site linear forms (degree-$m$ polynomial); compare NMSE.

## Success criteria
- AR(1): single-exp $R^2>0.99$, recovered $\xi$ within ~10% of $-1/\ln\rho$.
- Power-law: visibly worse single-exp fit; straight on log–log, curved on semilog.
- Counting: elbow $D$ matches $\lceil\sqrt{M+1}\rceil$.
- MPS beats linear (which should be at chance) on the multiplicative task.

## Reproduce
```bash
.venv/bin/python -m pytest -q                      # 29 unit tests (math vs known answers)
.venv/bin/python scripts/run_synthetic_validation.py
```
Outputs: `figures/`, `results/runs/synthetic_validation/results.json`.

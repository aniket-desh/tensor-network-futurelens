# Experiment 07 — Strong baselines + KL/logit objective · Plan

**Motivation (next-steps #3, #4).** Exp 02–05 compared the MPS only to linear/MLP and
trained on residual MSE. Two gaps: (a) the baselines weren't strong at *sequence*
structure; (b) MSE may be the wrong objective for future-token prediction.

## Questions
1. Does the MPS beat baselines that are good at local sequence structure — attention
   pooling, 1D conv, low-rank bilinear (a fixed second-order, MPS-flavoured baseline)?
2. Does training on the FutureLens objective — teacher KL through the frozen unembed —
   change the ranking vs residual MSE?

## Setup
Layer 6, $m=8$, $n=4$, $p=64$, 150k windows. All probes use a learned φ (init PCA) +
constant channel (for the MPS). Two objectives: residual MSE and teacher KL (decode
predicted & true $r^L$ through the folded-LN unembed). Metrics: NMSE, teacher KL, top-1.

## Reproduce
```bash
python scripts/exp07_strong_baselines.py --objective mse --layer 6 --device cuda:0
python scripts/exp07_strong_baselines.py --objective kl  --layer 6 --device cuda:0
python scripts/plot_exp07.py
```

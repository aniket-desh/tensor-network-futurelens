# Experiment 04 — B5 masked-MPS completion · Plan

**Motivation (next-steps #3).** Exp 02/03 used a B4 *readout* (contract observed sites,
then heads). The theory is a *completion*: one chain of $m+n$ sites, condition on the
first $m$, complete the next $n$. Does that closer-to-theory geometry help?

## Model (B5, `MaskedMPSCompletion`)
Chain of $m+n$ sites; observed sites get features, the $n$ future sites get **learned
mask vectors**; contract left-to-right; decode the cumulative environment at each
future site $\to \hat r^L_{t+s}$. Same external signature as the readout so it uses the
same φ / trainer / metrics.

## Question
Does B5 masked completion beat the B4 readout (and the MLP) on the same task?

## Conditions (per layer, learned φ + constant channel — the Exp 03 best config)
`mlp` (frozen-PCA reference) · `b4_readout` · `b5_masked`, MPS swept $D\in\{16,32\}$.
Layers 6 and 12. Metrics: NMSE, teacher-KL, top-1.

## Reproduce
```bash
python scripts/train_masked_mps.py --layer 6  --device cuda:0
CUDA_VISIBLE_DEVICES=1 python scripts/train_masked_mps.py --layer 12 --device cuda:0
python scripts/plot_masked.py
```

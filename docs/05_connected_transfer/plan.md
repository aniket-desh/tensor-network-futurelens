# Experiment 05 — Connected-only test + TI transfer spectrum · Plan

The deepest checks of *whether the MPS's specific mechanism* (connected finite-ξ
correlation modes via a transfer matrix) is what helps — not just a generic learned
feature map.

## (a) Connected-only (next-steps #4)
Exp 01: residual correlation = persistent (long-range) subspace + finite-ξ bulk. The
MPS advantage is supposed to live in the **connected** (finite-ξ) modes. If so, removing
the persistent subspace should make the MPS's edge over the MLP *grow*.
- Whiten a layer (PCA $p=64$); estimate the persistent projector $P$ from the top
  singular vectors of $\hat C(\Delta{=}32)$ (those with σ>0.5).
- Build the autoregressive completion task (predict whitened future from whitened
  observed) on the **full** features and on **connected-only** features ($v-Pv$).
- Train MLP and MPS(+const) on each; compare the gap NMSE(MPS)−NMSE(MLP).
- Layers 6 (small persistent subspace) and 12 (large).

## (b) TI transfer spectrum (next-steps #5, Phase 6)
Train a **translation-invariant** MPS (+const +learned φ); extract its transfer-matrix
correlation lengths $\xi_\mu=-1/\ln|\lambda_\mu/\lambda_1|$ and compare to the empirical
bulk ξ from Exp 01. If the MPS uses the transfer mechanism, learned ξ should track
empirical ξ and its layer dependence.

## Reproduce
```bash
python scripts/exp05_connected_transfer.py --mode connected --layer 12 --device cuda:0
python scripts/exp05_connected_transfer.py --mode connected --layer 6  --device cuda:0
python scripts/exp05_connected_transfer.py --mode transfer  --layer 6  --device cuda:0
python scripts/exp05_connected_transfer.py --mode transfer  --layer 12 --device cuda:0
python scripts/plot_exp05.py
```

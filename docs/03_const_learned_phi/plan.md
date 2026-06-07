# Experiment 03 — Constant channel + learned φ completion · Plan

**Motivation (next-steps #1, #2).** Exp 02 found an MPS *readout* loses to baselines.
Two likely reasons, from review feedback: (1) a pure-multiplicative MPS
$\psi=\sum W_{a_1\ldots a_m}v_{1,a_1}\cdots v_{m,a_m}$ **cannot represent the additive /
linear part** of the task; (2) the MPS only ever saw a *frozen* PCA feature map. Fix
both and re-test.

## Question
Does adding a **constant channel** ($\tilde v=(1,v)$, so
$A_j(\tilde v)=A_j^0+\sum_a v_{j,a}A_j^a$) and a **learned φ** (init from PCA, trained
jointly) make the MPS competitive with / beat parameter-matched baselines?

## Conditions (per layer; one shared raw-residual dataset)
- `linear`, `mlp` — frozen-PCA φ (Exp-02 references)
- `linear_learned`, `mlp_learned` — **learned φ** (fairness: same φ treatment as the MPS)
- `mps_pca` — frozen PCA, no const (reproduces Exp 02)
- `mps_pca_const` — frozen PCA + constant channel (isolates #1)
- `mps_learned_const` — learned φ + constant channel (#1 + #2)
- MPS swept over $D\in\{16,32\}$.

## Task / metrics
Same as Exp 02: observed window ($m=8$) of source-layer residuals → 4 future
final-layer residuals; NMSE, teacher-KL, top-1 (folded-LN unembed). Layers 6 (short
bulk ξ) and 12 (long-range control). 150k windows.

## Decision rule
- `mps_pca_const` ≈ linear ⇒ confirms the missing additive component was the issue.
- `mps_learned_const` beats `mlp_learned` ⇒ the MPS structure adds value beyond a
  learned feature map + generic nonlinearity (the result Exp 02 was missing).

## Reproduce
```bash
python scripts/train_probes_v2.py --layer 6  --device cuda:0
CUDA_VISIBLE_DEVICES=1 python scripts/train_probes_v2.py --layer 12 --device cuda:0
python scripts/plot_probes_v2.py
```

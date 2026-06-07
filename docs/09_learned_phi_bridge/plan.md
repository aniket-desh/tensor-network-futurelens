# Experiment 09 — Bridge: correlations in the learned-φ space · Plan

**Motivation (the "most interesting follow-up").** Exp 06 measured high-rank, many-mode
correlation structure in the *fixed PCA* basis. But the predictive wins used a *learned*
φ. Does the learned φ select a *simpler* (lower-rank / longer-ξ) correlation structure —
a space where the MPS transfer story would hold — or the same many-mode structure?

## Method
Learn a task-relevant φ via a learned-φ + LINEAR completion probe (φ = predictive linear
basis, no nonlinearity confound). Then realize (Ho-Kalman) the residual correlation
spectrum in the whitened learned-φ space and compare to the PCA-space result (Exp 06):
effective mode count, persistent-subspace size, bulk ξ.

## Reproduce
```bash
python scripts/exp09_bridge.py --layer 6 --device cuda:0
```

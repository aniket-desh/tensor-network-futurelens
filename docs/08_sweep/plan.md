# Experiment 08 — Systematic horizon (n) and bond (D) sweeps · Plan

**Motivation (next-step #5).** Test whether the MPS advantage appears in a particular
regime: longer future horizon (more sites for correlations to propagate) or larger bond.

## Plots
- (MPS − best baseline) NMSE vs horizon $n\in\{1,2,4,8\}$ (m=8).
- MPS NMSE vs bond $D\in\{2,4,8,16,32\}$ (m=8, n=4), with the Exp-06 implied $D$ marked.

All probes: learned φ + constant channel. Layer 6, 150k windows.

## Reproduce
```bash
python scripts/exp08_sweep.py --part horizon --device cuda:0
python scripts/exp08_sweep.py --part bond    --device cuda:0
python scripts/plot_exp08.py
```

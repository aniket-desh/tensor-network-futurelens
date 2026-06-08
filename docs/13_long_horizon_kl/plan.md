# Experiment 13 — Long-horizon completion under the KL objective · Plan

**Motivation.** Bond dimension is already swept (Exp 08: D=2→32, MPS saturates *at* the
baseline by D~16), so a bigger-D sweep is not the high-value move. Exp 08 also showed the
MPS−baseline gap *shrinks* with horizon (+0.009 @n=1 → ~0 @n=4,8) — so if any MPS regime
exists it is at **longer horizons**. And Exp 07 showed the **training objective matters
more than architecture**. This experiment combines both: long horizons + KL objective +
strong baselines, and asks whether the MPS gap finally turns **positive**.

## Setup
GPT-2 small, layer 6, m=8, p=64, learned φ + (const for MPS). Horizons n ∈ {4, 8, 16, 32}.
Train each probe with the **teacher-KL objective** (decode predicted & true future
residuals through the unembed; KL to the model's own future distribution — no transformer
forward needed for completion). Metric: top-1 agreement with the model's own future token,
per horizon. Strong baselines: MLP, conv1d, bilinear. Plus a D=64 sanity at n≤8.

## Question
Does `MPS − best baseline` (top-1, mean over horizons) become **positive** at n=16, 32,
or does it stay tied (as at n≤8)?

## Reproduce
```bash
python scripts/exp13_long_horizon.py --horizons 4 8 16 --device cuda:0 --tag a --d64
CUDA_VISIBLE_DEVICES=1 python scripts/exp13_long_horizon.py --horizons 32 --device cuda:0 --tag b
python scripts/plot_exp13.py
```

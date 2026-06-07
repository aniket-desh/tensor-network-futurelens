# Experiment 11 — Scale up to GPT-2 medium · Plan

**Motivation (next-step #6).** Does the GPT-2-small picture (finite-ξ bulk + growing
persistent subspace; MPS competitive but no edge) hold at larger scale?

## Setup
GPT-2 medium (24 blocks, $d_{\text{model}}=1024$), WikiText-103, 6000 sequences × 256
(~1.5M positions), layers {0,4,8,12,16,20,24}, fp16, cached across both A40s
(logit-lens sanity exact).

## Parts
- **corr**: per-layer whitened correlation diagnostics — bulk ξ (floor+exp on trace),
  persistent fraction, Ho-Kalman effective mode count. Compare trend to GPT-2 small.
- **predict**: layer-12 completion, learned-φ MLP vs bilinear vs MPS+const (D16), MSE.

## Reproduce
```bash
python scripts/cache_residuals.py --model gpt2-medium --build-only --num-sequences 6000
python scripts/cache_residuals.py --model gpt2-medium --device cuda:0 --start 0 --end 3000 --layers 0 4 8 12 16 20 24
CUDA_VISIBLE_DEVICES=1 python scripts/cache_residuals.py --model gpt2-medium --device cuda:0 --start 3000 --end 6000 --layers 0 4 8 12 16 20 24
python scripts/exp11_scale.py --part corr    --device cuda:0
python scripts/exp11_scale.py --part predict --layer 12 --device cuda:0
python scripts/plot_exp11.py
```

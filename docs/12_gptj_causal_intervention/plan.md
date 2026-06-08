# Experiment 12 — TN-parameterized causal intervention at GPT-J scale · Plan

**Motivation.** Exp 02–11 exhausted the *readout / completion* probe family (B4/B5/B6)
and found no MPS mechanism advantage. FutureLens's *strong* method was different — a
**causal intervention** (transplant a donor state into a learned soft-prompt context of
the frozen model and read its output). This is the one remaining matched-to-FutureLens
lever, tested at FutureLens's own scale (GPT-J-6B).

## Model
- Frozen GPT-J-6B (fp16, one A40, `from_pretrained_no_processing`).
- Donor map `g`: observed residual trajectory `(r^ℓ_{t-m+1..t})` at layer ℓ=14 (FutureLens's
  mid-layer sweet spot) → donor vector. Variants: **single** (last site only ≈ FutureLens
  m=1), **mlp**, **mps** (learned φ + const-channel MPS).
- Intervention: per-horizon learned soft prompt (length P) at `resid_pre[0]`; the donor
  replaces `resid_pre[ℓ]` at the last prompt position; read logits there. GPT-J's rotary
  positions make the soft-prompt override clean.

## Targets / metric
**FutureLens metric:** agreement with the model's *own* future token (teacher = argmax of
`decode(r^L_{t+s})`), not the realized corpus token (which is near-unpredictable 2–4 ahead).
Train CE to the teacher token; report per-horizon top-1 agreement + surprisal.

## Questions
1. Does the causal intervention beat the **readout** baseline (FutureLens's central claim)?
2. Does the **MPS** donor map beat **single-state** (does the trajectory help) and **MLP**
   (does the TN structure help) in the intervention setting?

## Reproduce
```bash
python scripts/cache_residuals.py --model gpt-j-6b --build-only --num-sequences 2000
python scripts/cache_residuals.py --model gpt-j-6b --device cuda:0 --start 0 --end 1000 --layers 14 --fp16 --no-process
CUDA_VISIBLE_DEVICES=1 python scripts/cache_residuals.py --model gpt-j-6b --device cuda:0 --start 1000 --end 2000 --layers 14 --fp16 --no-process
python scripts/exp12_causal.py --device cuda:0 --kinds single mlp mps
```

# Experiment 02 — GPT-2 Completion: Baselines vs MPS · Plan

**Goal (briefing Phases 3–4, model families B1/B2/B4).** Test whether the
finite-correlation-length bulk found in Experiment 01 gives an MPS a *predictive*
advantage over parameter-matched baselines, at the layers the diagnostics flagged.

## Question
Does an MPS readout beat parameter-matched multi-site linear / MLP probes at
completing future final-layer residuals — and does its performance improve with
bond dimension $D$ in a way consistent with the measured correlation structure?

## Task
From an observed window of source-layer-$\ell$ residuals (PCA-whitened, $p=64$) over
$m=8$ positions, predict the $n=4$ future **final-layer** residuals $r^L_{t+1..t+4}$;
decode through the frozen unembedding for token-level metrics (off-by-one
$r^L_{t+s}\to x_{t+s+1}$). This is the FutureLens setup with a multi-site window.

## Models (parameter counts reported)
- **B1** multi-site linear · **B2** MLP (hidden 256) · **B4** MPS readout (env, $D^2$
  hidden), $D\in\{4,8,16,32\}$.

## Layers
- **6** — shortest bulk $\xi$ (best MPS candidate per Exp 01).
- **12** — long-range / large persistent subspace (control); also autoregressive
  (source = target = final layer).

## Metrics (briefing §7)
NMSE per horizon; teacher KL (decode predicted vs true $r^L$); top-1 agreement; CE on
the realized token. Key plots: NMSE/KL vs $D$, and KL vs horizon $s$.

## Decision rule
- MPS beats matched baselines where bulk $\xi$ is short ⇒ **strong** evidence.
- MPS beats only single-site / improves with $D$ but loses to MLP ⇒ **weak** evidence.
- MPS needs huge $D$ / never competitive ⇒ negative evidence (still informative).

## Reproduce
```bash
python scripts/train_probes.py --layer 6  --device cuda:0
CUDA_VISIBLE_DEVICES=1 python scripts/train_probes.py --layer 12 --device cuda:0
python scripts/plot_probes.py
```

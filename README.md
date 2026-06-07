# Tensor Network FutureLens

Does a transformer's residual stream, read **along token position** at a fixed layer,
behave like a 1D many-body chain with **finite correlation length**? If so, a **matrix
product state (MPS)** is the right model for completing future residual sites, because
a bond-$D$ MPS represents a connected two-point function with up to $D^2-1$ exponential
modes. This repo measures that structure in GPT-2 and tests whether it gives an MPS a
predictive edge over parameter-matched baselines.

Full project spec: [`briefing.md`](briefing.md). Synthesized theory + cited-paper
notes: [`docs/literature_review.md`](docs/literature_review.md).

## Status

| # | experiment | result |
|---|---|---|
| 00 | [Synthetic validation](docs/00_synthetic_validation/summary.md) | ✅ pipeline validated: AR(1) $\xi$ recovered to ~3%; **transfer-matrix counting law $D\ge\sqrt{M+1}$ confirmed**; MPS layer correct (beats linear 8× more param-efficiently on a multiplicative task) |
| 01 | [GPT-2 correlation diagnostics](docs/01_gpt2_correlations/summary.md) | ✅ **finite-$\xi$ bulk + long-range persistent subspace**; bulk $\xi$ shortest in middle layers (6–8, $\xi\approx3$–4); persistent subspace grows with depth |
| 02 | [GPT-2 completion: baselines vs MPS](docs/02_gpt2_baselines_mps/summary.md) | ⚠️ **weak**: MPS *readout* (PCA, no const) improves with $D$ but loses to baselines |
| 03 | [Constant channel + learned φ](docs/03_const_learned_phi/summary.md) | ↗️ **const channel** lifts MPS to the linear baseline; **learned φ** is the big lever (helps all); MPS+const+learned-φ **ties/marginally beats** the best baseline at fewer params, saturates by $D{=}16$ |
| 04 | [B5 masked-MPS completion](docs/04_masked_mps/summary.md) | ➖ B5 masked ≈ B4 readout (marginally better KL/top-1); geometry isn't the missing ingredient |
| 05 | [Connected-only + TI transfer spectrum](docs/05_connected_transfer/summary.md) | ❎ **clarifying negative**: removing the persistent subspace doesn't help the MPS; trained transfer ξ≈2 doesn't track empirical ξ (3.3→7.8) — the MPS's *mechanism* isn't what carries predictivity |

**One-line takeaway:** the finite-correlation-length **premise survives** in GPT-2
residuals (Exp 01), and with the right architecture (constant channel + learned φ) an
MPS is **competitive** with the best parameter-matched baseline (Exp 03–04) — but the
**MPS-specific transfer-matrix mechanism is not what carries the predictive signal**
(Exp 05): a learned feature map + light nonlinearity does as well. The most informative
next directions test the *generative/representational* claim directly (a Born-machine
MPS of the residual stream; transfer ξ vs empirical with fixed φ) and **scale** to a
larger model — see Exp 05 "where this leaves the project".

## Layout

```
src/tn_futurelens/      # library (importable as `tn_futurelens`)
  data/                 # synthetic processes, windowing (+off-by-one), tokenisation, activation cache
  models/               # MPS readout (+transfer spectrum), phi feature maps, baselines B0–B3
  analysis/             # two-point correlations, exp/Prony/Hankel fits, transfer-mode diagnostics
  training/             # losses (MSE/NMSE/cosine, teacher-KL/CE), training loop
  utils/                # seed, config/provenance, logging, plotting
scripts/                # cache_residuals, compute_correlations, train_probes, plot_probes, run_synthetic_validation
docs/<NN_experiment>/   # plan.md + summary.md (+ figures/) per experiment
tests/                  # 29 unit tests (transfer spectrum, Prony recovery, off-by-one, AR/power-law)
results/{tables,runs}/  # CSV/JSON outputs (gitignored; reproducible from scripts)
```

## Setup

```bash
bash scripts/runpod_setup.sh          # uv sync; creates .env template
# fill .env (HF_TOKEN, GH_TOKEN); ANTHROPIC/WANDB optional (we log to TensorBoard + JSON/CSV)
uv sync --extra interp                 # adds transformer-lens (<3, legacy API), transformers (<4.44), datasets
source scripts/runpod_activate.sh
```

Hardware used: 2× NVIDIA A40 (48GB), torch 2.4.1+cu124.

## Reproduce

```bash
.venv/bin/python -m pytest -q                       # unit tests
.venv/bin/python scripts/run_synthetic_validation.py   # Exp 00
# Exp 01–02 (GPT-2): cache residuals across both GPUs, then diagnose + train
.venv/bin/python scripts/cache_residuals.py --build-only --num-sequences 8000
.venv/bin/python scripts/cache_residuals.py --device cuda:0 --start 0 --end 4000
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/cache_residuals.py --device cuda:0 --start 4000 --end 8000
.venv/bin/python scripts/compute_correlations.py        # Exp 01
.venv/bin/python scripts/train_probes.py --layer 6 --device cuda:0   # Exp 02
.venv/bin/python scripts/plot_probes.py
```

## Conventions
`snake_case` everything (except class names). Markdown math uses `$...$`. Each
experiment has `docs/<exp>/{plan.md,summary.md}` with inline plots. Every result is
written to disk (JSON/CSV) so figures are reproducible without a dashboard.

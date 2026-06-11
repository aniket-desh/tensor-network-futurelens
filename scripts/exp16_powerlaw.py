#!/usr/bin/env python
r"""Exp 16D — power-law vs exponential decay of bulk correlations across block scales.

Sprint-1 found block coarse-graining leaves ξ scale-invariant in block units (≈8 at
b=1,2,4,8) and raises the mode count — suggestive of self-similar/critical structure
rather than a gapped chain. This diagnostic makes that quantitative:

1. compute whitened connected correlations C(Δ) at block sizes b ∈ {1,2,4,8};
2. project out the PERSISTENT subspace first (Ho-Kalman modes with |λ| > 0.9), so the
   fits see only the decaying bulk: we use the trace-norm correlation curve of the
   residual after removing the top persistent PCA directions of the lag-averaged
   long-range correlation;
3. fit log C vs Δ (exponential) and log C vs log Δ (power law) over Δ ∈ [2, Δmax],
   compare R² / AIC at every scale.

If the bulk is genuinely scale-free, the power law should fit as well or better than
the exponential at every block size, with a roughly b-independent exponent α.

  python scripts/exp16_powerlaw.py --layers 6 8 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tn_futurelens.analysis.correlations import CorrelationAccumulator
from tn_futurelens.models.phi import PCAPhi
from tn_futurelens.utils.logging import get_logger

LOG = get_logger("exp16d")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gpt2_exp16_powerlaw"


def block_mean(V, b):
    S, T, p = V.shape
    Tb = (T // b) * b
    return V[:, :Tb].reshape(S, T // b, b, p).mean(2)


def corr_curve(layer, b, p, pca, shards, max_delta, n_persist=8):
    """Whitened two-point correlations; persistent directions removed.

    Persistent subspace estimated from the symmetrized long-lag correlation
    (mean over Δ in the top half of the lag range): its top-`n_persist` eigvecs
    are projected out of the per-Δ correlation matrices before taking norms.
    """
    md = max(10, max_delta // b)
    acc = CorrelationAccumulator(p, md, device="cpu", dtype=torch.float32)
    for sp in shards:
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        V = pca(sh["residuals"][layer][:, 1:, :].float())
        acc.update(block_mean(V, b))
        del sh, V
    res = acc.finalize(whiten=True)
    C = res.Chat.numpy()                              # [md+1, p, p]
    # persistent directions from long-lag average (symmetrized)
    tail = C[md // 2:]
    Csym = ((tail + np.transpose(tail, (0, 2, 1))) / 2).mean(0)
    w, U = np.linalg.eigh(Csym)
    P = U[:, np.argsort(-np.abs(w))[:n_persist]]      # [p, k]
    proj = np.eye(p) - P @ P.T
    curve = np.array([np.linalg.norm(proj @ C[d] @ proj) for d in range(md + 1)])
    return curve / max(curve[1], 1e-12), md


def fit_decays(curve, dmin=2):
    """Return (exp fit, power fit): R², AIC, parameter."""
    md = len(curve) - 1
    ds = np.arange(dmin, md + 1)
    y = np.log(np.clip(curve[dmin:], 1e-12, None))
    out = {}
    for name, x in (("exp", ds.astype(float)), ("pow", np.log(ds.astype(float)))):
        A = np.stack([x, np.ones_like(x)], 1)
        coef, res_, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        n = len(y)
        out[name] = {"slope": round(float(coef[0]), 4),
                     "r2": round(1 - ss_res / max(ss_tot, 1e-12), 4),
                     "aic": round(n * np.log(max(ss_res / n, 1e-12)) + 4, 2),
                     "param": round(float(-1 / coef[0]), 2) if name == "exp" and coef[0] < 0
                              else round(float(-coef[0]), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--max-delta", type=int, default=48)
    ap.add_argument("--n-persist", type=int, default=8)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(CACHE.glob("shard_*.pt"))
    rows, curves = [], {}
    for layer in args.layers:
        s0 = torch.load(shards[0], map_location="cpu", weights_only=False)
        d_model = s0["meta"]["d_model"]
        r0 = s0["residuals"][layer][:, 1:, :].reshape(-1, d_model).float()
        gen = torch.Generator().manual_seed(0)
        pca = PCAPhi(d_model, args.p).fit(r0[torch.randperm(r0.shape[0], generator=gen)[:60000]])
        del s0, r0
        for b in args.blocks:
            curve, md = corr_curve(layer, b, args.p, pca, shards, args.max_delta,
                                   args.n_persist)
            fits = fit_decays(curve)
            row = {"layer": layer, "b": b, "max_delta": md,
                   "exp": fits["exp"], "pow": fits["pow"],
                   "winner": "pow" if fits["pow"]["aic"] < fits["exp"]["aic"] else "exp"}
            rows.append(row)
            curves[f"L{layer}_b{b}"] = [round(float(x), 5) for x in curve]
            LOG.info(f"L{layer} b={b}: exp R2={fits['exp']['r2']} xi={fits['exp']['param']} | "
                     f"pow R2={fits['pow']['r2']} alpha={fits['pow']['param']} | "
                     f"AIC winner: {row['winner']}")
    json.dump({"rows": rows, "curves": curves},
              open(OUTDIR / "powerlaw_vs_exp.json", "w"), indent=2)
    LOG.info(f"wrote {OUTDIR/'powerlaw_vs_exp.json'}")


if __name__ == "__main__":
    main()

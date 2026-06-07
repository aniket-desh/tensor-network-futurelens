#!/usr/bin/env python
r"""Exp 08 — systematic horizon (n) and bond (D) sweeps (next-step #5).

Key plots:
  * (MPS - best baseline) NMSE vs horizon n  -- does the MPS edge grow with horizon?
  * MPS NMSE vs bond D, with the empirical implied-D from Exp 06 marked.

All probes use learned phi (+const for MPS). Layer 6. Build windows once at n=8 and
slice for smaller n. Run --part horizon and --part bond on the two GPUs in parallel.

  python scripts/exp08_sweep.py --part horizon --device cuda:0
  python scripts/exp08_sweep.py --part bond    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tn_futurelens.models.baselines import BilinearProbe, MultiSiteMLP
from tn_futurelens.models.mps import MPSReadout
from tn_futurelens.models.phi import LearnedLinearPhi, PCAPhi
from tn_futurelens.models.probes import PhiHead
from tn_futurelens.training.eval import build_completion_dataset, nmse_per_horizon, standardize_targets
from tn_futurelens.training.train import train_regression_probe
from tn_futurelens.utils.logging import get_logger
from tn_futurelens.utils.seed import set_seed

LOG = get_logger("exp08")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gpt2_exp08"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["horizon", "bond"], required=True)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=150000)
    ap.add_argument("--epochs", type=int, default=90)
    args = ap.parse_args()
    set_seed(0); OUTDIR.mkdir(parents=True, exist_ok=True)
    dev, p, m = args.device, args.p, args.m

    X, Y8_raw, Tk, meta = build_completion_dataset(CACHE, args.layer, m, 8, args.n_windows)
    flat = X.reshape(-1, X.shape[-1])
    pca = PCAPhi(X.shape[-1], p).fit(flat[torch.randperm(flat.shape[0])[:60000]]).to(dev)
    d_out = Y8_raw.shape[-1]
    ntr = int(0.85 * X.shape[0])
    Xtr, Xva = X[:ntr], X[ntr:]

    def lp():
        return LearnedLinearPhi(meta["d_model"], p).init_from_pca(pca).to(dev)

    def fit(model, Yraw_n):
        Yz, _, _ = standardize_targets(Yraw_n, ntr)
        train_regression_probe(model, Xtr, Yz[:ntr], Xva, Yz[ntr:], epochs=args.epochs, lr=1.5e-3,
                               batch_size=4096, device=dev, patience=15, seed=0)
        return float(np.mean(nmse_per_horizon(model, Xva, Yz[ntr:], dev)))

    if args.part == "horizon":
        out = {"layer": args.layer, "m": m, "by_n": {}}
        for n in [1, 2, 4, 8]:
            Yraw = Y8_raw[:, :n]
            models = {
                "mlp": PhiHead(lp(), MultiSiteMLP(p, m, d_out, n, hidden=256, depth=2)),
                "bilinear": PhiHead(lp(), BilinearProbe(p, m, d_out, n, rank=64)),
                "mps": PhiHead(lp(), MPSReadout(p=p, D=16, n_sites=m, readout="env",
                               out_dim=d_out, n_heads=n, const_channel=True, seed=0)),
            }
            r = {name: round(fit(model, Yraw), 4) for name, model in models.items()}
            best_base = min(r["mlp"], r["bilinear"])
            r["mps_minus_best_baseline"] = round(r["mps"] - best_base, 4)
            out["by_n"][str(n)] = r
            LOG.info(f"  n={n}: mlp={r['mlp']:.3f} bilinear={r['bilinear']:.3f} mps={r['mps']:.3f} "
                     f"(mps-best={r['mps_minus_best_baseline']:+.3f})")
        json.dump(out, open(OUTDIR / "horizon.json", "w"), indent=2)
    else:
        n = 4
        Yraw = Y8_raw[:, :n]
        out = {"layer": args.layer, "m": m, "n": n, "by_D": {}}
        for D in [2, 4, 8, 16, 32]:
            model = PhiHead(lp(), MPSReadout(p=p, D=D, n_sites=m, readout="env",
                            out_dim=d_out, n_heads=n, const_channel=True, seed=0))
            out["by_D"][str(D)] = round(fit(model, Yraw), 4)
            LOG.info(f"  D={D}: nmse={out['by_D'][str(D)]:.4f}")
        # reference baselines
        out["mlp"] = round(fit(PhiHead(lp(), MultiSiteMLP(p, m, d_out, n, hidden=256, depth=2)), Yraw), 4)
        out["bilinear"] = round(fit(PhiHead(lp(), BilinearProbe(p, m, d_out, n, rank=64)), Yraw), 4)
        json.dump(out, open(OUTDIR / "bond.json", "w"), indent=2)
    LOG.info(f"wrote {args.part}.json")


if __name__ == "__main__":
    main()

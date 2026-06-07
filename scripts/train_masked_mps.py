#!/usr/bin/env python
r"""Exp 04 — B5 masked-MPS completion (next-steps #3).

Compares, with learned-phi + constant channel (the best config from Exp 03):
  mlp                      (reference baseline)
  b4_readout   (MPSReadout, observed sites only -> heads)
  b5_masked    (MaskedMPSCompletion, m+n chain with learned future mask sites)
sweeping bond dimension D. B5 is the closer-to-theory "clamp observed, complete future"
geometry the project is actually about.

  python scripts/train_masked_mps.py --layer 6 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tn_futurelens.data.activation_cache import load_model
from tn_futurelens.models.baselines import MultiSiteMLP
from tn_futurelens.models.masked_mps import MaskedMPSCompletion
from tn_futurelens.models.mps import MPSReadout
from tn_futurelens.models.phi import LearnedLinearPhi, PCAPhi
from tn_futurelens.models.probes import PhiHead
from tn_futurelens.training.eval import (
    build_completion_dataset,
    standardize_targets,
    summarize,
)
from tn_futurelens.training.train import train_regression_probe
from tn_futurelens.utils.logging import get_logger
from tn_futurelens.utils.seed import set_seed

LOG = get_logger("b5")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gpt2_masked_mps"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=150000)
    ap.add_argument("--D", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--epochs", type=int, default=110)
    args = ap.parse_args()
    set_seed(0); OUTDIR.mkdir(parents=True, exist_ok=True)
    dev, p, m, n = args.device, args.p, args.m, args.n

    LOG.info(f"layer {args.layer}: building dataset")
    X, Y_raw, Tk, meta = build_completion_dataset(CACHE, args.layer, m, n, args.n_windows)
    d_out = Y_raw.shape[-1]
    flat = X.reshape(-1, X.shape[-1])
    pca = PCAPhi(X.shape[-1], p).fit(flat[torch.randperm(flat.shape[0])[:60000]]).to(dev)

    n_tr = int(0.85 * X.shape[0])
    Y, tgt_mean, tgt_std = standardize_targets(Y_raw, n_tr)
    Xtr, Ytr, Xva, Yva = X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:]
    Yva_raw, Tkva = Y_raw[n_tr:], Tk[n_tr:]
    tm_d, ts_d = tgt_mean.to(dev), tgt_std.to(dev)
    gpt = load_model("gpt2", device=dev)

    def learned_phi():
        return LearnedLinearPhi(meta["d_model"], p).init_from_pca(pca).to(dev)

    configs = [("mlp", lambda: PhiHead(pca, MultiSiteMLP(p, m, d_out, n, hidden=256, depth=2)))]
    for D in args.D:
        configs.append((f"b4_readout_D{D}",
                        lambda D=D: PhiHead(learned_phi(),
                            MPSReadout(p=p, D=D, n_sites=m, readout="env", out_dim=d_out,
                                       n_heads=n, const_channel=True, seed=0))))
        configs.append((f"b5_masked_D{D}",
                        lambda D=D: PhiHead(learned_phi(),
                            MaskedMPSCompletion(p=p, D=D, m=m, n=n, d_out=d_out,
                                                const_channel=True, seed=0))))

    results = []
    for name, ctor in configs:
        model = ctor()
        train_regression_probe(model, Xtr, Ytr, Xva, Yva, epochs=args.epochs, lr=1.5e-3,
                               batch_size=4096, device=dev, patience=20, seed=0)
        rec = summarize(name, model, Xva, Yva, Yva_raw, Tkva, tm_d, ts_d, gpt, dev)
        results.append(rec)
        LOG.info(f"  {name:16s} nmse={rec['nmse_mean']:.3f} kl={rec['kl_mean']:.3f} "
                 f"top1={rec['top1_mean']:.3f} params={rec['n_params']}")

    out = {"layer": args.layer, "m": m, "n": n, "p": p, "n_windows": int(X.shape[0]),
           "d_model": d_out, "results": results}
    json.dump(out, open(OUTDIR / f"layer_{args.layer}.json", "w"), indent=2)
    LOG.info(f"wrote {OUTDIR/f'layer_{args.layer}.json'}")


if __name__ == "__main__":
    main()

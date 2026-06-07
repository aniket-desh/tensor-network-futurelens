#!/usr/bin/env python
r"""Exp 11 — scale up to GPT-2 medium (next-step #6).

Does the GPT-2-small picture hold at larger scale?
  --part corr    : correlation diagnostics (bulk xi + Ho-Kalman mode count) per layer
  --part predict : best predictive comparison (MLP vs MPS+const, learned phi) at a mid layer

  python scripts/exp11_scale.py --part corr    --device cuda:0
  python scripts/exp11_scale.py --part predict --layer 12 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tn_futurelens.analysis.correlations import CorrelationAccumulator
from tn_futurelens.analysis.exp_fits import exp_plus_const_fit, single_exponential_fit
from tn_futurelens.analysis.realization import ho_kalman
from tn_futurelens.data.activation_cache import iter_shards, load_model
from tn_futurelens.models.baselines import BilinearProbe, MultiSiteMLP
from tn_futurelens.models.mps import MPSReadout
from tn_futurelens.models.phi import LearnedLinearPhi, PCAPhi
from tn_futurelens.models.probes import PhiHead
from tn_futurelens.training.eval import (
    build_completion_dataset,
    nmse_per_horizon,
    standardize_targets,
    token_metrics,
)
from tn_futurelens.training.train import train_regression_probe
from tn_futurelens.utils.logging import get_logger
from tn_futurelens.utils.seed import set_seed

LOG = get_logger("exp11")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2-medium" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gpt2_medium"
P = 64
MAX_DELTA = 40


def run_corr(device):
    meta = next(iter_shards(CACHE))["meta"]
    layers = meta["layers"]
    LOG.info(f"gpt2-medium corr: layers={layers} d_model={meta['d_model']}")
    # fit PCA per layer from shard 0
    s0 = next(iter_shards(CACHE))
    phis = {}
    for l in layers:
        r = s0["residuals"][l][:, 1:, :].reshape(-1, meta["d_model"]).float()
        phis[l] = PCAPhi(meta["d_model"], P).fit(r[torch.randperm(r.shape[0])[:60000]]).to(device)
    accs = {l: CorrelationAccumulator(P, MAX_DELTA, device=device, dtype=torch.float32) for l in layers}
    for sh in iter_shards(CACHE):
        for l in layers:
            accs[l].update(phis[l](sh["residuals"][l][:, 1:, :].float().to(device)))
    rows = []
    for l in layers:
        res = accs[l].finalize(whiten=True)
        d = res.deltas.cpu().numpy()
        op = res.operator(whitened=True).cpu().numpy()
        tr = res.trace(whitened=True).cpu().numpy()
        ec = exp_plus_const_fit(d, tr, 1, MAX_DELTA)
        r = ho_kalman(res.Chat.cpu().numpy(), rel_threshold=0.03, max_rank=30)
        sv = r.singular_values / r.singular_values[0]
        rows.append({"layer": l, "bulk_xi": round(ec.xi, 2), "persistent_frac": round(ec.persistent_frac, 3),
                     "eff_modes": int(np.sum(sv > 0.05)),
                     "op_xi": round(single_exponential_fit(d, op, 1, 12).xi, 2)})
        LOG.info(f"  L{l:2d}: bulk_xi={ec.xi:.2f} persist_frac={ec.persistent_frac:.2f} eff_modes={rows[-1]['eff_modes']}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows, "n_layers": meta["n_layers"]}, open(OUTDIR / "corr.json", "w"), indent=2)


def run_predict(layer, device):
    set_seed(0)
    m, n = 8, 4
    X, Y_raw, Tk, meta = build_completion_dataset(CACHE, layer, m, n, 150000)
    d_out = Y_raw.shape[-1]
    flat = X.reshape(-1, X.shape[-1])
    pca = PCAPhi(X.shape[-1], P).fit(flat[torch.randperm(flat.shape[0])[:60000]]).to(device)
    ntr = int(0.85 * X.shape[0])
    Yz, mean, std = standardize_targets(Y_raw, ntr)
    Xva, Yva_z, Yva_raw, Tkva = X[ntr:], Yz[ntr:], Y_raw[ntr:], Tk[ntr:]
    gpt = load_model("gpt2-medium", device=device)

    def lp():
        return LearnedLinearPhi(meta["d_model"], P).init_from_pca(pca).to(device)
    models = {
        "mlp": PhiHead(lp(), MultiSiteMLP(P, m, d_out, n, hidden=256, depth=2)),
        "bilinear": PhiHead(lp(), BilinearProbe(P, m, d_out, n, rank=64)),
        "mps_const_D16": PhiHead(lp(), MPSReadout(p=P, D=16, n_sites=m, readout="env",
                                  out_dim=d_out, n_heads=n, const_channel=True, seed=0)),
    }
    out = {"layer": layer, "d_model": d_out, "results": []}
    for name, model in models.items():
        train_regression_probe(model, X[:ntr], Yz[:ntr], Xva, Yva_z, epochs=90, lr=1.5e-3,
                               batch_size=4096, device=device, patience=15, seed=0)
        nmse = float(np.mean(nmse_per_horizon(model, Xva, Yva_z, device)))
        kl, top1 = token_metrics(model, Xva, Yva_raw, Tkva, mean.to(device), std.to(device), gpt, device)
        out["results"].append({"name": name, "nmse_mean": round(nmse, 4),
                               "kl_mean": round(float(np.mean(kl)), 4), "top1_mean": round(float(np.mean(top1)), 4)})
        LOG.info(f"  {name:14s} nmse={nmse:.3f} kl={np.mean(kl):.3f} top1={np.mean(top1):.3f}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUTDIR / f"predict_layer_{layer}.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["corr", "predict"], required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.part == "corr":
        run_corr(args.device)
    else:
        run_predict(args.layer, args.device)


if __name__ == "__main__":
    main()

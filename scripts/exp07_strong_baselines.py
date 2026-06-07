#!/usr/bin/env python
r"""Exp 07 — strong sequence baselines + KL/logit objective (next-steps #3,#4).

Does the MPS beat baselines that are GOOD at local sequence structure (attention, 1D
conv, low-rank bilinear), not just dense MLPs? And does training on the FutureLens
objective (teacher KL through the frozen unembed) change the ranking vs residual MSE?

All probes use a learned phi (init PCA) + constant channel where applicable, for a fair
comparison. One process per (objective) so the two A40s run MSE and KL in parallel.

  python scripts/exp07_strong_baselines.py --objective mse --layer 6 --device cuda:0
  python scripts/exp07_strong_baselines.py --objective kl  --layer 6 --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tn_futurelens.data.activation_cache import load_model
from tn_futurelens.models.baselines import (
    AttentionPool,
    BilinearProbe,
    Conv1DProbe,
    MultiSiteMLP,
)
from tn_futurelens.models.mps import MPSReadout
from tn_futurelens.models.phi import LearnedLinearPhi, PCAPhi
from tn_futurelens.models.probes import PhiHead
from tn_futurelens.training.eval import (
    build_completion_dataset,
    nmse_per_horizon,
    standardize_targets,
    token_metrics,
)
from tn_futurelens.utils.logging import count_parameters, get_logger
from tn_futurelens.utils.seed import set_seed

LOG = get_logger("exp07")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gpt2_exp07"


def decode(gpt, r):  # r [B,768] -> logits [B,vocab]
    return gpt.ln_final(r) @ gpt.W_U + gpt.b_U


def train(model, Xtr, Ytr_z, Ytr_raw, Xva, Yva_z, Yva_raw, *, objective, gpt, tgt_mean, tgt_std,
          device, epochs=90, lr=1.5e-3, bs=None, patience=15):
    model = model.to(device)
    Xtr, Ytr_z, Ytr_raw = Xtr.to(device), Ytr_z.to(device), Ytr_raw.to(device)
    Xva, Yva_z = Xva.to(device), Yva_z.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(0)
    bs = bs or (1024 if objective == "kl" else 4096)
    best, best_state, since = float("inf"), copy.deepcopy(model.state_dict()), 0
    n = Xtr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred_z = model(Xtr[idx])
            if objective == "mse":
                loss = ((pred_z - Ytr_z[idx]) ** 2).sum(-1).mean()
            else:  # kl through the frozen unembed
                pred_raw = pred_z * tgt_std + tgt_mean
                tgt_raw = Ytr_raw[idx]
                loss = 0.0
                for s in range(pred_raw.shape[1]):
                    ls = F.log_softmax(decode(gpt, pred_raw[:, s]), -1)
                    lt = F.log_softmax(decode(gpt, tgt_raw[:, s]), -1)
                    loss = loss + (lt.exp() * (lt - ls)).sum(-1).mean()
                loss = loss / pred_raw.shape[1]
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean(nmse_per_horizon(model, Xva, Yva_z, device)))
        if v < best - 1e-5:
            best, best_state, since = v, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["mse", "kl"], required=True)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=150000)
    ap.add_argument("--epochs", type=int, default=90)
    args = ap.parse_args()
    set_seed(0); OUTDIR.mkdir(parents=True, exist_ok=True)
    dev, p, m, n = args.device, args.p, args.m, args.n

    X, Y_raw, Tk, meta = build_completion_dataset(CACHE, args.layer, m, n, args.n_windows)
    d_out = Y_raw.shape[-1]
    flat = X.reshape(-1, X.shape[-1])
    pca = PCAPhi(X.shape[-1], p).fit(flat[torch.randperm(flat.shape[0])[:60000]]).to(dev)
    ntr = int(0.85 * X.shape[0])
    Yz, tgt_mean, tgt_std = standardize_targets(Y_raw, ntr)
    Xtr, Ytr_z, Ytr_raw = X[:ntr], Yz[:ntr], Y_raw[:ntr]
    Xva, Yva_z, Yva_raw, Tkva = X[ntr:], Yz[ntr:], Y_raw[ntr:], Tk[ntr:]
    tm_d, ts_d = tgt_mean.to(dev), tgt_std.to(dev)
    gpt = load_model("gpt2", device=dev)

    def lp():
        return LearnedLinearPhi(meta["d_model"], p).init_from_pca(pca).to(dev)

    builders = {
        "mlp": lambda: PhiHead(lp(), MultiSiteMLP(p, m, d_out, n, hidden=256, depth=2)),
        "attention": lambda: PhiHead(lp(), AttentionPool(p, m, d_out, n, d_model=128, n_heads=4)),
        "conv1d": lambda: PhiHead(lp(), Conv1DProbe(p, m, d_out, n, hidden=128, layers=2)),
        "bilinear": lambda: PhiHead(lp(), BilinearProbe(p, m, d_out, n, rank=64)),
        "mps_const_D16": lambda: PhiHead(lp(), MPSReadout(p=p, D=16, n_sites=m, readout="env",
                                          out_dim=d_out, n_heads=n, const_channel=True, seed=0)),
    }
    results = []
    for name, build in builders.items():
        model = build()
        model = train(model, Xtr, Ytr_z, Ytr_raw, Xva, Yva_z, Yva_raw, objective=args.objective,
                      gpt=gpt, tgt_mean=tm_d, tgt_std=ts_d, device=dev, epochs=args.epochs)
        nmse = nmse_per_horizon(model, Xva, Yva_z, dev)
        kl, top1 = token_metrics(model, Xva, Yva_raw, Tkva, tm_d, ts_d, gpt, dev)
        rec = {"name": name, "objective": args.objective, "n_params": count_parameters(model),
               "nmse_mean": round(float(np.mean(nmse)), 4),
               "kl_mean": round(float(np.mean(kl)), 4), "top1_mean": round(float(np.mean(top1)), 4)}
        results.append(rec)
        LOG.info(f"  [{args.objective}] {name:14s} nmse={rec['nmse_mean']:.3f} "
                 f"kl={rec['kl_mean']:.3f} top1={rec['top1_mean']:.3f} params={rec['n_params']}")

    out = {"layer": args.layer, "objective": args.objective, "results": results}
    json.dump(out, open(OUTDIR / f"layer_{args.layer}_{args.objective}.json", "w"), indent=2)
    LOG.info(f"wrote layer_{args.layer}_{args.objective}.json")


if __name__ == "__main__":
    main()

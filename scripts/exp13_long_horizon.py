#!/usr/bin/env python
r"""Exp 13 — long-horizon completion under the KL objective with strong baselines.

Per the analysis: bond dim is already swept (Exp 08, saturates at baseline by D~16); the
high-value test is LONGER HORIZONS (the MPS-vs-baseline gap shrank with n in Exp 08, so
if any MPS regime exists it is at n>>8). Train completion probes with the KL/logit
objective (Exp 07 showed objective >> architecture) and strong baselines, and track
whether the MPS gap turns POSITIVE or merely stays tied at n=16,32.

  python scripts/exp13_long_horizon.py --horizons 4 8 16 --device cuda:0 --tag a
  python scripts/exp13_long_horizon.py --horizons 32     --device cuda:0 --tag b
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
from tn_futurelens.models.baselines import BilinearProbe, Conv1DProbe, MultiSiteMLP
from tn_futurelens.models.mps import MPSReadout
from tn_futurelens.models.phi import LearnedLinearPhi, PCAPhi
from tn_futurelens.models.probes import PhiHead
from tn_futurelens.training.eval import build_completion_dataset, standardize_targets
from tn_futurelens.utils.logging import get_logger
from tn_futurelens.utils.seed import set_seed

LOG = get_logger("exp13")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gpt2_longhorizon"
NMAX = 32


def decode(gpt, r):
    return gpt.ln_final(r) @ gpt.W_U + gpt.b_U


def train_kl(probe, Xtr, Rtr, Xva, Rva, ttok_va, n, *, gpt, mean, std, device, epochs, bs):
    probe = probe.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1.5e-3)
    gen = torch.Generator().manual_seed(0)
    ntr = Xtr.shape[0]
    best, best_state, since = -1.0, copy.deepcopy(probe.state_dict()), 0
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(ntr, generator=gen)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = probe(Xtr[idx].to(device)) * std + mean        # [B,n,768]
            rt = Rtr[idx].to(device)
            loss = 0.0
            for s in range(n):
                sl = F.log_softmax(decode(gpt, pred[:, s]), -1)
                tl = F.softmax(decode(gpt, rt[:, s]), -1)
                loss = loss + (tl * (tl.clamp_min(1e-12).log() - sl)).sum(-1).mean()
            (loss / n).backward(); opt.step()
        # eval: teacher-token top-1 agreement (mean over horizons)
        probe.eval()
        with torch.no_grad():
            acc = []
            for i in range(0, Xva.shape[0], 256):
                pred = probe(Xva[i:i + 256].to(device)) * std + mean
                for s in range(n):
                    a = (decode(gpt, pred[:, s]).argmax(-1) == ttok_va[i:i + 256, s].to(device)).float()
                    if len(acc) <= s:
                        acc.append([])
                    acc[s].append(a)
        per_h = [torch.cat(a).mean().item() for a in acc]
        score = float(np.mean(per_h))
        if score > best:
            best, best_state, since = score, copy.deepcopy(probe.state_dict()), 0
        else:
            since += 1
            if since >= 3:
                break
    probe.load_state_dict(best_state)
    # final per-horizon top1
    probe.eval()
    with torch.no_grad():
        acc = [[] for _ in range(n)]
        for i in range(0, Xva.shape[0], 256):
            pred = probe(Xva[i:i + 256].to(device)) * std + mean
            for s in range(n):
                acc[s].append((decode(gpt, pred[:, s]).argmax(-1) == ttok_va[i:i + 256, s].to(device)).float())
    return [torch.cat(a).mean().item() for a in acc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=160)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--d64", action="store_true", help="also run MPS D=64 (sanity)")
    args = ap.parse_args()
    set_seed(0); OUTDIR.mkdir(parents=True, exist_ok=True)
    dev, m, p = args.device, args.m, args.p

    gpt = load_model("gpt2", device=dev)  # folded; gives W_U/ln_final consistent with the cache
    X, Rfull, _, meta = build_completion_dataset(CACHE, args.layer, m, NMAX, args.n_windows)
    d_out = Rfull.shape[-1]
    LOG.info(f"{X.shape[0]} windows; traj {tuple(X.shape)}; future resid {tuple(Rfull.shape)}")
    # teacher tokens for all horizons (model's own argmax at each future position)
    with torch.no_grad():
        ttok = torch.empty(Rfull.shape[0], NMAX, dtype=torch.long)
        for s in range(NMAX):
            for i in range(0, Rfull.shape[0], 512):
                ttok[i:i + 512, s] = decode(gpt, Rfull[i:i + 512, s].to(dev).float()).argmax(-1).cpu()
    flat = X.reshape(-1, d_out)
    pca = PCAPhi(d_out, p).fit(flat[torch.randperm(flat.shape[0])[:60000]]).to(dev)
    ntr = int(0.85 * X.shape[0])

    def lp():
        return LearnedLinearPhi(d_out, p).init_from_pca(pca).to(dev)

    out = {"layer": args.layer, "m": m, "by_n": {}}
    for n in args.horizons:
        R = Rfull[:, :n].float()
        Rz, mean, std = standardize_targets(R, ntr)
        mean, std = mean.to(dev), std.to(dev)
        Xtr, Xva = X[:ntr], X[ntr:]
        Rztr, Rzva = Rz[:ntr], Rz[ntr:]              # not used directly (KL on raw)
        Rtr_raw = R[:ntr]                             # true residuals for teacher KL
        ttok_va = ttok[ntr:, :n]
        builders = {
            "mlp": lambda: PhiHead(lp(), MultiSiteMLP(p, m, d_out, n, hidden=256, depth=2)),
            "conv1d": lambda: PhiHead(lp(), Conv1DProbe(p, m, d_out, n, hidden=128, layers=2)),
            "bilinear": lambda: PhiHead(lp(), BilinearProbe(p, m, d_out, n, rank=64)),
            "mps_D16": lambda: PhiHead(lp(), MPSReadout(p=p, D=16, n_sites=m, readout="env",
                                       out_dim=d_out, n_heads=n, const_channel=True, seed=0)),
        }
        if args.d64 and n <= 8:
            builders["mps_D64"] = lambda: PhiHead(lp(), MPSReadout(p=p, D=64, n_sites=m,
                                   readout="env", out_dim=d_out, n_heads=n, const_channel=True, seed=0))
        res = {}
        for name, b in builders.items():
            per_h = train_kl(b(), Xtr, Rtr_raw, Xva, None, ttok_va, n,
                             gpt=gpt, mean=mean, std=std, device=dev, epochs=args.epochs, bs=args.bs)
            res[name] = {"top1_per_h": [round(x, 4) for x in per_h],
                         "top1_mean": round(float(np.mean(per_h)), 4)}
            LOG.info(f"  n={n} {name:9s} top1_mean={res[name]['top1_mean']} (h1={per_h[0]:.3f})")
        base = min(res[k]["top1_mean"] for k in ("mlp", "conv1d", "bilinear"))
        best_base = max(res[k]["top1_mean"] for k in ("mlp", "conv1d", "bilinear"))
        res["mps_minus_best_baseline"] = round(res["mps_D16"]["top1_mean"] - best_base, 4)
        out["by_n"][str(n)] = res
        LOG.info(f"  n={n}: mps-best_baseline(top1) = {res['mps_minus_best_baseline']:+.4f}")
    json.dump(out, open(OUTDIR / f"results_{args.tag}.json", "w"), indent=2)
    LOG.info(f"wrote results_{args.tag}.json")


if __name__ == "__main__":
    main()

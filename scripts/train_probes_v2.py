#!/usr/bin/env python
r"""Exp 03 — completion probes with constant channel + learned phi (next-steps #1,#2).

Addresses the Exp 02 weak result: a pure-multiplicative MPS cannot represent the
additive/linear part of the task. We add (1) a constant channel so the MPS spans
constants + linear + interactions, and (2) a learned phi (init from PCA) trained jointly.

Conditions (per layer), all sharing one dataset of RAW observed residuals:
  linear, mlp                          (PCA-frozen phi; same references as Exp 02)
  mps_pca         (PCA-frozen, no const)        -> reproduces Exp 02
  mps_pca_const   (PCA-frozen, const channel)   -> #1
  mps_learned_const (learned phi init PCA, const)-> #1 + #2
MPS swept over D. Metrics: NMSE, teacher-KL, top-1 (decode via folded-LN unembed).

  python scripts/train_probes_v2.py --layer 6 --device cuda:0
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
from tn_futurelens.data.windows import WindowSpec, make_windows
from tn_futurelens.models.baselines import MultiSiteLinear, MultiSiteMLP
from tn_futurelens.models.mps import MPSReadout
from tn_futurelens.models.phi import LearnedLinearPhi, PCAPhi
from tn_futurelens.models.probes import PhiHead
from tn_futurelens.training.train import train_regression_probe
from tn_futurelens.utils.logging import count_parameters, get_logger
from tn_futurelens.utils.seed import set_seed

LOG = get_logger("probes_v2")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gpt2_probes_v2"


def build_dataset(layer, m, n, n_windows, device):
    """RAW observed source residuals [N,m,d] + raw target final residuals [N,n,d] + tokens."""
    shards = sorted(CACHE.glob("shard_*.pt"))
    meta = torch.load(shards[0], map_location="cpu", weights_only=False)["meta"]
    final = meta["final_layer"]
    spec = WindowSpec(m=m, n=n)
    Xs, Ys, Ts = [], [], []
    got = 0
    for sp in shards:
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        src = sh["residuals"][layer].float()                 # [S, T, d] (cpu)
        tgt = sh["residuals"][final].float()
        toks = sh["tokens"]
        S = src.shape[0]
        for i in range(S):
            w = make_windows(src[i], target=tgt[i], token_ids=toks[i], spec=spec)
            if w["anchors"].numel() == 0:
                continue
            Xs.append(w["observed"]); Ys.append(w["future"]); Ts.append(w["future_token_ids"])
            got += w["anchors"].numel()
        del sh, src, tgt
        if got >= n_windows:
            break
    X = torch.cat(Xs)[:n_windows]; Y = torch.cat(Ys)[:n_windows]; Tk = torch.cat(Ts)[:n_windows]
    return X, Y, Tk, meta


def fit_pca(X, p, device, n_sample=60000):
    flat = X.reshape(-1, X.shape[-1])
    idx = torch.randperm(flat.shape[0])[:n_sample]
    return PCAPhi(X.shape[-1], p).fit(flat[idx]).to(device)


@torch.no_grad()
def token_metrics(model, Xva, Yva_raw, Tk, tgt_mean, tgt_std, gpt, device, n_eval=4000):
    idx = torch.randperm(Xva.shape[0])[:n_eval]
    X = Xva[idx].to(device); Yt = Yva_raw[idx].to(device); tok = Tk[idx].to(device)
    pred = model(X) * tgt_std + tgt_mean
    kl, top1 = [], []
    for s in range(Yt.shape[1]):
        tl = gpt.ln_final(Yt[:, s]) @ gpt.W_U + gpt.b_U
        sl = gpt.ln_final(pred[:, s]) @ gpt.W_U + gpt.b_U
        lt, ls = F.log_softmax(tl, -1), F.log_softmax(sl, -1)
        kl.append((lt.exp() * (lt - ls)).sum(-1).mean().item())
        top1.append((sl.argmax(-1) == tl.argmax(-1)).float().mean().item())
    return kl, top1


def nmse_ph(model, Xva, Yva_z, device):
    with torch.no_grad():
        pred = model(Xva.to(device)); Y = Yva_z.to(device)
        num = ((pred - Y) ** 2).sum(-1); den = ((Y - Y.mean(0, keepdim=True)) ** 2).sum(-1)
        return [(num[:, s].mean() / den[:, s].mean().clamp_min(1e-8)).item() for s in range(Y.shape[1])]


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
    dev = args.device
    p, m, n = args.p, args.m, args.n

    LOG.info(f"layer {args.layer}: building raw-residual dataset")
    X, Y_raw, Tk, meta = build_dataset(args.layer, m, n, args.n_windows, dev)
    d_out = Y_raw.shape[-1]
    LOG.info(f"  {X.shape[0]} windows  X={tuple(X.shape)} Y={tuple(Y_raw.shape)}")
    pca = fit_pca(X, p, dev)

    n_tr = int(0.85 * X.shape[0])
    tgt_mean = Y_raw[:n_tr].mean(0, keepdim=True); tgt_std = Y_raw[:n_tr].std(0, keepdim=True).clamp_min(1e-6)
    Y = (Y_raw - tgt_mean) / tgt_std
    Xtr, Ytr, Xva, Yva = X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:]
    Yva_raw, Tkva = Y_raw[n_tr:], Tk[n_tr:]
    tm_d, ts_d = tgt_mean.to(dev), tgt_std.to(dev)

    gpt = load_model("gpt2", device=dev)

    def frozen_pca():
        return copy.deepcopy(pca)

    def learned_phi():
        return LearnedLinearPhi(meta["d_model"], p).init_from_pca(pca).to(dev)

    def make_model(kind, D=None):
        # baselines: frozen-PCA phi vs learned phi (fair comparison to the learned-phi MPS)
        if kind == "linear":
            return PhiHead(frozen_pca(), MultiSiteLinear(p, m, d_out, n))
        if kind == "linear_learned":
            return PhiHead(learned_phi(), MultiSiteLinear(p, m, d_out, n))
        if kind == "mlp":
            return PhiHead(frozen_pca(), MultiSiteMLP(p, m, d_out, n, hidden=256, depth=2))
        if kind == "mlp_learned":
            return PhiHead(learned_phi(), MultiSiteMLP(p, m, d_out, n, hidden=256, depth=2))
        # MPS variants
        const = kind.endswith("const")
        head = MPSReadout(p=p, D=D, n_sites=m, readout="env", out_dim=d_out, n_heads=n,
                          const_channel=const, seed=0)
        phi = learned_phi() if kind.startswith("mps_learned") else frozen_pca()
        return PhiHead(phi, head)

    configs = [("linear", None), ("linear_learned", None), ("mlp", None), ("mlp_learned", None)]
    for D in args.D:
        configs += [("mps_pca", D), ("mps_pca_const", D), ("mps_learned_const", D)]

    results = []
    for kind, D in configs:
        model = make_model(kind, D)
        train_regression_probe(model, Xtr, Ytr, Xva, Yva, epochs=args.epochs, lr=1.5e-3,
                               batch_size=4096, device=dev, patience=20, seed=0)
        nmse = nmse_ph(model, Xva, Yva, dev)
        kl, top1 = token_metrics(model, Xva, Yva_raw, Tkva, tm_d, ts_d, gpt, dev)
        name = kind if D is None else f"{kind}_D{D}"
        rec = {"name": name, "kind": kind, "D": D, "n_params": count_parameters(model),
               "nmse_mean": round(float(np.mean(nmse)), 4), "nmse_per_h": [round(x, 4) for x in nmse],
               "kl_mean": round(float(np.mean(kl)), 4), "kl_per_h": [round(x, 4) for x in kl],
               "top1_mean": round(float(np.mean(top1)), 4), "top1_per_h": [round(x, 4) for x in top1]}
        results.append(rec)
        LOG.info(f"  {name:22s} nmse={rec['nmse_mean']:.3f} kl={rec['kl_mean']:.3f} "
                 f"top1={rec['top1_mean']:.3f} params={rec['n_params']}")

    out = {"layer": args.layer, "m": m, "n": n, "p": p, "n_windows": int(X.shape[0]),
           "d_model": d_out, "results": results}
    path = OUTDIR / f"layer_{args.layer}.json"
    json.dump(out, open(path, "w"), indent=2)
    LOG.info(f"wrote {path}")


if __name__ == "__main__":
    main()

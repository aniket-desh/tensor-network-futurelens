#!/usr/bin/env python
r"""GPT-2 FutureLens completion probes (briefing Phases 3-4, model families B1/B2/B4).

Task: from an observed window of source-layer-l residuals (PCA-whitened, p=64) over
m positions, predict the n future FINAL-layer residuals r^L_{t+1..t+n}; decode them
through the frozen unembedding for token-level metrics (off-by-one: r^L_{t+s} -> x_{t+s+1}).

Compares parameter-matched multi-site linear (B1), MLP (B2), and an MPS readout (B4)
sweeping bond dimension D. One layer per process so the two A40s run different layers.

  python scripts/train_probes.py --layer 6  --device cuda:0
  python scripts/train_probes.py --layer 12 --device cuda:0   # (CUDA_VISIBLE_DEVICES=1)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tn_futurelens.data.activation_cache import load_model
from tn_futurelens.data.windows import WindowSpec, make_windows
from tn_futurelens.models.baselines import MultiSiteLinear, MultiSiteMLP
from tn_futurelens.models.mps import MPSReadout
from tn_futurelens.models.phi import PCAPhi
from tn_futurelens.training.train import train_regression_probe
from tn_futurelens.utils.logging import count_parameters, get_logger
from tn_futurelens.utils.seed import set_seed

LOG = get_logger("probes")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gpt2_probes"


def build_dataset(layer, m, n, p, n_windows, device):
    """Build (X_feat[N,m,p], Y_resid[N,n,768], tok[N,n]) windows; PCA-whiten source."""
    shards = sorted(CACHE.glob("shard_*.pt"))
    meta = torch.load(shards[0], map_location="cpu", weights_only=False)["meta"]
    final = meta["final_layer"]
    # fit PCA on source layer from shard 0
    s0 = torch.load(shards[0], map_location="cpu", weights_only=False)
    r = s0["residuals"][layer][:, 1:, :].reshape(-1, meta["d_model"]).float()
    phi = PCAPhi(meta["d_model"], p).fit(r[torch.randperm(r.shape[0])[:60000]]).to(device)
    spec = WindowSpec(m=m, n=n)

    Xs, Ys, Ts = [], [], []
    got = 0
    for sp in shards:
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        src = sh["residuals"][layer].float().to(device)      # [S, T, d]
        tgt = sh["residuals"][final].float().to(device)      # [S, T, d]
        toks = sh["tokens"]                                  # [S, T]
        feat = phi(src)                                       # [S, T, p]
        S = src.shape[0]
        for i in range(S):
            w = make_windows(feat[i], target=tgt[i], token_ids=toks[i], spec=spec)
            if w["anchors"].numel() == 0:
                continue
            Xs.append(w["observed"].cpu()); Ys.append(w["future"].cpu())
            Ts.append(w["future_token_ids"])
            got += w["anchors"].numel()
        del sh, src, tgt, feat
        if got >= n_windows:
            break
    X = torch.cat(Xs)[:n_windows]
    Y = torch.cat(Ys)[:n_windows]
    Tk = torch.cat(Ts)[:n_windows]
    return X, Y, Tk, meta


@torch.no_grad()
def token_metrics(model, Xva, Yva_raw, Tk, tgt_mean, tgt_std, gpt, device, n_eval=4000):
    """Teacher-KL and top-1 agreement (decode predicted/true r^L via frozen unembed)."""
    idx = torch.randperm(Xva.shape[0])[:n_eval]
    X = Xva[idx].to(device)
    Yt = Yva_raw[idx].to(device)            # raw true future r^L  [B,n,768]
    tok = Tk[idx].to(device)                # realized tokens      [B,n]
    pred_z = model(X)                        # standardized prediction [B,n,768]
    pred = pred_z * tgt_std + tgt_mean
    n = Yt.shape[1]
    kl, top1, ce = [], [], []
    for s in range(n):
        tl = gpt.ln_final(Yt[:, s]) @ gpt.W_U + gpt.b_U      # teacher logits
        sl = gpt.ln_final(pred[:, s]) @ gpt.W_U + gpt.b_U    # student logits
        logp_t = F.log_softmax(tl, -1); logp_s = F.log_softmax(sl, -1)
        kl.append((logp_t.exp() * (logp_t - logp_s)).sum(-1).mean().item())
        top1.append((sl.argmax(-1) == tl.argmax(-1)).float().mean().item())
        ce.append(F.cross_entropy(sl, tok[:, s]).item())
    return {"kl_per_h": [round(x, 4) for x in kl],
            "top1_agree_per_h": [round(x, 4) for x in top1],
            "ce_per_h": [round(x, 4) for x in ce]}


def nmse_per_horizon(model, Xva, Yva_z, device):
    with torch.no_grad():
        pred = model(Xva.to(device))
        Y = Yva_z.to(device)
        num = ((pred - Y) ** 2).sum(-1)                       # [B,n]
        den = ((Y - Y.mean(0, keepdim=True)) ** 2).sum(-1)
        return [(num[:, s].mean() / den[:, s].mean().clamp_min(1e-8)).item()
                for s in range(Y.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=240000)
    ap.add_argument("--D", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()
    set_seed(0)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dev = args.device

    LOG.info(f"layer {args.layer}: building dataset (m={args.m}, n={args.n}, p={args.p})")
    X, Y_raw, Tk, meta = build_dataset(args.layer, args.m, args.n, args.p, args.n_windows, dev)
    d_out = Y_raw.shape[-1]
    LOG.info(f"  {X.shape[0]} windows; X={tuple(X.shape)} Y={tuple(Y_raw.shape)}")

    n_tr = int(0.85 * X.shape[0])
    # standardize targets per-dim from train split
    tgt_mean = Y_raw[:n_tr].mean(0, keepdim=True)
    tgt_std = Y_raw[:n_tr].std(0, keepdim=True).clamp_min(1e-6)
    Y = (Y_raw - tgt_mean) / tgt_std
    Xtr, Ytr, Xva, Yva = X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:]
    Yva_raw, Tkva = Y_raw[n_tr:], Tk[n_tr:]
    tgt_mean_d, tgt_std_d = tgt_mean.to(dev), tgt_std.to(dev)

    LOG.info("  loading GPT-2 for token-level decoding metrics")
    gpt = load_model("gpt2", device=dev)

    def evaluate(name, model):
        nmse = nmse_per_horizon(model, Xva, Yva, dev)
        tm = token_metrics(model, Xva, Yva_raw, Tkva, tgt_mean_d, tgt_std_d, gpt, dev)
        rec = {"name": name, "n_params": count_parameters(model),
               "nmse_per_h": [round(x, 4) for x in nmse],
               "nmse_mean": round(float(np.mean(nmse)), 4), **tm,
               "kl_mean": round(float(np.mean(tm["kl_per_h"])), 4),
               "top1_mean": round(float(np.mean(tm["top1_agree_per_h"])), 4)}
        LOG.info(f"  {name:14s} nmse={rec['nmse_mean']:.3f} kl={rec['kl_mean']:.3f} "
                 f"top1={rec['top1_mean']:.3f} params={rec['n_params']}")
        return rec

    results = []
    p, m, n = args.p, args.m, args.n
    configs = [
        ("multisite_linear", lambda: MultiSiteLinear(p, m, d_out, n)),
        ("mlp_h256", lambda: MultiSiteMLP(p, m, d_out, n, hidden=256, depth=2)),
    ]
    for D in args.D:
        configs.append((f"mps_env_D{D}",
                        lambda D=D: MPSReadout(p=p, D=D, n_sites=m, translation_invariant=False,
                                               readout="env", out_dim=d_out, n_heads=n, seed=0)))
    for name, ctor in configs:
        model = ctor()
        train_regression_probe(model, Xtr, Ytr, Xva, Yva, epochs=args.epochs, lr=1.5e-3,
                               batch_size=4096, device=dev, patience=20, seed=0)
        results.append(evaluate(name, model))

    out = {"layer": args.layer, "m": m, "n": n, "p": p, "n_windows": int(X.shape[0]),
           "d_model": d_out, "results": results}
    path = OUTDIR / f"layer_{args.layer}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    LOG.info(f"wrote {path}")


if __name__ == "__main__":
    main()

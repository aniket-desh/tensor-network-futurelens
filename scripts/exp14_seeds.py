#!/usr/bin/env python
r"""Exp 14 — multi-seed, attention-baseline, held-out-test stress test of the Exp 13 edge.

Exp 13 found the project's only positive: under the KL objective, MPS-D16 edges the best
of three baselines at intermediate horizons (n=8: +0.7%, n=16: +0.5% top-1), single seed.
This script decides whether that edge is real, fixing four loopholes:

  1. multiple seeds (init + batch order),
  2. adds the attention baseline (B3, AttentionPool) Exp 13 omitted,
  3. three-way split: early-stop epoch chosen on a SELECT set, reported on held-out TEST
     (Exp 13 selected the epoch on the same set it reported),
  4. saves per-window per-horizon correctness for paired bootstrap/McNemar statistics.

The select-set metric at the chosen epoch reproduces Exp 13's protocol for comparison.

  python scripts/exp14_seeds.py --seeds 0 1 --device cuda:0 --tag s01
  python scripts/exp14_seeds.py --seeds 2 3 --device cuda:1 --tag s23
"""

from __future__ import annotations

import argparse
import copy
import json
import time
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
from tn_futurelens.utils.logging import count_parameters, get_logger
from tn_futurelens.utils.seed import set_seed

LOG = get_logger("exp14")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
NMAX = 32

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def decode_all(gpt, r, bf16=True):
    """[B,n,d] (or [B,d]) -> logits, stacked matmul; bf16 autocast for speed (A40)."""
    shp = r.shape
    flat = r.reshape(-1, shp[-1])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        out = gpt.ln_final(flat) @ gpt.W_U + gpt.b_U
    return out.reshape(*shp[:-1], -1)


@torch.no_grad()
def top1_correct(probe, X, ttok, *, gpt, mean, std, device, bs=512):
    """Per-window per-horizon bool correctness [N, n] on CPU."""
    probe.eval()
    outs = []
    for i in range(0, X.shape[0], bs):
        pred = probe(X[i:i + bs].to(device).float()) * std + mean
        logits = decode_all(gpt, pred)
        outs.append((logits.argmax(-1) == ttok[i:i + bs].to(device)).cpu())
    return torch.cat(outs)


@torch.no_grad()
def teacher_kl(probe, X, R, *, gpt, mean, std, device, bs=256):
    """Per-horizon mean teacher-KL (nats)."""
    probe.eval()
    n = R.shape[1]
    tot = torch.zeros(n)
    cnt = 0
    for i in range(0, X.shape[0], bs):
        pred = probe(X[i:i + bs].to(device).float()) * std + mean
        rt = R[i:i + bs].to(device).float()
        sl = F.log_softmax(decode_all(gpt, pred).float(), -1)
        tl = F.log_softmax(decode_all(gpt, rt).float(), -1)
        kl = (tl.exp() * (tl - sl)).sum(-1)          # [B, n]
        tot += kl.sum(0).cpu()
        cnt += kl.shape[0]
    return (tot / cnt).tolist()


def train_probe(probe, Xtr, Rtr, Xsel, ttok_sel, *, gpt, mean, std, device,
                epochs, bs, seed, patience=3):
    probe = probe.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1.5e-3)
    gen = torch.Generator().manual_seed(seed)
    ntr = Xtr.shape[0]
    best, best_state, since, ep_ran = -1.0, copy.deepcopy(probe.state_dict()), 0, 0
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(ntr, generator=gen)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = probe(Xtr[idx].to(device).float()) * std + mean      # [B,n,768]
            rt = Rtr[idx].to(device).float()
            sl = F.log_softmax(decode_all(gpt, pred).float(), -1)
            tl = F.softmax(decode_all(gpt, rt).float(), -1)
            loss = (tl * (tl.clamp_min(1e-12).log() - sl)).sum(-1).mean()
            loss.backward()
            opt.step()
        ep_ran = ep + 1
        sel_score = top1_correct(probe, Xsel, ttok_sel, gpt=gpt, mean=mean, std=std,
                                 device=device).float().mean().item()
        if sel_score > best:
            best, best_state, since = sel_score, copy.deepcopy(probe.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                break
    probe.load_state_dict(best_state)
    return probe, best, ep_ran


class SitePermute(torch.nn.Module):
    """Apply a FIXED site permutation before the head — destroys 1D chain order.

    With per-site cores the MPS cannot relabel the order of matrix multiplication,
    so if its edge relies on transfer-matrix propagation along the token chain,
    shuffling must hurt; if the MPS is merely an order-insensitive multilinear
    feature map, it won't.
    """

    def __init__(self, head, m):
        super().__init__()
        self.head = head
        gen = torch.Generator().manual_seed(123)        # fixed across seeds/models
        self.register_buffer("perm", torch.randperm(m, generator=gen))

    def forward(self, v):
        return self.head(v[:, self.perm])


def build_probe(name, *, seed, d_model, p, m, d_out, n, pca):
    set_seed(seed)
    lp = LearnedLinearPhi(d_model, p).init_from_pca(pca)
    base = name.replace("_shuf", "").replace("_noconst", "")
    if base == "mlp":
        head = MultiSiteMLP(p, m, d_out, n, hidden=256, depth=2)
    elif base == "conv1d":
        head = Conv1DProbe(p, m, d_out, n, hidden=128, layers=2)
    elif base == "bilinear":
        head = BilinearProbe(p, m, d_out, n, rank=64)
    elif base == "attention":
        head = AttentionPool(p, m, d_out, n, d_model=256, n_heads=4)
    elif base.startswith("mps_D"):
        D = int(base.split("D")[1])
        head = MPSReadout(p=p, D=D, n_sites=m, readout="env", out_dim=d_out,
                          n_heads=n, const_channel="_noconst" not in name, seed=seed)
    else:
        raise ValueError(name)
    if "_shuf" in name:
        head = SitePermute(head, m)
    return PhiHead(lp, head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--models", nargs="+",
                    default=["mlp", "conv1d", "bilinear", "attention", "mps_D16"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-select", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=160)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--outdir", default="gpt2_exp14_seeds")
    ap.add_argument("--prep", default=None,
                    help="path to exp14_prep.py output; default derived from layer/m")
    args = ap.parse_args()
    OUTDIR = ROOT / "results" / "runs" / args.outdir
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dev, m, p = args.device, args.m, args.p

    gpt = load_model("gpt2", device=dev)
    prep_path = Path(args.prep) if args.prep else CACHE / f"exp14_prep_L{args.layer}_m{m}.pt"
    prep = torch.load(prep_path, map_location="cpu", weights_only=False)
    assert prep["layer"] == args.layer and prep["m"] == m and prep["p"] == p
    X, Rfull, ttok = prep["X"], prep["Rfull"], prep["ttok"]        # X/Rfull fp16
    mean32, std32 = prep["mean"], prep["std"]
    d_model = Rfull.shape[-1]
    N = X.shape[0]
    ntr, nsel = prep["split"][0], prep["split"][1]
    # stride-1 windows: each 256-token sequence yields exactly 216 windows (m=8, n=32,
    # +realized token) -> cluster id = index // 216 for sequence-level bootstrap
    LOG.info(f"{N} windows: train {ntr}, select {nsel}, test {N - ntr - nsel} "
             f"(~{(N - ntr - nsel) // 216} independent sequences in test)")

    pca = PCAPhi(d_model, p)
    pca.load_state_dict(prep["pca"])
    pca = pca.to(dev)

    Xtr, Xsel, Xte = X[:ntr], X[ntr:ntr + nsel], X[ntr + nsel:]
    res_path = OUTDIR / f"results_{args.tag}.json"
    out = {"layer": args.layer, "m": m, "p": p, "n_windows": N,
           "split": prep["split"], "runs": []}

    for n in args.horizons:
        R = Rfull[:, :n]                                   # fp16 view; .float() per batch
        mean, std = mean32[:, :n].to(dev), std32[:, :n].to(dev)
        Rtr = R[:ntr]
        ttok_sel, ttok_te = ttok[ntr:ntr + nsel, :n], ttok[ntr + nsel:, :n]
        for seed in args.seeds:
            corr_blob = {}
            for name in args.models:
                t0 = time.time()
                probe = build_probe(name, seed=seed, d_model=d_model, p=p, m=m,
                                    d_out=d_model, n=n, pca=pca)
                n_par = count_parameters(probe)
                probe, sel_best, ep_ran = train_probe(
                    probe, Xtr, Rtr, Xsel, ttok_sel, gpt=gpt, mean=mean, std=std,
                    device=dev, epochs=args.epochs, bs=args.bs, seed=seed)
                corr = top1_correct(probe, Xte, ttok_te, gpt=gpt, mean=mean, std=std, device=dev)
                kl = teacher_kl(probe, Xte, R[ntr + nsel:], gpt=gpt, mean=mean, std=std, device=dev)
                corr_blob[name] = corr
                per_h = corr.float().mean(0).tolist()
                rec = {"n": n, "seed": seed, "model": name, "n_params": n_par,
                       "sel_top1_best": round(sel_best, 4),
                       "test_top1_mean": round(corr.float().mean().item(), 4),
                       "test_top1_per_h": [round(x, 4) for x in per_h],
                       "test_kl_per_h": [round(x, 4) for x in kl],
                       "epochs_ran": ep_ran, "secs": round(time.time() - t0, 1)}
                out["runs"].append(rec)
                json.dump(out, open(res_path, "w"), indent=2)
                LOG.info(f"n={n} seed={seed} {name:9s} test_top1={rec['test_top1_mean']:.4f} "
                         f"sel={sel_best:.4f} params={n_par:,} ep={ep_ran} {rec['secs']}s")
            torch.save(corr_blob, OUTDIR / f"correct_n{n}_seed{seed}.pt")
    LOG.info(f"wrote {res_path}")


if __name__ == "__main__":
    main()

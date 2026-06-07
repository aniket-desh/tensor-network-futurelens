#!/usr/bin/env python
r"""Exp 10 — Born-machine MPS: discrete conditional completion (briefing B6, next-step #2).

The most theory-faithful test: quantize PCA-whitened residuals into codebook symbols
z = Q(phi(r)), train a Born MPS P(z)=|Psi(z)|^2/Z over chains, then CONDITION on the
observed symbols and predict the next symbol -- "clamp observed sites, complete the
future" literally. Compare the Born conditional to unigram, bigram, and a discriminative
MLP classifier on the same quantized task.

  python scripts/exp10_born.py --layer 6 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tn_futurelens.data.activation_cache import iter_shards
from tn_futurelens.models.born_mps import BornMPS
from tn_futurelens.models.phi import PCAPhi
from tn_futurelens.utils.logging import get_logger
from tn_futurelens.utils.seed import set_seed

LOG = get_logger("exp10")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gpt2_born"


def kmeans(x, K, iters=25, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:K]].clone()
    for _ in range(iters):
        d = torch.cdist(x, c)
        a = d.argmin(1)
        for k in range(K):
            sel = a == k
            if sel.any():
                c[k] = x[sel].mean(0)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--n-chains", type=int, default=120000)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    set_seed(0); OUTDIR.mkdir(parents=True, exist_ok=True)
    dev, p, K, m, n = args.device, args.p, args.K, args.m, args.n
    N = m + n

    # 1) PCA-whiten + k-means codebook
    s0 = next(iter_shards(CACHE)); d_model = s0["meta"]["d_model"]
    r0 = s0["residuals"][args.layer][:, 1:, :].reshape(-1, d_model).float()
    pca = PCAPhi(d_model, p).fit(r0[torch.randperm(r0.shape[0])[:60000]]).to(dev)
    sample = pca(r0[torch.randperm(r0.shape[0])[:60000]].to(dev))
    cents = kmeans(sample, K, seed=0)
    LOG.info(f"k-means codebook K={K} fit")

    # 2) quantize all positions -> symbols, build length-N chains
    chains = []
    for sh in iter_shards(CACHE):
        feat = pca(sh["residuals"][args.layer][:, 1:, :].float().to(dev))  # [S,T,p]
        S, T, _ = feat.shape
        z = torch.cdist(feat.reshape(-1, p), cents).argmin(1).reshape(S, T)  # [S,T] symbols
        for start in range(0, T - N + 1, N):                                  # non-overlapping chains
            chains.append(z[:, start:start + N].cpu())
        if sum(c.shape[0] for c in chains) >= args.n_chains:
            break
    Z = torch.cat(chains)[:args.n_chains].to(dev)
    ntr = int(0.85 * Z.shape[0]); Ztr, Zte = Z[:ntr], Z[ntr:]
    LOG.info(f"{Z.shape[0]} chains of length {N}; symbols in [0,{K})")

    # 3) train Born MPS by NLL
    born = BornMPS(K, args.D, N, seed=0).to(dev)
    opt = torch.optim.Adam(born.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    best, best_nll = None, float("inf")
    import copy
    for ep in range(args.epochs):
        perm = torch.randperm(ntr, generator=gen)
        for i in range(0, ntr, 1024):
            opt.zero_grad(); loss = born.nll(Ztr[perm[i:i + 1024]]); loss.backward(); opt.step()
        with torch.no_grad():
            v = born.nll(Zte[:4096]).item()
        if v < best_nll:
            best_nll, best = v, copy.deepcopy(born.state_dict())
    born.load_state_dict(best)
    LOG.info(f"Born MPS test NLL={best_nll:.3f} (uniform={N*np.log(K):.3f})")

    # 4) conditional next-symbol prediction at site m, given z_{0:m}
    obs, truth = Zte[:, :m], Zte[:, m]
    lp = born.conditional_next_logprob(obs)
    born_acc = (lp.argmax(-1) == truth).float().mean().item()
    born_nll_next = F.nll_loss(lp, truth).item()

    # baselines (on the quantized task)
    uni = torch.bincount(Ztr.reshape(-1), minlength=K).argmax()
    uni_acc = (truth == uni).float().mean().item()
    # bigram P(z_m | z_{m-1})
    big = torch.zeros(K, K, device=dev)
    for a, b in zip(Ztr[:, m - 1], Ztr[:, m]):
        big[a, b] += 1
    big_pred = big.argmax(1)
    big_acc = (big_pred[obs[:, -1]] == truth).float().mean().item()
    # discriminative MLP on observed symbols (embed) -> predict next symbol
    emb = torch.nn.Embedding(K, 32).to(dev)
    mlp = torch.nn.Sequential(torch.nn.Linear(m * 32, 256), torch.nn.GELU(), torch.nn.Linear(256, K)).to(dev)
    params = list(emb.parameters()) + list(mlp.parameters())
    o2 = torch.optim.Adam(params, lr=2e-3)
    for ep in range(40):
        perm = torch.randperm(ntr, generator=gen)
        for i in range(0, ntr, 2048):
            idx = perm[i:i + 2048]
            o2.zero_grad()
            logits = mlp(emb(Ztr[idx, :m]).reshape(idx.shape[0], -1))
            F.cross_entropy(logits, Ztr[idx, m]).backward(); o2.step()
    with torch.no_grad():
        mlp_pred = mlp(emb(obs).reshape(obs.shape[0], -1)).argmax(-1)
    mlp_acc = (mlp_pred == truth).float().mean().item()

    out = {"layer": args.layer, "K": K, "D": args.D, "N": N,
           "born_test_nll": round(best_nll, 3), "uniform_nll": round(float(N * np.log(K)), 3),
           "next_symbol_acc": {"born": round(born_acc, 4), "mlp": round(mlp_acc, 4),
                               "bigram": round(big_acc, 4), "unigram": round(uni_acc, 4)},
           "born_next_nll": round(born_nll_next, 4)}
    LOG.info(f"  next-symbol acc: born={born_acc:.3f} mlp={mlp_acc:.3f} bigram={big_acc:.3f} unigram={uni_acc:.3f}")
    json.dump(out, open(OUTDIR / f"layer_{args.layer}.json", "w"), indent=2)
    LOG.info(f"wrote layer_{args.layer}.json")


if __name__ == "__main__":
    main()

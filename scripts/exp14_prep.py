#!/usr/bin/env python
r"""Exp 14 prep — build the completion dataset ONCE and save it (fp16) to disk.

The container has a ~93 GiB cgroup memory cap; building the 100k-window dataset in
every run process (~28 GB transient peak each) OOM-killed concurrent jobs twice.
This script pays the build cost once and saves:
  X fp16 [N,m,768], Rfull fp16 [N,32,768], ttok int64 [N,32] (teacher tokens, fp32
  decode), train-split target mean/std [1,32,768], PCA(p=64) state fit on train.

  python scripts/exp14_prep.py --layer 6 --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tn_futurelens.data.activation_cache import load_model
from tn_futurelens.models.phi import PCAPhi
from tn_futurelens.training.eval import build_completion_dataset
from tn_futurelens.utils.logging import get_logger

LOG = get_logger("exp14prep")
ROOT = Path(__file__).resolve().parents[1]
NMAX = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-select", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=50000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    CACHE = ROOT / "data" / "cache" / args.model / "wikitext103"
    out = Path(args.out) if args.out else CACHE / f"exp14_prep_L{args.layer}_m{args.m}.pt"
    dev = args.device

    n_windows = args.n_train + args.n_select + args.n_test
    X, Rfull, _, meta = build_completion_dataset(CACHE, args.layer, args.m, NMAX, n_windows)
    N, d_model = X.shape[0], X.shape[-1]
    ntr = args.n_train
    LOG.info(f"built {N} windows (d={d_model})")

    gpt = load_model(args.model, device=dev)
    with torch.no_grad():
        ttok = torch.empty(N, NMAX, dtype=torch.long)
        for i in range(0, N, 512):
            r = Rfull[i:i + 512].to(dev).float()
            lg = gpt.ln_final(r) @ gpt.W_U + gpt.b_U
            ttok[i:i + 512] = lg.argmax(-1).cpu()
    LOG.info("teacher tokens done (fp32 decode)")

    mean = Rfull[:ntr].mean(0, keepdim=True)                      # [1,32,768]
    std = Rfull[:ntr].std(0, keepdim=True).clamp_min(1e-6)

    flat_tr = X[:ntr].reshape(-1, d_model)
    gen = torch.Generator().manual_seed(0)
    pca = PCAPhi(d_model, args.p).fit(flat_tr[torch.randperm(flat_tr.shape[0], generator=gen)[:60000]])
    LOG.info("PCA fit on train windows")

    torch.save({"X": X.half(), "Rfull": Rfull.half(), "ttok": ttok,
                "mean": mean, "std": std, "pca": pca.state_dict(),
                "layer": args.layer, "m": args.m, "p": args.p,
                "split": [args.n_train, args.n_select, N - args.n_train - args.n_select]},
               out)
    LOG.info(f"wrote {out} ({out.stat().st_size/1e9:.1f} GB)")


if __name__ == "__main__":
    main()

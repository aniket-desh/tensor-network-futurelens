#!/usr/bin/env python
r"""Cache GPT-2 residual streams on WikiText-103 (briefing Phase 1).

Two-step, shardable for multi-GPU:
  1. build the token sequences once (CPU):   --build-only
  2. cache a contiguous range on one GPU:     --device cuda:0 --start 0 --end 3000
Run two copies over disjoint ranges on cuda:0 / cuda:1 to use both A40s.

Example (driven by scripts/cache_gpt2_2gpu.sh):
  python scripts/cache_residuals.py --build-only --num-sequences 6000
  python scripts/cache_residuals.py --device cuda:0 --start 0    --end 3000
  python scripts/cache_residuals.py --device cuda:1 --start 3000 --end 6000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tn_futurelens.data.activation_cache import (
    cache_residuals,
    load_model,
    logit_lens_check,
    save_shard,
)
from tn_futurelens.data.token_datasets import build_token_sequences
from tn_futurelens.utils.config import git_commit_hash, utc_timestamp
from tn_futurelens.utils.logging import get_logger

LOG = get_logger("cache")
ROOT = Path(__file__).resolve().parents[1]


def default_out_dir(model: str) -> Path:
    return ROOT / "data" / "cache" / model / "wikitext103"


def build_sequences(args, out_dir: Path) -> None:
    from transformers import AutoTokenizer

    LOG.info(f"building {args.num_sequences} sequences (seq_len={args.seq_len})")
    tok = AutoTokenizer.from_pretrained(args.model)
    seqs = build_token_sequences(tok, seq_len=args.seq_len,
                                 num_sequences=args.num_sequences, split="train")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": seqs.tokens, "article_ids": seqs.article_ids,
                "seq_len": seqs.seq_len, "bos_token_id": seqs.bos_token_id,
                "model": args.model, "dataset": "wikitext-103-raw-v1",
                "git_commit": git_commit_hash(), "timestamp": utc_timestamp()},
               out_dir / "tokens.pt")
    LOG.info(f"saved {out_dir/'tokens.pt'}  tokens shape={tuple(seqs.tokens.shape)} "
             f"({seqs.tokens.numel():,} token positions)")


def cache_range(args, out_dir: Path) -> None:
    data = torch.load(out_dir / "tokens.pt", weights_only=False)
    tokens = data["tokens"]
    article_ids = data["article_ids"]
    S = tokens.shape[0]
    start, end = args.start, min(args.end if args.end > 0 else S, S)
    LOG.info(f"caching sequences [{start}:{end}] of {S} on {args.device}")

    model = load_model(args.model, device=args.device)
    n_layers = model.cfg.n_layers
    layers = [l for l in args.layers if l <= n_layers]

    if start == 0:
        diff = logit_lens_check(model, tokens[:8])
        LOG.info(f"logit-lens sanity: max|manual-model logits| = {diff:.2e} "
                 f"({'OK' if diff < 1e-2 else 'WARN: LN folding?'})")

    meta_base = {"model": args.model, "n_layers": n_layers, "d_model": model.cfg.d_model,
                 "seq_len": data["seq_len"], "layers": layers, "final_layer": n_layers,
                 "store_dtype": "float16", "git_commit": git_commit_hash(),
                 "timestamp": utc_timestamp(), "dataset": "wikitext-103-raw-v1"}

    sh = args.shard_size
    for s0 in range(start, end, sh):
        s1 = min(s0 + sh, end)
        tok_chunk = tokens[s0:s1]
        res = cache_residuals(model, tok_chunk, layers, batch_size=args.batch_size)
        shard_idx = s0 // sh
        path = save_shard(out_dir, shard_idx, res, tok_chunk, article_ids[s0:s1],
                          {**meta_base, "seq_range": [s0, s1]})
        LOG.info(f"  wrote {path.name}  seqs[{s0}:{s1}]  layers={sorted(res)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--num-sequences", type=int, default=6000)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 2, 4, 6, 8, 10, 12])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args.model)
    if args.build_only or not (out_dir / "tokens.pt").exists():
        build_sequences(args, out_dir)
        if args.build_only:
            return
    cache_range(args, out_dir)


if __name__ == "__main__":
    main()

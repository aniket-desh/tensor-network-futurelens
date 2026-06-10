#!/usr/bin/env python
r"""Exp 15 — block coarse-graining (TASK Experiment D, part 1: structure).

Question: token-level residual correlations are finite-ξ but HIGH-RANK (Exp 06: ~27-48
modes at p=64 — too many for a small-D MPS to be uniquely efficient). Does an RG-style
block-spin step v̄_I = mean(v_{Ib..Ib+b-1}) lower the effective mode count and create an
MPS-friendly regime (few modes, short block-ξ)?

Method: PCA p=64 at the token level (affine, so block-mean of PCA = PCA of block-mean),
block-average with b ∈ {1,2,4,8}, then whitened two-point function + Ho-Kalman over the
block chain. Report effective modes (Hankel SV > 5% of max), implied D, persistent-mode
count, and bulk ξ in block and token units.

  python scripts/exp15_block.py --layers 6 8 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tn_futurelens.analysis.correlations import CorrelationAccumulator
from tn_futurelens.analysis.realization import ho_kalman
from tn_futurelens.models.phi import PCAPhi
from tn_futurelens.utils.logging import get_logger

LOG = get_logger("exp15")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gpt2_exp15_block"
PERSIST = 0.9


def block_mean(V: torch.Tensor, b: int) -> torch.Tensor:
    """[S, T, p] -> [S, T//b, p] non-overlapping block means."""
    S, T, p = V.shape
    Tb = (T // b) * b
    return V[:, :Tb].reshape(S, T // b, b, p).mean(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-delta", type=int, default=40)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(CACHE.glob("shard_*.pt"))
    rows = []
    for layer in args.layers:
        # fit token-level PCA on shard 0 (drop BOS position, as in Exp 06)
        s0 = torch.load(shards[0], map_location="cpu", weights_only=False)
        d_model = s0["meta"]["d_model"]
        r0 = s0["residuals"][layer][:, 1:, :].reshape(-1, d_model).float()
        gen = torch.Generator().manual_seed(0)
        pca = PCAPhi(d_model, args.p).fit(r0[torch.randperm(r0.shape[0], generator=gen)[:60000]])
        del s0, r0
        for b in args.blocks:
            md = max(8, args.max_delta // b)        # need enough lags, >=8 for Hankel
            acc = CorrelationAccumulator(args.p, md, device=args.device, dtype=torch.float32)
            for sp in shards:
                sh = torch.load(sp, map_location="cpu", weights_only=False)
                V = pca(sh["residuals"][layer][:, 1:, :].float())  # [S, T-1, p]
                acc.update(block_mean(V, b).to(args.device))
                del sh, V
            res = acc.finalize(whiten=True)
            r = ho_kalman(res.Chat.cpu().numpy(), rel_threshold=0.03, max_rank=30)
            sv = r.singular_values / r.singular_values[0]
            eff = int(np.sum(sv > 0.05))
            mags = np.abs(r.eigenvalues)
            bulk_xi = np.sort(r.xis[mags < PERSIST])[::-1]
            row = {"layer": layer, "b": b, "eff_modes_sv": eff,
                   "implied_D": int(np.ceil(np.sqrt(eff + 1))),
                   "n_persistent": int(np.sum(mags > PERSIST)),
                   "bulk_xi_block": [round(float(x), 2) for x in bulk_xi[:4]],
                   "bulk_xi_tokens": [round(float(x * b), 2) for x in bulk_xi[:4]],
                   "hankel_sv_top16": [round(float(x), 4) for x in sv[:16]]}
            rows.append(row)
            LOG.info(f"L{layer} b={b}: eff_modes={eff} impliedD={row['implied_D']} "
                     f"persistent={row['n_persistent']} xi_block={row['bulk_xi_block'][:2]} "
                     f"xi_tokens={row['bulk_xi_tokens'][:2]}")
    json.dump({"rows": rows}, open(OUTDIR / "modes_vs_block.json", "w"), indent=2)
    LOG.info(f"wrote {OUTDIR/'modes_vs_block.json'}")


if __name__ == "__main__":
    main()

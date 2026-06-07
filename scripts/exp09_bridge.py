#!/usr/bin/env python
r"""Exp 09 — bridge: correlations in the LEARNED-phi space (next-step, "most interesting").

Exp 06 measured correlations in the fixed PCA basis and found high-rank, many-mode
structure. But the predictive wins (Exp 03) came from a LEARNED phi. So: after training a
task-relevant learned phi, re-measure the residual correlation structure IN THAT phi
space. Does the learned phi select a lower-rank / simpler correlation structure (a space
where the MPS transfer story would hold), or the same many-mode structure?

We learn phi via a learned-phi + LINEAR completion probe (so phi = the predictive linear
basis, no nonlinearity confound), extract W_phi, then realize the correlation spectrum
(Ho-Kalman) in the whitened learned-phi space and compare to the PCA-space result (Exp 06).

  python scripts/exp09_bridge.py --layer 6 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tn_futurelens.analysis.correlations import CorrelationAccumulator
from tn_futurelens.analysis.realization import ho_kalman
from tn_futurelens.data.activation_cache import iter_shards
from tn_futurelens.models.baselines import MultiSiteLinear
from tn_futurelens.models.phi import LearnedLinearPhi, PCAPhi
from tn_futurelens.models.probes import PhiHead
from tn_futurelens.training.eval import build_completion_dataset, standardize_targets
from tn_futurelens.training.train import train_regression_probe
from tn_futurelens.utils.logging import get_logger
from tn_futurelens.utils.seed import set_seed

LOG = get_logger("exp09")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gpt2_exp09"
EXP06 = ROOT / "results" / "runs" / "gpt2_exp06" / "gpt2.json"
MAX_DELTA = 40
PERSIST = 0.9


def realize_in_phi(layer, phi, device, p):
    """Accumulate whitened C(Delta) of phi(residual) over all shards, then Ho-Kalman."""
    acc = CorrelationAccumulator(p, MAX_DELTA, device=device, dtype=torch.float32)
    phi = phi.to(device)
    for shard in iter_shards(CACHE):
        r = shard["residuals"][layer][:, 1:, :].float().to(device)
        acc.update(phi(r))
    res = acc.finalize(whiten=True)
    r = ho_kalman(res.Chat.cpu().numpy(), rel_threshold=0.03, max_rank=30)
    sv = r.singular_values / r.singular_values[0]
    eff = int(np.sum(sv > 0.05))
    mags = np.abs(r.eigenvalues)
    bulk = np.sort(r.xis[mags < PERSIST])[::-1]
    return {"eff_modes_sv": eff, "n_persistent": int(np.sum(mags > PERSIST)),
            "bulk_xi_top": [round(float(x), 2) for x in bulk[:5]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=150000)
    args = ap.parse_args()
    set_seed(0); OUTDIR.mkdir(parents=True, exist_ok=True)
    dev, p, m, n = args.device, args.p, 8, 4

    # 1) learn a task-relevant phi via a learned-phi + linear completion probe
    X, Y_raw, _, meta = build_completion_dataset(CACHE, args.layer, m, n, args.n_windows)
    flat = X.reshape(-1, X.shape[-1])
    pca = PCAPhi(X.shape[-1], p).fit(flat[torch.randperm(flat.shape[0])[:60000]]).to(dev)
    ntr = int(0.85 * X.shape[0])
    Yz, _, _ = standardize_targets(Y_raw, ntr)
    learned_phi = LearnedLinearPhi(meta["d_model"], p).init_from_pca(pca).to(dev)
    probe = PhiHead(learned_phi, MultiSiteLinear(p, m, Y_raw.shape[-1], n))
    train_regression_probe(probe, X[:ntr], Yz[:ntr], X[ntr:], Yz[ntr:], epochs=90, lr=1.5e-3,
                           batch_size=4096, device=dev, patience=15, seed=0)
    LOG.info("  trained learned-phi linear probe; analyzing correlation structure")

    # 2) realize correlation spectrum in PCA space vs learned-phi space (same layer)
    pca_space = realize_in_phi(args.layer, pca, dev, p)
    phi_space = realize_in_phi(args.layer, probe.phi, dev, p)
    exp06 = {r["layer"]: r for r in json.load(open(EXP06))["rows"]}[args.layer]

    out = {"layer": args.layer,
           "pca_space": pca_space, "learned_phi_space": phi_space,
           "exp06_pca_eff_modes": exp06["eff_modes_sv"]}
    LOG.info(f"  PCA space: eff_modes={pca_space['eff_modes_sv']} persist={pca_space['n_persistent']} "
             f"bulk_xi={pca_space['bulk_xi_top'][:3]}")
    LOG.info(f"  learned-phi space: eff_modes={phi_space['eff_modes_sv']} persist={phi_space['n_persistent']} "
             f"bulk_xi={phi_space['bulk_xi_top'][:3]}")
    json.dump(out, open(OUTDIR / f"layer_{args.layer}.json", "w"), indent=2)
    LOG.info(f"wrote layer_{args.layer}.json")


if __name__ == "__main__":
    main()

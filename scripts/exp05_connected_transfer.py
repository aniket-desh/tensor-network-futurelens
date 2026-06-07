#!/usr/bin/env python
r"""Exp 05 — connected-only test (#4) + TI transfer-spectrum check (#5).

(a) connected-only: GPT-2 residual correlation = persistent (long-range) subspace +
    finite-xi bulk (Exp 01). The MPS advantage is supposed to live in the *connected*
    (finite-xi) modes. We project out the persistent subspace and ask whether the
    MPS's edge over the MLP *grows* on the connected-only task.

(b) TI transfer spectrum: train a translation-invariant MPS (+const +learned phi) and
    compare its transfer-matrix correlation lengths xi_mu to the empirical bulk xi
    from Exp 01 -- the "is it actually using the transfer mechanism?" check (Phase 6).

  python scripts/exp05_connected_transfer.py --mode connected --layer 12 --device cuda:0
  python scripts/exp05_connected_transfer.py --mode transfer --layer 6  --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tn_futurelens.analysis.correlations import two_point_function
from tn_futurelens.analysis.transfer_modes import mps_transfer_report
from tn_futurelens.data.windows import WindowSpec, make_windows
from tn_futurelens.models.baselines import MultiSiteMLP
from tn_futurelens.models.mps import MPSReadout
from tn_futurelens.models.phi import LearnedLinearPhi, PCAPhi
from tn_futurelens.models.probes import PhiHead
from tn_futurelens.training.eval import standardize_targets
from tn_futurelens.training.train import train_regression_probe
from tn_futurelens.utils.logging import get_logger
from tn_futurelens.utils.seed import set_seed

LOG = get_logger("exp05")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gpt2_exp05"


def load_layer_seqs(layer, device):
    """Return list of per-sequence residual tensors [T,d] (BOS dropped) + d_model."""
    seqs = []
    for sp in sorted(CACHE.glob("shard_*.pt")):
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        r = sh["residuals"][layer][:, 1:, :].float()  # drop BOS
        for i in range(r.shape[0]):
            seqs.append(r[i])
        del sh
    return seqs


# ---------------- (a) connected-only ----------------
def connected_experiment(layer, device, m=8, n=4, p=64, k_persist=None, n_windows=120000,
                         epochs=110):
    set_seed(0)
    seqs = load_layer_seqs(layer, device)
    d_model = seqs[0].shape[1]
    # fit PCA-whitening on a sample
    sample = torch.cat([s for s in seqs[:400]], 0)
    pca = PCAPhi(d_model, p).fit(sample[torch.randperm(sample.shape[0])[:60000]])

    # estimate persistent projector from whitened Chat(delta=32)
    wsample = torch.stack([pca(seqs[i]) for i in range(min(600, len(seqs)))])  # [S,T,p]
    res = two_point_function(wsample, max_delta=33, whiten=True)
    svals = torch.linalg.svdvals(res.Chat[32])
    if k_persist is None:
        k_persist = int((svals > 0.5).sum().item())
    U, S, Vh = torch.linalg.svd(res.Chat[32])
    Vk = U[:, :k_persist]                      # [p, k] persistent directions
    P = Vk @ Vk.T                              # projector onto persistent subspace
    LOG.info(f"layer {layer}: k_persist={k_persist} (svals>0.5); building windows")

    def make_set(project_out):
        Xs, Ys = [], []
        got = 0
        spec = WindowSpec(m=m, n=n)
        for s in seqs:
            v = pca(s)                          # [T, p] whitened
            if project_out:
                v = v - v @ P.T                 # remove persistent subspace
            w = make_windows(v, spec=spec)
            if w["anchors"].numel() == 0:
                continue
            Xs.append(w["observed"]); Ys.append(w["future"])
            got += w["anchors"].numel()
            if got >= n_windows:
                break
        return torch.cat(Xs)[:n_windows], torch.cat(Ys)[:n_windows]

    out = {"layer": layer, "k_persist": k_persist, "conditions": {}}
    for tag, proj in [("full", False), ("connected_only", True)]:
        X, Y_raw = make_set(proj)
        ntr = int(0.85 * X.shape[0])
        Y, _, _ = standardize_targets(Y_raw, ntr)
        Xtr, Ytr, Xva, Yva = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
        row = {}
        for name, model in [("mlp", MultiSiteMLP(p, m, p, n, hidden=256, depth=2)),
                            ("mps_const", MPSReadout(p=p, D=16, n_sites=m, readout="env",
                                                     out_dim=p, n_heads=n, const_channel=True, seed=0))]:
            r = train_regression_probe(model, Xtr, Ytr, Xva, Yva, epochs=epochs, lr=1.5e-3,
                                       batch_size=4096, device=device, patience=20, seed=0)
            row[name] = round(r.best_val_nmse, 4)
        row["mps_minus_mlp"] = round(row["mps_const"] - row["mlp"], 4)
        out["conditions"][tag] = row
        LOG.info(f"  {tag:14s} mlp={row['mlp']:.3f} mps={row['mps_const']:.3f} "
                 f"(mps-mlp={row['mps_minus_mlp']:+.3f})")
    return out


# ---------------- (b) TI transfer spectrum ----------------
def transfer_experiment(layer, device, m=8, n=4, p=64, n_windows=120000, epochs=140):
    set_seed(0)
    from tn_futurelens.training.eval import build_completion_dataset
    X, Y_raw, Tk, meta = build_completion_dataset(CACHE, layer, m, n, n_windows)
    d_out = Y_raw.shape[-1]
    flat = X.reshape(-1, X.shape[-1])
    pca = PCAPhi(X.shape[-1], p).fit(flat[torch.randperm(flat.shape[0])[:60000]]).to(device)
    ntr = int(0.85 * X.shape[0])
    Y, _, _ = standardize_targets(Y_raw, ntr)
    phi = LearnedLinearPhi(meta["d_model"], p).init_from_pca(pca).to(device)
    ti = MPSReadout(p=p, D=12, n_sites=m, translation_invariant=True, readout="env",
                    out_dim=d_out, n_heads=n, const_channel=True, seed=0)
    model = PhiHead(phi, ti)
    r = train_regression_probe(model, X[:ntr], Y[:ntr], X[ntr:], Y[ntr:], epochs=epochs,
                               lr=1.5e-3, batch_size=4096, device=device, patience=25, seed=0)
    rep = mps_transfer_report(ti.to("cpu"))
    xis = np.sort(rep.correlation_lengths[np.isfinite(rep.correlation_lengths)])[::-1]
    LOG.info(f"layer {layer}: TI MPS val_nmse={r.best_val_nmse:.3f} "
             f"leading|lambda|={rep.magnitudes[0]:.3f} gap={rep.spectral_gap:.3f}")
    LOG.info(f"  learned transfer xi (top 6): {[round(float(x),2) for x in xis[:6]]}")
    return {"layer": layer, "val_nmse": round(r.best_val_nmse, 4),
            "leading_mag": float(rep.magnitudes[0]), "spectral_gap": float(rep.spectral_gap),
            "learned_xi_top": [round(float(x), 3) for x in xis[:8]],
            "n_modes_xi_gt_1": int(np.sum(xis > 1.0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["connected", "transfer"], required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "connected":
        out = connected_experiment(args.layer, args.device)
        json.dump(out, open(OUTDIR / f"connected_layer_{args.layer}.json", "w"), indent=2)
    else:
        out = transfer_experiment(args.layer, args.device)
        json.dump(out, open(OUTDIR / f"transfer_layer_{args.layer}.json", "w"), indent=2)
    LOG.info(f"wrote results for mode={args.mode} layer={args.layer}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
r"""Exp 06 — mechanistic test: realize the residual correlation spectrum (fixed PCA basis).

The clean, confound-free version of "does an MPS transfer matrix represent the measured
residual correlations?" We do NOT train a predictor (an autoregressive MPS with a fixed
linear head cannot even fit AR(1) -- see note in summary). Instead we realize the
empirical matrix correlation sequence ``C(Delta)`` directly via Ho-Kalman / ERA:
  * block-Hankel singular spectrum -> number of modes (state dim); MPS needs D^2-1 >= rank
  * realized state-matrix eigenvalues = the correlation decay modes; xi = -1/ln|lambda|
all in the SAME fixed PCA-whitened basis the correlations were measured in.

  --mode synth          : validate Ho-Kalman recovers planted xi from *measured* (noisy) C
  --mode gpt2           : GPT-2 all layers, fixed PCA p=64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tn_futurelens.analysis.correlations import CorrelationAccumulator, two_point_function
from tn_futurelens.analysis.realization import ho_kalman
from tn_futurelens.data.synthetic import ar1_process, multi_mode_ar_process
from tn_futurelens.models.phi import PCAPhi
from tn_futurelens.utils.logging import get_logger
from tn_futurelens.utils.plotting import save_fig, set_style
from tn_futurelens.utils.seed import set_seed

import matplotlib.pyplot as plt

LOG = get_logger("exp06")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gpt2_exp06"
FIGDIR = ROOT / "docs" / "06_mechanistic_realization" / "figures"
CORR = ROOT / "results" / "runs" / "gpt2_correlations" / "results.json"
MAX_DELTA = 40
PERSIST = 0.9   # |lambda| above this counts as a persistent (long-range) mode


def run_synth():
    LOG.info("Exp 06 synthetic validation: Ho-Kalman on MEASURED correlations")
    set_seed(0)
    out = {}
    V = ar1_process(600, 400, p=6, rho=0.85, seed=0)
    res = two_point_function(V, max_delta=MAX_DELTA, whiten=True)
    r = ho_kalman(res.Chat.cpu().numpy(), rel_threshold=0.05)
    bulk = np.sort(r.xis[np.abs(r.eigenvalues) < PERSIST])[::-1]
    LOG.info(f"  AR(1) xi_true=6.15 -> rank={r.rank} top xi={np.round(np.sort(r.xis)[::-1][:4],2)}")
    out["ar1"] = {"xi_true": 6.15, "rank": r.rank, "top_xi": [round(float(x), 2) for x in np.sort(r.xis)[::-1][:4]]}

    multi_xi = [2.0, 8.0, 30.0]
    rhos = [float(np.exp(-1 / x)) for x in multi_xi]
    Vm, _ = multi_mode_ar_process(800, 500, p=16, rhos=rhos, seed=0, orthonormal_mixing=True)
    res = two_point_function(Vm, max_delta=MAX_DELTA, whiten=True)
    r = ho_kalman(res.Chat.cpu().numpy(), rel_threshold=0.05)
    LOG.info(f"  multi xi_true={multi_xi} -> rank={r.rank} top xi={np.round(np.sort(r.xis)[::-1][:6],2)}")
    out["multi"] = {"xi_true": multi_xi, "rank": r.rank, "top_xi": [round(float(x), 2) for x in np.sort(r.xis)[::-1][:6]]}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUTDIR / "synth.json", "w"), indent=2)
    return out


def gpt2_layer_C(layer, p, device):
    shards = sorted(CACHE.glob("shard_*.pt"))
    s0 = torch.load(shards[0], map_location="cpu", weights_only=False)
    d_model = s0["meta"]["d_model"]
    r0 = s0["residuals"][layer][:, 1:, :].reshape(-1, d_model).float()
    pca = PCAPhi(d_model, p).fit(r0[torch.randperm(r0.shape[0])[:60000]]).to(device)
    acc = CorrelationAccumulator(p, MAX_DELTA, device=device, dtype=torch.float32)
    for sp in shards:
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        acc.update(pca(sh["residuals"][layer][:, 1:, :].float().to(device)))
        del sh
    return acc.finalize(whiten=True)


def run_gpt2(device):
    LOG.info("Exp 06 GPT-2: Ho-Kalman realization of residual correlations (fixed PCA p=64)")
    layers = [0, 2, 4, 6, 8, 10, 12]
    emp = {r["layer"]: r for r in json.load(open(CORR))["rows"]}
    rows, spectra = [], {}
    for layer in layers:
        res = gpt2_layer_C(layer, p=64, device=device)
        Chat = res.Chat.cpu().numpy()
        r = ho_kalman(Chat, rel_threshold=0.03, max_rank=30)
        # robust mode count from the block-Hankel singular spectrum (uncapped)
        sv = r.singular_values / r.singular_values[0]
        eff_modes = int(np.sum(sv > 0.05))
        mags = np.abs(r.eigenvalues)
        bulk_xi = np.sort(r.xis[mags < PERSIST])[::-1]
        row = {"layer": layer, "eff_modes_sv": eff_modes, "realized_rank": r.rank,
               "n_persistent_modes": int(np.sum(mags > PERSIST)),
               "bulk_xi_top": [round(float(x), 2) for x in bulk_xi[:5]],
               "exp01_bulk_xi": emp[layer]["bulk_xi"],
               "implied_D": int(np.ceil(np.sqrt(eff_modes + 1)))}
        rows.append(row)
        spectra[layer] = {"sv": sv, "mags": mags, "xis": r.xis}
        LOG.info(f"  L{layer:2d}: eff_modes(SV>5%)={eff_modes} impliedD={row['implied_D']} "
                 f"realized_bulk_xi={row['bulk_xi_top'][:3]} (Exp01={row['exp01_bulk_xi']})")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows}, open(OUTDIR / "gpt2.json", "w"), indent=2, default=str)
    _plot(rows, spectra, emp)
    return rows


def _plot(rows, spectra, emp):
    set_style()
    layers = [r["layer"] for r in rows]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(layers)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # block-Hankel singular spectrum (mode strengths) -> many significant modes
    for L, c in zip(layers, cmap):
        s = spectra[L]["sv"][:64]
        axes[0].semilogy(np.arange(1, len(s) + 1), s, "-", color=c, label=f"L{L}")
    axes[0].axhline(0.05, color="grey", ls=":", lw=1)
    axes[0].set_xlabel("mode index"); axes[0].set_ylabel("normalized Hankel singular value")
    axes[0].set_title("Correlation mode spectrum (block-Hankel SVs)\nmany modes > 5% ⇒ high-rank, not few-mode")
    axes[0].legend(fontsize=7, ncol=2)

    # realized bulk dominant xi vs Exp01 bulk xi
    realized = [r["bulk_xi_top"][0] if r["bulk_xi_top"] else np.nan for r in rows]
    exp01 = [r["exp01_bulk_xi"] for r in rows]
    x = np.arange(len(layers))
    axes[1].plot(x, exp01, "s--", color="#d62728", label="Exp 01 bulk ξ (fit)")
    axes[1].plot(x, realized, "o-", color="#08306b", label="Ho-Kalman dominant bulk ξ")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"L{L}" for L in layers])
    axes[1].set_xlabel("layer"); axes[1].set_ylabel("correlation length ξ")
    axes[1].set_title("Realized vs fitted bulk correlation length\n(same fixed PCA basis)")
    axes[1].legend(fontsize=8)

    # effective mode count + implied bond dimension per layer
    axes[2].plot(x, [r["eff_modes_sv"] for r in rows], "o-", color="#1f77b4",
                 label="eff. modes (Hankel SV > 5%)")
    axes[2].plot(x, [r["implied_D"] for r in rows], "s-", color="#d62728",
                 label=r"implied bond $D=\lceil\sqrt{M{+}1}\rceil$")
    axes[2].set_xticks(x); axes[2].set_xticklabels([f"L{L}" for L in layers])
    axes[2].set_xlabel("layer"); axes[2].set_ylabel("count")
    axes[2].set_title("Many correlation modes per layer\n⇒ large bond dimension needed")
    axes[2].legend(fontsize=8)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    save_fig(fig, FIGDIR / "fig1_realization.png")
    LOG.info(f"  saved {FIGDIR/'fig1_realization.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synth", "gpt2", "both"], default="both")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.mode in ("synth", "both"):
        run_synth()
    if args.mode in ("gpt2", "both"):
        run_gpt2(args.device)


if __name__ == "__main__":
    main()

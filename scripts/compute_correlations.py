#!/usr/bin/env python
r"""GPT-2 residual-stream correlation diagnostics (briefing Phase 2 / §5).

For each cached source layer (PCA-whitened to p=64), measure the whitened two-point
function vs token lag. Transformer residuals turn out to need a TWO-PART description:
a small long-range "persistent" subspace (a few directions correlated across all
positions) on top of a finite-correlation-length bulk. We therefore report:
  * operator norm  ||C_hat(D)||_op   -> tracks the single most-persistent direction
  * trace          Tr C_hat(D)       -> total correlation; fit floor + amp*exp(-D/xi)
  * # persistent directions (singular values of C_hat still > 0.5 at lag 32)
  * Hankel mode count of the trace.

Outputs:
  results/tables/correlation_fits_gpt2.csv
  docs/01_gpt2_correlations/figures/*.png
  results/runs/gpt2_correlations/results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tn_futurelens.analysis.correlations import CorrelationAccumulator
from tn_futurelens.analysis.exp_fits import (
    effective_mode_count_hankel,
    exp_plus_const_fit,
    single_exponential_fit,
)
from tn_futurelens.models.phi import PCAPhi
from tn_futurelens.utils.config import save_run_metadata
from tn_futurelens.utils.logging import CSVMetricWriter, get_logger
from tn_futurelens.utils.plotting import save_fig, set_style

import matplotlib.pyplot as plt

LOG = get_logger("corr")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt2" / "wikitext103"
FIGDIR = ROOT / "docs" / "01_gpt2_correlations" / "figures"
OUTDIR = ROOT / "results" / "runs" / "gpt2_correlations"
TABLE = ROOT / "results" / "tables" / "correlation_fits_gpt2.csv"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_DELTA = 64
PCA_P = 64
DROP_BOS = 1
PERSIST_DELTA = 32      # lag at which to count "persistent" directions
PERSIST_THRESH = 0.5    # singular value above this at PERSIST_DELTA == persistent


def fit_pcas(shard0, layers, n_sample=60000):
    phis = {}
    for layer in layers:
        r = shard0["residuals"][layer][:, DROP_BOS:, :]
        flat = r.reshape(-1, r.shape[-1]).float()
        idx = torch.randperm(flat.shape[0])[:n_sample]
        phis[layer] = PCAPhi(flat.shape[1], PCA_P).fit(flat[idx]).to(DEVICE)
    LOG.info(f"  fit PCA phi for layers {layers}")
    return phis


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    shards = sorted(CACHE.glob("shard_*.pt"))
    meta = torch.load(shards[0], map_location="cpu", weights_only=False)["meta"]
    layers = meta["layers"]
    LOG.info(f"layers={layers}  d_model={meta['d_model']}  device={DEVICE}  shards={len(shards)}")

    shard0 = torch.load(shards[0], map_location="cpu", weights_only=False)
    phis = fit_pcas(shard0, layers)
    del shard0

    accs = {l: CorrelationAccumulator(PCA_P, MAX_DELTA, device=DEVICE, dtype=torch.float32)
            for l in layers}
    for sp in shards:
        shard = torch.load(sp, map_location="cpu", weights_only=False)
        for l in layers:
            r = shard["residuals"][l][:, DROP_BOS:, :].float().to(DEVICE)
            accs[l].update(phis[l](r))
        del shard
        LOG.info(f"  accumulated {sp.name}")

    writer = CSVMetricWriter(TABLE)
    curves, rows = {}, []
    for layer in layers:
        res = accs[layer].finalize(whiten=True)
        d = res.deltas.cpu().numpy()
        op = res.operator(whitened=True).cpu().numpy()
        fro = res.frobenius(whitened=True).cpu().numpy()
        tr = res.trace(whitened=True).cpu().numpy()
        svals = res.singular_values(whitened=True).cpu().numpy()  # [nD, p]

        ec = exp_plus_const_fit(d, tr, delta_min=1, delta_max=MAX_DELTA)
        op_fit = single_exponential_fit(d, op, 1, 16)
        M, _ = effective_mode_count_hankel(tr[:40], rel_threshold=0.03)
        n_persist = int(np.sum(svals[min(PERSIST_DELTA, len(d) - 1)] > PERSIST_THRESH))

        curves[layer] = {"d": d, "op": op, "fro": fro, "tr": tr, "svals": svals, "ec": ec}
        row = {"model": "gpt2", "layer": layer, "phi": f"pca_whiten_p{PCA_P}", "p": PCA_P,
               "bulk_xi": round(ec.xi, 3), "expconst_r2": round(ec.r2, 4),
               "floor_trace": round(ec.floor, 3), "persistent_frac": round(ec.persistent_frac, 4),
               "n_persist_dims_lag32": n_persist, "op_xi": round(op_fit.xi, 2),
               "hankel_M": M, "predicted_D": int(np.ceil(np.sqrt(max(M, n_persist))))}
        rows.append(row)
        writer.append(row)
        LOG.info(f"  layer {layer:2d}: bulk_xi={ec.xi:.2f} (R2={ec.r2:.3f}) floor={ec.floor:.2f} "
                 f"persist_frac={ec.persistent_frac:.2f} n_persist={n_persist} M={M}")

    _plot(curves, rows)
    with open(OUTDIR / "results.json", "w") as f:
        json.dump({"rows": rows, "meta": {k: meta[k] for k in
                   ["model", "n_layers", "d_model", "layers", "seq_len"]},
                   "max_delta": MAX_DELTA, "persist_delta": PERSIST_DELTA}, f, indent=2, default=str)
    save_run_metadata(OUTDIR, config={"experiment": "gpt2_correlations", "pca_p": PCA_P},
                      metrics={"rows": rows})
    LOG.info(f"wrote {TABLE} and {OUTDIR/'results.json'}")
    print(json.dumps(rows, indent=2))


def _plot(curves, rows):
    set_style()
    layers = sorted(curves)
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(layers)))

    # Fig 1: trace correlation decay + exp+const fits (the key plot)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for layer, col in zip(layers, cmap):
        c = curves[layer]
        ax.plot(c["d"][1:], c["tr"][1:], "o", ms=3, color=col, label=f"layer {layer}")
        ec = c["ec"]
        if np.isfinite(ec.xi):
            dd = np.arange(1, c["d"].max() + 1)
            ax.plot(dd, ec.floor + ec.amplitude * np.exp(-dd / ec.xi), "-", color=col, lw=1.2)
    ax.set_xlabel(r"$\Delta$ (token lag)")
    ax.set_ylabel(r"$\mathrm{Tr}\,\hat C^\ell(\Delta)$  (whitened, $p{=}64$)")
    ax.set_title("GPT-2 residual correlation: a finite-$\\xi$ bulk decays onto a\n"
                 "long-range floor (markers=data, lines=floor+amp$\\cdot e^{-\\Delta/\\xi}$)")
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, FIGDIR / "fig1_trace_decay_fits.png")

    # Fig 2: bulk xi, persistent fraction, n_persist vs layer
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ly = [r["layer"] for r in rows]
    axes[0].plot(ly, [r["bulk_xi"] for r in rows], "o-", color="#1f77b4")
    axes[0].set_xlabel("layer $\\ell$"); axes[0].set_ylabel(r"bulk correlation length $\xi$")
    axes[0].set_title("Bulk (decaying-part) correlation length by layer")
    for r in rows:
        axes[0].annotate(f"$R^2$={r['expconst_r2']:.2f}", (r["layer"], r["bulk_xi"]),
                         fontsize=7, textcoords="offset points", xytext=(0, 6))
    ax2 = axes[1]
    ax2.bar([x - 0.5 for x in ly], [r["persistent_frac"] for r in rows], width=1.0,
            color="#d62728", alpha=0.4, label="persistent fraction (floor / lag-1)")
    ax2.set_ylabel("persistent fraction", color="#d62728"); ax2.set_ylim(0, 1)
    ax2b = ax2.twinx()
    ax2b.plot(ly, [r["n_persist_dims_lag32"] for r in rows], "ks-", label="# persistent dims (lag 32)")
    ax2b.set_ylabel("# directions still correlated at lag 32")
    ax2.set_xlabel("layer $\\ell$")
    ax2.set_title("Long-range structure grows with depth")
    lines = ax2.get_legend_handles_labels()[0] + ax2b.get_legend_handles_labels()[0]
    labs = ax2.get_legend_handles_labels()[1] + ax2b.get_legend_handles_labels()[1]
    ax2.legend(lines, labs, fontsize=8, loc="upper left")
    save_fig(fig, FIGDIR / "fig2_bulk_xi_persistence.png")

    # Fig 3: singular-value spectrum of C_hat(Delta) at several lags (representative layer)
    rep = layers[len(layers) // 2]
    c = curves[rep]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for dlag, col in zip([1, 2, 4, 8, 16, 32], plt.cm.plasma(np.linspace(0, 0.85, 6))):
        if dlag < len(c["d"]):
            ax.plot(np.arange(1, PCA_P + 1), np.sort(c["svals"][dlag])[::-1], color=col,
                    label=f"$\\Delta$={dlag}")
    ax.axhline(PERSIST_THRESH, color="grey", ls=":", lw=1)
    ax.set_xlabel("singular-value index"); ax.set_ylabel(r"singular value of $\hat C(\Delta)$")
    ax.set_title(f"Layer {rep}: a few directions stay correlated (top, persistent),\n"
                 "the bulk decays with lag")
    ax.legend(fontsize=8)
    save_fig(fig, FIGDIR / "fig3_singular_spectrum.png")
    LOG.info(f"  saved 3 figures to {FIGDIR}")


if __name__ == "__main__":
    main()

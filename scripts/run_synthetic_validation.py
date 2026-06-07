#!/usr/bin/env python
r"""Synthetic-validation experiment (briefing §14).

Validates, against processes with KNOWN answers, the parts of the pipeline that
the briefing's theory actually constrains:

  Part 1  Correlation diagnostics recover ground truth:
            AR(1) -> single exponential with xi = -1/ln rho;
            multi-mode AR -> several modes; power-law -> non-exponential (control).
  Part 2  The transfer-matrix COUNTING claim "bond D -> D^2-1 correlation modes,
            so representing M modes needs D ~ sqrt(M)" : we fit the measured
            correlation with (D^2-1)-mode models and show the reconstruction-error
            elbow lands at D ~ sqrt(M).
  Part 3  The trainable MPS layer is correct and captures multiplicative cross-site
            structure a linear probe cannot (and, honestly, has no edge on
            linear-Gaussian processes whose optimal predictor is linear).

Writes figures to docs/00_synthetic_validation/figures/ and a results JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tn_futurelens.analysis.correlations import two_point_function
from tn_futurelens.analysis.exp_fits import (
    effective_mode_count_hankel,
    hankel_singular_values,
    prony_modes,
    single_exponential_fit,
)
from tn_futurelens.data.synthetic import (
    ar1_process,
    multi_mode_ar_process,
    power_law_process,
)
from tn_futurelens.models.baselines import MultiSiteLinear, MultiSiteMLP
from tn_futurelens.models.mps import MPSReadout
from tn_futurelens.training.train import train_regression_probe
from tn_futurelens.utils.config import save_run_metadata
from tn_futurelens.utils.logging import get_logger
from tn_futurelens.utils.plotting import save_fig, set_style
from tn_futurelens.utils.seed import set_seed

import matplotlib.pyplot as plt

LOG = get_logger("synthetic")
ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs" / "00_synthetic_validation" / "figures"
OUTDIR = ROOT / "results" / "runs" / "synthetic_validation"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
C = {"ar1": "#1f77b4", "multi": "#d62728", "power": "#2ca02c"}


def xi_to_rho(xi):
    return float(np.exp(-1.0 / xi))


# ----------------------------------------------------------------------------
# Part 1: correlation diagnostics
# ----------------------------------------------------------------------------
def part1_diagnostics() -> dict:
    LOG.info("Part 1: correlation diagnostics")
    set_seed(0)
    max_delta = 50
    # well-separated modes so the (ill-conditioned) mode counting is clean
    multi_xi = [2.0, 8.0, 30.0]
    multi_rho = [xi_to_rho(x) for x in multi_xi]

    procs = {
        "ar1": (ar1_process(500, 700, p=8, rho=0.85, seed=0),
                {"true_M": 1, "true_xi": [-1 / np.log(0.85)]}),
        "multi": (multi_mode_ar_process(800, 800, p=16, rhos=multi_rho, seed=0,
                                        orthonormal_mixing=True)[0],
                  {"true_M": 3, "true_xi": multi_xi}),
        "power": (power_law_process(500, 800, p=8, alpha=1.0, seed=0)[0],
                  {"true_M": None, "true_xi": None}),
    }

    curves, rows = {}, []
    for name, (data, gt) in procs.items():
        res = two_point_function(data, max_delta=max_delta, whiten=True)
        d = res.deltas.cpu().numpy()
        op = res.operator(whitened=True).cpu().numpy()
        tr = res.trace(whitened=True).cpu().numpy()
        fit = single_exponential_fit(d, op, 1, 12)
        M_hankel, sv = effective_mode_count_hankel(tr[:30], rel_threshold=0.03)
        curves[name] = {"d": d, "op": op, "sv": sv, "fit": fit}
        rows.append({
            "process": name, "true_M": gt["true_M"],
            "true_xi": None if gt["true_xi"] is None else [round(x, 2) for x in gt["true_xi"]],
            "fit_xi_op": round(fit.xi, 2), "fit_r2": round(fit.r2, 4),
            "hankel_M": M_hankel, "predicted_D": int(np.ceil(np.sqrt(M_hankel))),
        })
        LOG.info(f"  {name}: xi={fit.xi:.2f} r2={fit.r2:.4f} hankelM={M_hankel}")
    _plot_part1(curves)
    return {"table": rows, "multi_xi": multi_xi}


def _plot_part1(curves):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    titles = {"ar1": "AR(1) $\\rho{=}0.85$ — 1 mode",
              "multi": "3-mode AR ($\\xi{=}2,8,30$)",
              "power": "power-law $(1{+}\\Delta)^{-1}$ — control"}
    # panel 0+1: decay curves with single-exp fit
    for ax, name in zip(axes[:2], ["ar1", "multi"]):
        c = curves[name]
        m = c["op"] > 1e-4
        ax.semilogy(c["d"][m], c["op"][m], "o", ms=3.5, color=C[name],
                    label=r"$\|\hat C(\Delta)\|_{op}$")
        f = c["fit"]
        dd = np.arange(1, 22)
        ax.semilogy(dd, np.exp(f.log_amplitude + f.slope * dd), "k--", lw=1.3,
                    label=fr"single-exp $\xi$={f.xi:.2f}, $R^2$={f.r2:.3f}")
        ax.set_title(titles[name]); ax.set_xlabel(r"$\Delta$ (token lag)")
        ax.set_ylabel("whitened correlation"); ax.legend(fontsize=8)
    # panel 2: power-law on log-log to show it's a line there (curved on semilog)
    c = curves["power"]; m = c["op"] > 1e-4
    axes[2].loglog(c["d"][m] + 1, c["op"][m], "o", ms=3.5, color=C["power"],
                   label=r"$\|\hat C\|_{op}$ (power-law)")
    axes[2].loglog(c["d"][m] + 1, (c["d"][m] + 1) ** -1.0, "k--", lw=1.2,
                   label=r"$(1+\Delta)^{-1}$ reference")
    axes[2].set_title(titles["power"] + f"\nsingle-exp $R^2$={c['fit'].r2:.3f} (worse)")
    axes[2].set_xlabel(r"$1+\Delta$"); axes[2].set_ylabel("whitened correlation")
    axes[2].legend(fontsize=8)
    fig.suptitle("Two-point correlation decay: AR is a straight line on a log plot "
                 "(exponential); power-law is straight on log–log (not exponential)", y=1.02)
    save_fig(fig, FIGDIR / "fig1_correlation_decay.png")
    LOG.info(f"  saved {FIGDIR/'fig1_correlation_decay.png'}")


# ----------------------------------------------------------------------------
# Part 2: D ~ sqrt(M) representation test (the transfer-matrix counting claim)
# ----------------------------------------------------------------------------
def part2_dsqrtm() -> dict:
    LOG.info("Part 2: D ~ sqrt(M) representation test")
    set_seed(0)
    Ms = [1, 2, 4, 7]
    D_values = [1, 2, 3, 4]
    max_delta = 60
    out = {}
    for M in Ms:
        xis = np.geomspace(1.8, 35.0, M)        # well-separated correlation lengths
        rhos = [xi_to_rho(x) for x in xis]
        V, _ = multi_mode_ar_process(900, 900, p=24, rhos=[float(r) for r in rhos],
                                     seed=M, orthonormal_mixing=True)
        res = two_point_function(V, max_delta=max_delta, whiten=True)
        c = res.trace(whitened=True).cpu().numpy()
        var_c = float(np.var(c))
        errs = []
        for D in D_values:
            K = D * D - 1                       # # correlation modes a bond-D MPS supports
            if K < 1:
                rmse = float(np.sqrt(var_c))    # K=0: only a constant
            else:
                rmse = prony_modes(c, n_modes=min(K, max_delta // 2 - 1)).reconstruction_rmse
            errs.append({"D": D, "K_modes": K, "rmse": rmse})
        out[M] = {"xis": [round(float(x), 2) for x in xis], "errs": errs,
                  "sqrt_M": float(np.sqrt(M))}
        LOG.info(f"  M={M}: rmse vs D = " + ", ".join(f"D{e['D']}={e['rmse']:.2e}" for e in errs))
    _plot_part2(out, D_values)
    return out


def _plot_part2(out, D_values):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(out)))
    elbow = {}
    for (M, r), col in zip(out.items(), cmap):
        Ds = [e["D"] for e in r["errs"]]
        rmse = [max(e["rmse"], 1e-12) for e in r["errs"]]
        axes[0].semilogy(Ds, rmse, "o-", color=col, label=f"M={M} modes")
        # elbow: smallest D whose rmse is within 5x of the best (noise floor)
        best = min(rmse)
        elbow[M] = next(e["D"] for e in r["errs"] if max(e["rmse"], 1e-12) <= 5 * best)
        axes[0].axvline(np.sqrt(M), color=col, ls=":", alpha=0.5)
    axes[0].set_xlabel("bond dimension $D$  (supports $D^2{-}1$ modes)")
    axes[0].set_ylabel("correlation reconstruction RMSE")
    axes[0].set_title("Fitting the measured correlation with a $(D^2{-}1)$-mode model\n"
                      "(dotted verticals = $\\sqrt{M}$)")
    axes[0].set_xticks(D_values); axes[0].legend(fontsize=9)

    Ms = sorted(elbow)
    axes[1].plot(Ms, [elbow[M] for M in Ms], "ks-", ms=8, label="elbow $D$ (RMSE hits floor)")
    grid = np.linspace(1, max(Ms), 60)
    axes[1].plot(grid, np.ceil(np.sqrt(grid + 1)), "--", color="#1f77b4",
                 label=r"$\lceil\sqrt{M{+}1}\,\rceil$  ($D^2{-}1\geq M$)")
    axes[1].set_xlabel("ground-truth mode count $M$")
    axes[1].set_ylabel("bond dimension needed")
    axes[1].set_title("Representation cost: useful $D$ tracks $\\sqrt{M}$")
    axes[1].legend(fontsize=9)
    save_fig(fig, FIGDIR / "fig2_d_sqrt_m.png")
    LOG.info(f"  saved fig2; elbow_D={elbow}")
    return elbow


# ----------------------------------------------------------------------------
# Part 3: trainable MPS correctness on a multiplicative cross-site task
# ----------------------------------------------------------------------------
def part3_mps_correctness() -> dict:
    LOG.info("Part 3: trainable MPS on multiplicative cross-site task")
    set_seed(0)
    B, p, m, d_out = 20000, 8, 5, 4
    X = torch.randn(B, m, p) * 0.7 + 0.3
    # target channel c = product over sites of a per-site linear form -> degree-m polynomial.
    # An MPS (product of input-dependent matrices) represents this; a linear probe cannot.
    A = torch.randn(d_out, m, p) * 0.5
    proj = torch.einsum("cjp,bjp->bcj", A, X)         # [B, d_out, m]
    Y = proj.prod(dim=-1).unsqueeze(1)                # [B, 1, d_out]  (n_horizons=1)
    Y = (Y - Y.mean(0)) / Y.std(0)
    n_tr = int(0.8 * B)
    sp = lambda t: (t[:n_tr], t[n_tr:])
    Xtr, Xva = sp(X); Ytr, Yva = sp(Y)

    results = {}
    models = {
        "linear": MultiSiteLinear(p, m, d_out, 1),
        "mlp": MultiSiteMLP(p, m, d_out, 1, hidden=128, depth=2),
        "mps_D4": MPSReadout(p=p, D=4, n_sites=m, readout="env", out_dim=d_out, n_heads=1, seed=0),
        "mps_D8": MPSReadout(p=p, D=8, n_sites=m, readout="env", out_dim=d_out, n_heads=1, seed=0),
    }
    for name, model in models.items():
        r = train_regression_probe(model, Xtr, Ytr, Xva, Yva, epochs=200, lr=2e-3,
                                   batch_size=4096, device=DEVICE, patience=30, seed=0)
        results[name] = {"val_nmse": round(r.best_val_nmse, 4), "n_params": r.n_params}
        LOG.info(f"  {name:10s} val_nmse={r.best_val_nmse:.4f} params={r.n_params}")
    _plot_part3(results)
    return results


def _plot_part3(results):
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    names = list(results)
    nmse = [results[n]["val_nmse"] for n in names]
    colors = ["#7f7f7f", "#ff7f0e", "#1f77b4", "#08519c"]
    bars = ax.bar(names, nmse, color=colors)
    for b, n in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{results[n]['val_nmse']:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("validation NMSE (lower = better)")
    ax.set_title("Multiplicative cross-site task: MPS captures the product\nstructure a linear "
                 "probe cannot (NMSE→1 = predicting the mean)")
    ax.axhline(1.0, color="k", ls=":", lw=1, alpha=0.6)
    save_fig(fig, FIGDIR / "fig3_mps_multiplicative.png")
    LOG.info(f"  saved {FIGDIR/'fig3_mps_multiplicative.png'}")


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOG.info(f"device={DEVICE}")
    out = {
        "diagnostics": part1_diagnostics(),
        "d_sqrt_m": {str(k): v for k, v in part2_dsqrtm().items()},
        "mps_correctness": part3_mps_correctness(),
    }
    with open(OUTDIR / "results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    save_run_metadata(OUTDIR, config={"experiment": "synthetic_validation", "device": DEVICE},
                      metrics={"mps_correctness": out["mps_correctness"]})
    print("\n==== DIAGNOSTICS ====")
    print(json.dumps(out["diagnostics"]["table"], indent=2, default=str))
    print("==== MPS CORRECTNESS ====")
    print(json.dumps(out["mps_correctness"], indent=2))


if __name__ == "__main__":
    main()

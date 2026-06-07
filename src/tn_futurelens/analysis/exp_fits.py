r"""Exponential-decay fitting of correlation summaries (briefing §5.2).

  * single_exponential_fit: log-linear least squares  log c(Delta) ~ log A - Delta/xi,
    with R^2 (the primary diagnostic).
  * prony_modes: linear Prony / least-squares extraction of a SUM of exponentials
    c(Delta) = sum_mu a_mu lambda_mu^Delta  (matches the MPS transfer-matrix form).
  * estimate_mode_count: choose M by BIC over Prony fits -> predicted D ~ sqrt(M).

All operate on a 1-D scalar correlation summary c(Delta) (e.g. the whitened
operator norm ||Chat(Delta)||_op), Delta = 0, 1, 2, ...
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SingleExpFit:
    xi: float           # correlation length
    log_amplitude: float
    r2: float
    slope: float        # = -1/xi
    delta_range: tuple[int, int]


def single_exponential_fit(
    deltas: np.ndarray,
    values: np.ndarray,
    delta_min: int = 1,
    delta_max: int | None = None,
) -> SingleExpFit:
    r"""Fit ``log c ~ log A - Delta/xi`` by least squares over ``[delta_min, delta_max]``."""
    deltas = np.asarray(deltas, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if delta_max is None:
        delta_max = int(deltas.max())
    mask = (deltas >= delta_min) & (deltas <= delta_max) & (values > 0)
    d = deltas[mask]
    y = np.log(values[mask])
    if d.size < 2:
        return SingleExpFit(np.nan, np.nan, np.nan, np.nan, (delta_min, delta_max))
    A = np.vstack([d, np.ones_like(d)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, intercept])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    xi = -1.0 / slope if slope < 0 else np.inf
    return SingleExpFit(
        xi=float(xi),
        log_amplitude=float(intercept),
        r2=float(r2),
        slope=float(slope),
        delta_range=(delta_min, delta_max),
    )


@dataclass
class PronyFit:
    n_modes: int
    lambdas: np.ndarray     # decay bases z_mu (real or complex), |z|<1 expected
    amplitudes: np.ndarray  # a_mu
    xis: np.ndarray         # -1/ln|z_mu|
    reconstruction_rmse: float
    bic: float


def prony_modes(c: np.ndarray, n_modes: int) -> PronyFit:
    r"""Least-squares Prony: fit ``c[k] = sum_mu a_mu z_mu^k`` with ``n_modes`` modes.

    Step 1 (find poles): solve the LS Hankel system for the linear-prediction
    coefficients, then root the characteristic polynomial.
    Step 2 (find amplitudes): solve the Vandermonde LS system.
    """
    c = np.asarray(c, dtype=np.float64)
    N = c.size
    M = n_modes
    if N < 2 * M:
        raise ValueError(f"need >= 2*n_modes={2*M} samples, got {N}")
    # Step 1: Hankel system  sum_{j=1}^M p_j c[k-j] = -c[k],  k = M..N-1
    rows = N - M
    H = np.empty((rows, M))
    rhs = np.empty(rows)
    for r in range(rows):
        k = M + r
        idx = k - 1 - np.arange(M)        # indices k-1, k-2, ..., k-M (explicit, no neg-stop slice)
        H[r] = c[idx]
        rhs[r] = -c[k]
    p_coef, *_ = np.linalg.lstsq(H, rhs, rcond=None)
    # characteristic polynomial: z^M + p_1 z^{M-1} + ... + p_M = 0
    poly = np.concatenate([[1.0], p_coef])
    roots = np.roots(poly)
    # Step 2: amplitudes via Vandermonde LS over k = 0..N-1
    k = np.arange(N)
    V = roots[None, :] ** k[:, None]          # [N, M]
    amps, *_ = np.linalg.lstsq(V, c, rcond=None)
    recon = (V @ amps).real
    rmse = float(np.sqrt(np.mean((recon - c) ** 2)))
    # BIC for model selection (gaussian residuals); params ~ 2*M (pole+amp)
    n_params = 2 * M
    sigma2 = max(rmse ** 2, 1e-300)
    bic = N * np.log(sigma2) + n_params * np.log(max(N, 2))
    mag = np.abs(roots).clip(1e-12, 1 - 1e-12)
    xis = -1.0 / np.log(mag)
    return PronyFit(
        n_modes=M,
        lambdas=roots,
        amplitudes=amps,
        xis=xis,
        reconstruction_rmse=rmse,
        bic=float(bic),
    )


@dataclass
class ModeCountEstimate:
    n_modes: int
    predicted_bond_dim: int   # ceil(sqrt(M))
    best_fit: PronyFit
    bic_by_m: dict[int, float]


def estimate_mode_count(
    c: np.ndarray,
    max_modes: int = 6,
    amplitude_rel_threshold: float = 1e-3,
) -> ModeCountEstimate:
    r"""Estimate effective exponential-mode count via BIC-selected Prony fit.

    Returns the BIC-optimal ``M`` (counting only modes whose amplitude exceeds
    ``amplitude_rel_threshold`` times the largest), and ``D ~ ceil(sqrt(M))``.
    """
    c = np.asarray(c, dtype=np.float64)
    N = c.size
    m_max = min(max_modes, N // 2)
    bic_by_m: dict[int, float] = {}
    fits: dict[int, PronyFit] = {}
    for M in range(1, m_max + 1):
        try:
            fit = prony_modes(c, M)
        except Exception:
            continue
        bic_by_m[M] = fit.bic
        fits[M] = fit
    if not fits:
        raise RuntimeError("Prony fitting failed for all M")
    best_m = min(bic_by_m, key=bic_by_m.get)
    best = fits[best_m]
    # count significant modes
    a = np.abs(best.amplitudes)
    sig = int(np.sum(a > amplitude_rel_threshold * a.max())) if a.size else best_m
    sig = max(sig, 1)
    return ModeCountEstimate(
        n_modes=sig,
        predicted_bond_dim=int(np.ceil(np.sqrt(sig))),
        best_fit=best,
        bic_by_m=bic_by_m,
    )

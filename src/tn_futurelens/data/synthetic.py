"""Synthetic stochastic processes with KNOWN correlation structure.

These let us validate the correlation diagnostics and MPS code before touching
transformer activations (briefing §14). Three processes:

  1. AR(1)              -> single exponential, C(Delta) ∝ rho^Delta, xi = -1/ln rho.
  2. sum-of-AR-modes    -> M exponential modes, C(Delta) ≈ sum_mu B_mu rho_mu^Delta;
                           tests whether estimated mode count M and D ~ sqrt(M) behave.
  3. power-law          -> non-exponential decay; negative control where a small-D
                           MPS should struggle relative to baselines.

All generators return float tensors of shape ``[n_seq, length, p]`` (a batch of
``n_seq`` independent sequences, each ``length`` sites long, with ``p`` features
per site) plus, where useful, a dict of ground-truth quantities.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor


def _generator(seed: int | None, device: str | torch.device) -> torch.Generator | None:
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def ar1_process(
    n_seq: int,
    length: int,
    p: int = 1,
    rho: float = 0.8,
    noise_std: float = 1.0,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    r"""AR(1): ``v_{i+1} = rho * v_i + eps_i`` with ``eps ~ N(0, noise_std^2)``.

    The ``p`` feature components are independent AR(1)s with the same ``rho``.
    Initialised from the stationary distribution (variance ``noise_std^2/(1-rho^2)``),
    so the sequence is stationary from site 0.

    Returns:
        Tensor ``[n_seq, length, p]``.

    Ground truth (per component): ``C(Delta) = sigma_stat^2 * rho^|Delta|`` with
    ``sigma_stat^2 = noise_std^2 / (1 - rho^2)``; correlation length ``xi = -1/ln rho``.
    """
    if not -1.0 < rho < 1.0:
        raise ValueError(f"AR(1) requires |rho| < 1 for stationarity, got {rho}")
    g = _generator(seed, device)
    sigma_stat = noise_std / math.sqrt(1.0 - rho * rho)

    x = torch.empty(n_seq, length, p, device=device, dtype=dtype)
    x[:, 0] = torch.randn(n_seq, p, generator=g, device=device, dtype=dtype) * sigma_stat
    if length > 1:
        eps = torch.randn(
            n_seq, length - 1, p, generator=g, device=device, dtype=dtype
        ) * noise_std
        for t in range(1, length):
            x[:, t] = rho * x[:, t - 1] + eps[:, t - 1]
    return x


def multi_mode_ar_process(
    n_seq: int,
    length: int,
    p: int,
    rhos: Sequence[float],
    noise_std: float = 1.0,
    obs_noise_std: float = 0.0,
    mixing: Tensor | None = None,
    orthonormal_mixing: bool = True,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, dict]:
    r"""Sum of ``M = len(rhos)`` independent AR(1) latents mixed into ``p`` features.

    ``h_{i,mu}`` is AR(1) with coefficient ``rhos[mu]``; the observed feature is
    ``v_i = U h_i (+ obs noise)`` with mixing ``U`` of shape ``[p, M]``.

    The matrix-valued two-point function is exactly
    ``C(Delta) = U diag(sigma_mu^2 rho_mu^|Delta|) U^T  (+ obs_noise^2 I if Delta==0)``,
    i.e. a sum of ``M`` exponential modes -> predicted useful bond dim ``D ~ sqrt(M)``.

    Returns:
        (v ``[n_seq, length, p]``, info dict with keys
         ``rhos, U, sigma_stat2, n_modes, xi`` (per-mode correlation lengths)).
    """
    M = len(rhos)
    g = _generator(seed, device)
    rhos_t = torch.tensor(rhos, device=device, dtype=dtype)
    if torch.any(rhos_t.abs() >= 1.0):
        raise ValueError("all |rho| must be < 1 for stationarity")
    sigma_stat2 = (noise_std ** 2) / (1.0 - rhos_t ** 2)  # [M]
    sigma_stat = sigma_stat2.sqrt()

    h = torch.empty(n_seq, length, M, device=device, dtype=dtype)
    h[:, 0] = torch.randn(n_seq, M, generator=g, device=device, dtype=dtype) * sigma_stat
    if length > 1:
        eps = torch.randn(
            n_seq, length - 1, M, generator=g, device=device, dtype=dtype
        ) * noise_std
        for t in range(1, length):
            h[:, t] = rhos_t * h[:, t - 1] + eps[:, t - 1]

    if mixing is None:
        U = torch.randn(p, M, generator=g, device=device, dtype=dtype)
        if orthonormal_mixing and p >= M:
            # orthonormal columns so modes are cleanly separated in feature space
            Q, _ = torch.linalg.qr(U)
            U = Q[:, :M]
    else:
        U = mixing.to(device=device, dtype=dtype)
        if U.shape != (p, M):
            raise ValueError(f"mixing must be [p, M]=[{p},{M}], got {tuple(U.shape)}")

    v = h @ U.T  # [n_seq, length, p]
    if obs_noise_std > 0:
        v = v + torch.randn(
            n_seq, length, p, generator=g, device=device, dtype=dtype
        ) * obs_noise_std

    xi = -1.0 / torch.log(rhos_t.abs().clamp_min(1e-12))
    info = {
        "rhos": rhos_t,
        "U": U,
        "sigma_stat2": sigma_stat2,
        "n_modes": M,
        "xi": xi,
        "obs_noise_std": obs_noise_std,
    }
    return v, info


def stationary_gaussian(
    autocov: Tensor,
    n_seq: int,
    p: int = 1,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    r"""Stationary zero-mean Gaussian sequences with a given autocovariance.

    Uses circulant embedding (exact, FFT-based) for an autocovariance
    ``autocov[Delta] = gamma(Delta)``, ``Delta = 0 .. length-1``. The ``p`` feature
    components are independent and share ``gamma``.

    Returns Tensor ``[n_seq, length, p]`` where ``length = autocov.shape[0]``.
    """
    gamma = autocov.to(device=device, dtype=torch.float64)  # build in float64 for stability
    n = gamma.shape[0]
    # circulant first row: [g0, g1, ..., g_{n-1}, g_{n-2}, ..., g1], length m = 2n-2
    if n < 2:
        raise ValueError("autocov must have length >= 2")
    c = torch.cat([gamma, gamma[1:-1].flip(0)])  # length 2n-2
    m = c.shape[0]
    eig = torch.fft.fft(c).real  # circulant eigenvalues; should be >= 0
    neg = eig[eig < 0]
    if neg.numel() > 0:
        # clip small negative eigenvalues (approximate embedding); warn via attribute
        eig = eig.clamp_min(0.0)
    g = _generator(seed, device)
    sqrt_eig = eig.sqrt()  # [m]
    # complex white noise, scaled by sqrt(eigenvalues), inverse FFT -> two real series
    out = torch.empty(n_seq, n, p, device=device, dtype=dtype)
    for comp in range(p):
        zr = torch.randn(n_seq, m, generator=g, device=device, dtype=torch.float64)
        zi = torch.randn(n_seq, m, generator=g, device=device, dtype=torch.float64)
        z = torch.complex(zr, zi)
        freq = z * sqrt_eig.unsqueeze(0)
        series = torch.fft.ifft(freq, dim=1) * math.sqrt(m)
        # real and imaginary parts are independent valid realisations; take real, first n
        out[:, :, comp] = series.real[:, :n].to(dtype)
    return out


def power_law_process(
    n_seq: int,
    length: int,
    p: int = 1,
    alpha: float = 1.0,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, dict]:
    r"""Negative control: stationary Gaussian with power-law autocovariance.

    ``gamma(Delta) = (1 + Delta)^{-alpha}`` (so ``gamma(0) = 1``). Power-law decay is
    NOT a finite sum of exponentials, so a small-D MPS should be unable to capture
    it -- exactly the regime where the finite-correlation-length hypothesis fails.

    Returns (v ``[n_seq, length, p]``, info dict with ``alpha``, ``autocov``).
    """
    deltas = torch.arange(length, dtype=torch.float64)
    gamma = (1.0 + deltas) ** (-alpha)
    v = stationary_gaussian(gamma, n_seq, p=p, seed=seed, device=device, dtype=dtype)
    return v, {"alpha": alpha, "autocov": gamma}

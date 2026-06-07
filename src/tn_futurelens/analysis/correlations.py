r"""Residual-stream two-point correlation diagnostics (briefing §5).

For centered physical features ``Vtilde_i = V_i - E[V_i]`` the matrix-valued
two-point function is

    C(Delta) = E_i[ Vtilde_i Vtilde_{i+Delta}^T ]  in R^{p x p}.

We summarise it by Frobenius / operator / trace norms, optionally after whitening

    Chat(Delta) = (C(0)+eps I)^{-1/2} C(Delta) (C(0)+eps I)^{-1/2},

which removes the static covariance so a pure AR(1) gives ||Chat(Delta)|| = rho^Delta.

Correlations are computed WITHIN sequences only (each row of the [S, T, p] batch is
an independent sequence / document) so we never correlate across document boundaries.
Under stationarity (constant mean in i) we use the single-pass identity
``C(Delta) = E[V_i V_{i+Delta}^T] - mu mu^T``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


def matrix_sqrt_inv(M: Tensor, eps: float = 1e-5) -> Tensor:
    """``(M + eps I)^{-1/2}`` via symmetric eigendecomposition (M assumed symmetric PSD)."""
    M = 0.5 * (M + M.transpose(-1, -2))
    p = M.shape[-1]
    M = M + eps * torch.eye(p, device=M.device, dtype=M.dtype)
    evals, evecs = torch.linalg.eigh(M)
    inv_sqrt = (evals.clamp_min(eps)) ** -0.5
    return (evecs * inv_sqrt.unsqueeze(-2)) @ evecs.transpose(-1, -2)


@dataclass
class CorrelationResult:
    """Output of a correlation computation."""

    deltas: Tensor                      # [nD] integer lags
    C: Tensor                           # [nD, p, p] matrix two-point fns
    mean: Tensor                        # [p] feature mean
    counts: Tensor                      # [nD] number of pairs averaged per lag
    Chat: Tensor | None = None          # [nD, p, p] whitened, if computed
    extra: dict = field(default_factory=dict)

    # -- scalar summaries -------------------------------------------------
    def frobenius(self, whitened: bool = False) -> Tensor:
        M = self.Chat if (whitened and self.Chat is not None) else self.C
        return torch.linalg.matrix_norm(M, ord="fro")

    def operator(self, whitened: bool = False) -> Tensor:
        M = self.Chat if (whitened and self.Chat is not None) else self.C
        return torch.linalg.matrix_norm(M, ord=2)

    def trace(self, whitened: bool = False) -> Tensor:
        M = self.Chat if (whitened and self.Chat is not None) else self.C
        return M.diagonal(dim1=-2, dim2=-1).sum(-1)

    def singular_values(self, whitened: bool = False) -> Tensor:
        M = self.Chat if (whitened and self.Chat is not None) else self.C
        return torch.linalg.svdvals(M)  # [nD, p]


class CorrelationAccumulator:
    """Streaming, single-pass accumulation of two-point statistics over chunks.

    Call :meth:`update` with chunks ``[S, T, p]`` (S independent sequences), then
    :meth:`finalize`. Memory is ``O(nD * p^2)``; works for the GPT-2-scale cache.
    """

    def __init__(self, p: int, max_delta: int, device="cpu", dtype=torch.float64):
        self.p = p
        self.deltas = torch.arange(0, max_delta + 1)
        self.device = device
        self.dtype = dtype
        self._sum = torch.zeros(p, device=device, dtype=dtype)         # sum_i V_i
        self._n = 0                                                     # count for mean
        self._G = torch.zeros(len(self.deltas), p, p, device=device, dtype=dtype)
        self._cnt = torch.zeros(len(self.deltas), device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, V: Tensor) -> None:
        if V.ndim == 2:
            V = V.unsqueeze(0)
        V = V.to(self.device, self.dtype)
        S, T, p = V.shape
        if p != self.p:
            raise ValueError(f"feature dim {p} != accumulator p {self.p}")
        self._sum += V.sum(dim=(0, 1))
        self._n += S * T
        for k, d in enumerate(self.deltas.tolist()):
            if d >= T:
                continue
            a = V[:, : T - d, :]      # [S, T-d, p]
            b = V[:, d:, :]           # [S, T-d, p]
            self._G[k] += torch.einsum("stp,stq->pq", a, b)
            self._cnt[k] += S * (T - d)

    @torch.no_grad()
    def finalize(self, whiten: bool = True, eps: float = 1e-5) -> CorrelationResult:
        mu = self._sum / max(self._n, 1)
        outer = torch.outer(mu, mu)
        C = torch.empty_like(self._G)
        for k in range(len(self.deltas)):
            cnt = self._cnt[k].clamp_min(1.0)
            C[k] = self._G[k] / cnt - outer
        # symmetrize C(0) for whitening
        Chat = None
        if whiten:
            W = matrix_sqrt_inv(C[0], eps=eps)
            Chat = torch.einsum("ij,djk,kl->dil", W, C, W)
        return CorrelationResult(
            deltas=self.deltas.clone(),
            C=C,
            mean=mu,
            counts=self._cnt.clone(),
            Chat=Chat,
        )


@torch.no_grad()
def two_point_function(
    V: Tensor,
    max_delta: int,
    whiten: bool = True,
    eps: float = 1e-5,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> CorrelationResult:
    """Convenience in-memory wrapper around :class:`CorrelationAccumulator`.

    ``V``: ``[S, T, p]`` or ``[T, p]``. Returns a :class:`CorrelationResult`.
    """
    if V.ndim == 2:
        V = V.unsqueeze(0)
    p = V.shape[-1]
    acc = CorrelationAccumulator(
        p, max_delta, device=device or V.device, dtype=dtype
    )
    acc.update(V)
    return acc.finalize(whiten=whiten, eps=eps)

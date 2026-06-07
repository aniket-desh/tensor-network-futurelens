r"""Physical site maps phi: R^{d_model} -> R^p.

Map a residual vector to a local physical feature vector for the MPS (briefing §3).
All maps here are AFFINE, which (a) sidesteps the bounded-domain requirement of the
classic cos/sin map -- transformer activations are unbounded reals -- and (b) is
provably absorbable into the MPS local tensor (briefing §3.2):

    A_j(W_phi r) = sum_b r_b ( sum_a (W_phi)_{ab} A_j^a ).

So a linear phi just learns the local physical basis while D controls inter-site
virtual correlations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class IdentityPhi(nn.Module):
    """phi(r) = r, so p = d_model. Simplest, expensive."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.p = d_model

    def forward(self, r: Tensor) -> Tensor:
        return r


class PCAPhi(nn.Module):
    r"""PCA / whitening map (briefing §3.2, option 2).

    phi(r) = (Lambda_p + eps I)^{-1/2} U_p^T (r - mu),

    with mu, U_p (top-p eigenvectors), Lambda_p (top-p eigenvalues) estimated from
    data via :meth:`fit`. Produces approximately whitened features (unit covariance).
    Buffers (not trained) so they serialize with the model.
    """

    def __init__(self, d_model: int, p: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.p = p
        self.eps = eps
        self.register_buffer("mean", torch.zeros(d_model))
        self.register_buffer("components", torch.zeros(p, d_model))  # U_p^T rows
        self.register_buffer("scale", torch.ones(p))  # (lambda + eps)^{-1/2}
        self._fitted = False

    @torch.no_grad()
    def fit(self, x: Tensor) -> "PCAPhi":
        """Fit mean/eigvecs/eigvals from samples ``x`` of shape ``[N, d_model]``."""
        if x.ndim != 2 or x.shape[1] != self.d_model:
            raise ValueError(f"expected [N, {self.d_model}], got {tuple(x.shape)}")
        x = x.to(torch.float64)
        mu = x.mean(dim=0)
        xc = x - mu
        # covariance via SVD of centered data (more stable than forming Sigma)
        # xc = U S V^T ; eigenvectors of cov are columns of V, eigenvalues S^2/(N-1)
        _, S, Vh = torch.linalg.svd(xc, full_matrices=False)
        eigvals = (S ** 2) / (x.shape[0] - 1)
        comps = Vh[: self.p]  # [p, d_model], rows are top-p eigenvectors
        lam_p = eigvals[: self.p]
        self.mean.copy_(mu.to(self.mean.dtype))
        self.components.copy_(comps.to(self.components.dtype))
        self.scale.copy_(((lam_p + self.eps) ** -0.5).to(self.scale.dtype))
        self._fitted = True
        return self

    def forward(self, r: Tensor) -> Tensor:
        # (r - mu) @ U_p^T -> [..., p], then scale per-component
        centered = r - self.mean
        proj = centered @ self.components.T  # [..., p]
        return proj * self.scale


class LearnedLinearPhi(nn.Module):
    r"""Learned linear map phi(r) = W_phi r + b_phi (briefing §3.2, option 3).

    ``p`` is a hyperparameter. Can be initialised from a fitted :class:`PCAPhi`.
    """

    def __init__(self, d_model: int, p: int, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.p = p
        self.linear = nn.Linear(d_model, p, bias=bias)

    @torch.no_grad()
    def init_from_pca(self, pca: PCAPhi) -> "LearnedLinearPhi":
        """Warm-start from PCA whitening: W = diag(scale) U_p, b = -W mu."""
        W = pca.scale[:, None] * pca.components  # [p, d_model]
        self.linear.weight.copy_(W.to(self.linear.weight.dtype))
        if self.linear.bias is not None:
            self.linear.bias.copy_((-(W @ pca.mean)).to(self.linear.bias.dtype))
        return self

    def forward(self, r: Tensor) -> Tensor:
        return self.linear(r)


def build_phi(kind: str, d_model: int, p: int | None = None, **kwargs) -> nn.Module:
    """Factory: ``kind`` in {"identity", "pca", "learned"}."""
    kind = kind.lower()
    if kind == "identity":
        return IdentityPhi(d_model)
    if kind == "pca":
        if p is None:
            raise ValueError("pca phi needs p")
        return PCAPhi(d_model, p, **kwargs)
    if kind == "learned":
        if p is None:
            raise ValueError("learned phi needs p")
        return LearnedLinearPhi(d_model, p, **kwargs)
    raise ValueError(f"unknown phi kind {kind!r}")

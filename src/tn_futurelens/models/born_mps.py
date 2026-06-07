r"""Born-machine MPS over discrete (quantized) residual symbols (briefing B6 / §6).

Models a probability distribution over chains of discrete symbols
``z = (z_1,...,z_N), z_j in {0..K-1}`` via the Born rule

    P(z) = |Psi(z)|^2 / Z ,   Psi(z) = l^T A^{z_1} A^{z_2} ... A^{z_N} r ,   Z = sum_z |Psi(z)|^2.

``Z`` is computed in O(N) by the double-layer transfer matrix E_j = sum_k A_j^k (x) A_j^k.
Conditioning is exact: given observed z_{1:m}, the distribution over the next symbol is
``P(z_{m+1}=k | z_{1:m}) ∝ v_k^T R v_k`` with ``v_k = (l^T prod_{j<=m} A_j^{z_j}) A_{m+1}^k``
and ``R`` the right marginal environment -- the literal "clamp observed sites, complete
the next site" object the project is about. Real cores; log-norm stabilised.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class BornMPS(nn.Module):
    def __init__(self, K: int, D: int, N: int, init_std: float = 0.1, seed: int | None = None):
        super().__init__()
        self.K, self.D, self.N = K, D, N
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        # differentiated random cores (NOT near-identity) so symbols matter; small scale
        cores = torch.randn(N, D, K, D, generator=gen) * (init_std / math.sqrt(D))
        # mild identity backbone for conditioning/stability
        cores += torch.eye(D)[None, :, None, :] * 0.3
        self.cores = nn.Parameter(cores)               # [N, D, K, D]
        self.left = nn.Parameter(torch.randn(D, generator=gen) * 0.1 + 1.0 / math.sqrt(D))
        self.right = nn.Parameter(torch.randn(D, generator=gen) * 0.1 + 1.0 / math.sqrt(D))

    def log_abs_amplitude(self, z: Tensor) -> Tensor:
        """log|Psi(z)| for a batch of integer chains z [B, N]."""
        B = z.shape[0]
        h = self.left.expand(B, self.D).clone()         # [B, D]
        log_norm = torch.zeros(B, device=z.device, dtype=self.cores.dtype)
        for j in range(self.N):
            Aj = self.cores[j][:, z[:, j], :]            # [D, B, D] (gather over symbols)
            Aj = Aj.permute(1, 0, 2)                      # [B, D, D]
            h = torch.bmm(h.unsqueeze(1), Aj).squeeze(1)  # [B, D]
            nrm = h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            h = h / nrm
            log_norm = log_norm + nrm.squeeze(-1).log()
        psi = (h * self.right).sum(-1)                    # [B]
        return psi.abs().clamp_min(1e-20).log() + log_norm

    def log_Z(self) -> Tensor:
        """log of the partition function via the double-layer transfer matrices."""
        D = self.D
        L = torch.outer(self.left, self.left).reshape(D * D)   # [D^2]
        log_norm = torch.zeros((), device=self.cores.device, dtype=self.cores.dtype)
        for j in range(self.N):
            A = self.cores[j]                                  # [D, K, D]
            E = torch.einsum("akc,bkd->abcd", A, A).reshape(D * D, D * D)  # [D^2, D^2]
            L = L @ E
            nrm = L.norm().clamp_min(1e-20)
            L = L / nrm
            log_norm = log_norm + nrm.log()
        R = torch.outer(self.right, self.right).reshape(D * D)
        z = (L * R).sum().clamp_min(1e-20)
        return z.log() + log_norm

    def nll(self, z: Tensor) -> Tensor:
        """Mean negative log-likelihood -E[log P(z)] = -2 E[log|Psi|] + log Z."""
        return -2.0 * self.log_abs_amplitude(z).mean() + self.log_Z()

    @torch.no_grad()
    def _right_env(self, start: int) -> Tensor:
        """Right marginal environment R = (prod_{j>=start} E_j)(r (x) r), as a [D,D] matrix."""
        D = self.D
        R = torch.outer(self.right, self.right).reshape(D * D)
        for j in range(self.N - 1, start - 1, -1):
            A = self.cores[j]
            E = torch.einsum("akc,bkd->abcd", A, A).reshape(D * D, D * D)
            R = E @ R
        return R.reshape(D, D)

    @torch.no_grad()
    def conditional_next_logprob(self, z_obs: Tensor) -> Tensor:
        """P(z_{m+1}=k | z_{1:m}) for observed z_obs [B, m]; returns log-probs [B, K]."""
        B, m = z_obs.shape
        h = self.left.expand(B, self.D).clone()
        for j in range(m):
            Aj = self.cores[j][:, z_obs[:, j], :].permute(1, 0, 2)
            h = torch.bmm(h.unsqueeze(1), Aj).squeeze(1)
            h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        A_next = self.cores[m]                               # [D, K, D]
        v = torch.einsum("bd,dke->bke", h, A_next)           # [B, K, D]
        R = self._right_env(m + 1)                            # [D, D]
        score = torch.einsum("bke,ef,bkf->bk", v, R, v).clamp_min(1e-30)  # unnormalised P
        return score.log() - score.sum(-1, keepdim=True).log()

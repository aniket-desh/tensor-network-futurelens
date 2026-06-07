r"""Translation-invariant autoregressive MPS sequence model (Exp 06, mechanistic test).

Models the residual feature sequence itself (a Gaussian-emission MPS): from the
cumulative left environment over sites $v_{1:i}$, predict the next site $v_{i+1}$.
Trained with MSE on next-site prediction, in a FIXED feature basis (no learned phi,
no final-residual readout). Its transfer-matrix correlation lengths $\xi_\mu$ are then
directly comparable to the empirically measured residual correlations in the same basis
-- the clean "is the MPS using the transfer mechanism?" test.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .mps import (
    correlation_lengths_from_core,
    transfer_spectrum_from_core,
)


class MPSAutoregressive(nn.Module):
    """TI MPS next-site predictor. forward(v[B,T,p]) -> preds[B,T-1,p] (predict v[:,1:])."""

    def __init__(self, p: int, D: int, const_channel: bool = True, init_std: float = 1e-2,
                 seed: int | None = None):
        super().__init__()
        self.p = p
        self.D = D
        self.const_channel = const_channel
        self.p_eff = p + 1 if const_channel else p
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        # IMPORTANT: a near-identity init (every physical slice = I/sqrt(p)) makes
        # M_i ~ scalar*I, so the cumulative environment is input-insensitive and the
        # model cannot even fit AR(1). Use DIFFERENTIATED random slices so M_i genuinely
        # depends on which features are active; per-site normalisation handles scale.
        core = torch.randn(D, self.p_eff, D, generator=gen) / math.sqrt(self.p_eff)
        if const_channel:
            core[:, 0, :] = core[:, 0, :] + torch.eye(D) * 0.5  # mild identity on the const site
        self.core = nn.Parameter(core)
        self.head = nn.Linear(D * D, p)  # decode env -> next-site prediction

    def _site_matrices(self, v: Tensor) -> Tensor:
        if self.const_channel:
            v = torch.cat([v.new_ones(v.shape[0], v.shape[1], 1), v], dim=-1)
        return torch.einsum("dpe,btp->btde", self.core, v)  # [B,T,D,D]

    def forward(self, v: Tensor) -> Tensor:
        M = self._site_matrices(v)  # [B,T,D,D]
        B, T = M.shape[0], M.shape[1]
        H = torch.eye(self.D, device=v.device, dtype=v.dtype).expand(B, self.D, self.D).clone()
        preds = []
        for i in range(T):
            H = torch.bmm(H, M[:, i])
            nrm = H.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            H = H / nrm
            preds.append(self.head(H.reshape(B, -1)))  # predict v_{i+1} from env over v_{1:i+1}
        pred = torch.stack(preds, dim=1)  # [B,T,p], pred[:,i] predicts v[:,i+1]
        return pred[:, :-1]               # align to targets v[:,1:]

    def transfer_spectrum(self) -> Tensor:
        return transfer_spectrum_from_core(self.core)

    def correlation_lengths(self) -> Tensor:
        return correlation_lengths_from_core(self.core)

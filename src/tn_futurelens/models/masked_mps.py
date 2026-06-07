r"""Masked MPS completion (briefing B5 / §6, §8 Phase 5).

Closer to the theory than the B4 readout: build a single chain of ``m + n`` physical
sites. The first ``m`` are the observed features; the next ``n`` are LEARNED mask
vectors (one per future site). We contract left-to-right and decode the cumulative
environment at each future site into a predicted future residual:

    observed:  u_j = v_j            (j = 0..m-1)
    future:    u_j = mask_{j-m}     (j = m..m+n-1, learned)
    h_j        = (M_0 M_1 ... M_j)              # left environment up to site j
    r_hat_{t+s} = head_s(flatten h_{m+s})       # s = 0..n-1

This implements "condition on the first m sites, complete the next n" with a causal
left-to-right environment. Same external signature as ``MPSReadout`` ([B,m,p]->[B,n,d_out])
so it drops into ``PhiHead`` and the standard trainer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .mps import _near_identity_core


class MaskedMPSCompletion(nn.Module):
    def __init__(
        self,
        p: int,
        D: int,
        m: int,
        n: int,
        d_out: int,
        const_channel: bool = False,
        init_std: float = 1e-2,
        seed: int | None = None,
    ):
        super().__init__()
        self.p = p
        self.m = m
        self.n = n
        self.D = D
        self.d_out = d_out
        self.const_channel = const_channel
        self.p_eff = p + 1 if const_channel else p
        N = m + n
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        cores = torch.stack([_near_identity_core(D, self.p_eff, init_std, gen) for _ in range(N)])
        self.cores = nn.Parameter(cores)                       # [N, D, p_eff, D]
        self.mask = nn.Parameter(torch.randn(n, p, generator=gen) * 0.02)  # learned future sites
        self.heads = nn.ModuleList([nn.Linear(D * D, d_out) for _ in range(n)])

    def forward(self, v_obs: Tensor) -> Tensor:
        if v_obs.shape[1] != self.m or v_obs.shape[2] != self.p:
            raise ValueError(f"expected [B,{self.m},{self.p}], got {tuple(v_obs.shape)}")
        B = v_obs.shape[0]
        mask = self.mask.unsqueeze(0).expand(B, self.n, self.p)
        u = torch.cat([v_obs, mask], dim=1)                    # [B, m+n, p]
        if self.const_channel:
            u = torch.cat([u.new_ones(B, u.shape[1], 1), u], dim=-1)
        M = torch.einsum("ndpe,bnp->bnde", self.cores, u)      # [B, N, D, D]

        H = torch.eye(self.D, device=u.device, dtype=u.dtype).expand(B, self.D, self.D).clone()
        preds = []
        for j in range(self.m + self.n):
            H = torch.bmm(H, M[:, j])
            nrm = H.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            H = H / nrm
            if j >= self.m:                                    # future site s = j - m
                preds.append(self.heads[j - self.m](H.reshape(B, -1)))
        return torch.stack(preds, dim=1)                       # [B, n, d_out]

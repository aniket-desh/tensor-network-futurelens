r"""FutureLens-style baselines (briefing §6: B0-B3).

All take an observed window ``v`` of shape ``[B, m, p]`` and predict ``n`` future
residual vectors, output shape ``[B, n, d_out]``. These control for "more context
helps" (B1), "generic nonlinearity helps" (B2), and "learned position mixing helps"
(B3), so an MPS win can't be explained by those alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SingleSiteLinear(nn.Module):
    """B0: linear map from the LAST observed site r_t only (FutureLens baseline)."""

    def __init__(self, p: int, d_out: int, n_horizons: int):
        super().__init__()
        self.n_horizons = n_horizons
        self.d_out = d_out
        self.head = nn.Linear(p, n_horizons * d_out)

    def forward(self, v: Tensor) -> Tensor:
        last = v[:, -1, :]                       # [B, p]
        out = self.head(last)
        return out.reshape(out.shape[0], self.n_horizons, self.d_out)


class MultiSiteLinear(nn.Module):
    """B1: linear map from the concatenated observed window."""

    def __init__(self, p: int, m: int, d_out: int, n_horizons: int):
        super().__init__()
        self.n_horizons = n_horizons
        self.d_out = d_out
        self.head = nn.Linear(m * p, n_horizons * d_out)

    def forward(self, v: Tensor) -> Tensor:
        flat = v.reshape(v.shape[0], -1)         # [B, m*p]
        out = self.head(flat)
        return out.reshape(out.shape[0], self.n_horizons, self.d_out)


class MultiSiteMLP(nn.Module):
    """B2: MLP over the concatenated observed window (parameter-match to MPS)."""

    def __init__(
        self, p: int, m: int, d_out: int, n_horizons: int, hidden: int = 256, depth: int = 2
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.d_out = d_out
        layers: list[nn.Module] = [nn.Linear(m * p, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, n_horizons * d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, v: Tensor) -> Tensor:
        flat = v.reshape(v.shape[0], -1)
        out = self.net(flat)
        return out.reshape(out.shape[0], self.n_horizons, self.d_out)


class AttentionPool(nn.Module):
    """B3: tiny self-attention over observed sites + learned query pooling."""

    def __init__(
        self, p: int, m: int, d_out: int, n_horizons: int, d_model: int = 128, n_heads: int = 4
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.d_out = d_out
        self.proj = nn.Linear(p, d_model)
        self.pos = nn.Parameter(torch.randn(1, m, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, n_horizons, d_model) * 0.02)
        self.head = nn.Linear(d_model, d_out)

    def forward(self, v: Tensor) -> Tensor:
        B = v.shape[0]
        x = self.proj(v) + self.pos                       # [B, m, d_model]
        x, _ = self.attn(x, x, x)                         # self-attend observed sites
        q = self.query.expand(B, -1, -1)                  # [B, n, d_model]
        pooled, _ = self.attn(q, x, x)                    # pool per horizon
        return self.head(pooled)                          # [B, n, d_out]


def build_baseline(kind: str, p: int, m: int, d_out: int, n_horizons: int, **kw) -> nn.Module:
    kind = kind.lower()
    if kind in ("b0", "single_linear", "single"):
        return SingleSiteLinear(p, d_out, n_horizons)
    if kind in ("b1", "multisite_linear", "multi"):
        return MultiSiteLinear(p, m, d_out, n_horizons)
    if kind in ("b2", "mlp"):
        return MultiSiteMLP(p, m, d_out, n_horizons, **kw)
    if kind in ("b3", "attention", "attn"):
        return AttentionPool(p, m, d_out, n_horizons, **kw)
    raise ValueError(f"unknown baseline {kind!r}")

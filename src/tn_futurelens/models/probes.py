r"""Compose a feature map phi with a per-window head (baseline or MPS).

The head sees the phi-mapped observed window. Keeping phi *inside* the module lets it
be either frozen (PCA) or learned (LearnedLinearPhi init from PCA) and trained jointly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PhiHead(nn.Module):
    """Apply ``phi`` per site to raw residuals ``[B, m, d_model]`` then run ``head``.

    ``head`` consumes ``[B, m, p]`` (e.g. MultiSiteLinear / MultiSiteMLP / MPSReadout)
    and returns ``[B, n_horizons, d_out]``.
    """

    def __init__(self, phi: nn.Module, head: nn.Module):
        super().__init__()
        self.phi = phi
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        B, m, d = x.shape
        v = self.phi(x.reshape(B * m, d)).reshape(B, m, -1)  # [B, m, p]
        return self.head(v)

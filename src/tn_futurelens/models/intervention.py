r"""TN-parameterized causal-intervention FutureLens (Exp 12, GPT-J scale).

FutureLens's *strong* method is a causal intervention, not a readout: transplant a donor
hidden state into a learned soft-prompt context of the frozen model and read its output.
Here the donor is produced from the observed residual *trajectory* by a learnable map
(MPS / MLP / linear / single-state), and a per-horizon learned soft prompt elicits the
token s steps ahead. This tests, at GPT-J scale, whether (a) using the trajectory via a
causal intervention beats a single-state / a readout, and (b) the TN parameterization of
the map beats generic ones.

GPT-J uses rotary position embeddings (positions handled in attention, not added to the
residual), so overriding ``resid_pre`` at layer 0 with the soft prompt is clean.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .mps import MPSReadout
from .phi import LearnedLinearPhi, PCAPhi


class TrajectoryDonorMap(nn.Module):
    """Map an observed residual trajectory ``[B, m, d_model]`` -> donor ``[B, d_model]``.

    kind: 'single' (last site only, ~FutureLens m=1), 'linear', 'mlp', 'mps'.
    For 'mps' a learned phi (init from PCA) reduces d_model->p before the MPS.
    """

    def __init__(self, kind: str, d_model: int, m: int, *, p: int = 64, D: int = 16,
                 hidden: int = 512, pca: PCAPhi | None = None, seed: int = 0):
        super().__init__()
        self.kind = kind
        self.d_model = d_model
        self.m = m
        if kind == "single":
            self.head = nn.Linear(d_model, d_model)
        elif kind == "linear":
            self.head = nn.Linear(m * d_model, d_model)
        elif kind == "mlp":
            self.head = nn.Sequential(nn.Linear(m * d_model, hidden), nn.GELU(),
                                      nn.Linear(hidden, d_model))
        elif kind == "mps":
            self.phi = LearnedLinearPhi(d_model, p)
            if pca is not None:
                self.phi.init_from_pca(pca)
            self.mps = MPSReadout(p=p, D=D, n_sites=m, readout="env", out_dim=d_model,
                                  n_heads=1, const_channel=True, seed=seed)
        else:
            raise ValueError(kind)

    def forward(self, traj: Tensor) -> Tensor:
        B, m, d = traj.shape
        if self.kind == "single":
            return self.head(traj[:, -1, :])
        if self.kind in ("linear", "mlp"):
            return self.head(traj.reshape(B, -1))
        v = self.phi(traj.reshape(B * m, d)).reshape(B, m, -1)
        return self.mps(v)[:, 0, :]  # [B, d_model]


class SoftPromptIntervention(nn.Module):
    """Learned soft prompts (one per horizon) + the donor map; runs the frozen model.

    forward returns, per horizon s, the logits at the donor read-position -> a predicted
    distribution for the token s steps after the observed window.
    """

    def __init__(self, donor_map: TrajectoryDonorMap, d_model: int, n_horizons: int,
                 prompt_len: int = 6, insert_layer: int = 14):
        super().__init__()
        self.donor_map = donor_map
        self.P = prompt_len
        self.ell = insert_layer
        self.n_horizons = n_horizons
        # one learned soft prompt per horizon (continuous resid_pre[0] vectors)
        self.soft_prompts = nn.Parameter(torch.randn(n_horizons, prompt_len, d_model) * 0.02)

    def _read_logits(self, model, donor: Tensor, sp: Tensor) -> Tensor:
        """Run frozen model with soft prompt sp [P,d] + donor insertion; return logits[:, -1]."""
        B = donor.shape[0]
        P, ell = self.P, self.ell
        tokens = torch.zeros(B, P, dtype=torch.long, device=donor.device)

        def hook_prompt(resid, hook):
            return sp.unsqueeze(0).expand(B, P, -1).to(resid.dtype)

        def hook_inject(resid, hook):
            resid = resid.clone()
            resid[:, P - 1, :] = donor.to(resid.dtype)
            return resid

        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[("blocks.0.hook_resid_pre", hook_prompt),
                       (f"blocks.{ell}.hook_resid_pre", hook_inject)],
            return_type="logits",
        )
        return logits[:, P - 1, :]  # [B, vocab]

    def forward(self, model, traj: Tensor) -> list[Tensor]:
        donor = self.donor_map(traj)
        return [self._read_logits(model, donor, self.soft_prompts[s]) for s in range(self.n_horizons)]

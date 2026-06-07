r"""Reusable training loop for regression probes (baselines + MPS).

Probes map an observed window ``X`` ``[N, m, p]`` to future targets ``Y``
``[N, n, d_out]``. Used for both synthetic validation and the GPT-2 experiments.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .losses import cosine_similarity, residual_mse, residual_nmse


@dataclass
class TrainResult:
    best_val_nmse: float
    best_val_cos: float
    best_epoch: int
    history: list[dict] = field(default_factory=list)
    n_params: int = 0


def _iter_minibatches(n: int, batch_size: int, generator: torch.Generator):
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        yield perm[i : i + batch_size]


def train_regression_probe(
    model: torch.nn.Module,
    X_train: Tensor,
    Y_train: Tensor,
    X_val: Tensor,
    Y_val: Tensor,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 4096,
    device: str | torch.device = "cpu",
    log_every: int = 0,
    patience: int | None = 30,
    seed: int = 0,
) -> TrainResult:
    """Train ``model`` to regress ``Y`` from ``X`` with Adam; early-stop on val NMSE."""
    gen = torch.Generator().manual_seed(seed)
    model = model.to(device)
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    best = TrainResult(best_val_nmse=float("inf"), best_val_cos=-1.0, best_epoch=-1,
                       n_params=n_params)
    best_state = copy.deepcopy(model.state_dict())
    since_improve = 0

    for epoch in range(epochs):
        model.train()
        for idx in _iter_minibatches(X_train.shape[0], batch_size, gen):
            opt.zero_grad()
            pred = model(X_train[idx])
            loss = residual_mse(pred, Y_train[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vp = model(X_val)
            v_nmse = residual_nmse(vp, Y_val).item()
            v_cos = cosine_similarity(vp, Y_val).item()
        best.history.append({"epoch": epoch, "val_nmse": v_nmse, "val_cos": v_cos})
        if log_every and epoch % log_every == 0:
            print(f"  epoch {epoch:4d}  val_nmse={v_nmse:.4f}  val_cos={v_cos:.4f}")

        if v_nmse < best.best_val_nmse - 1e-5:
            best.best_val_nmse = v_nmse
            best.best_val_cos = v_cos
            best.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            since_improve = 0
        else:
            since_improve += 1
            if patience is not None and since_improve >= patience:
                break

    model.load_state_dict(best_state)  # restore best
    return best

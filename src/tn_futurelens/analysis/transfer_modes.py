r"""Learned-MPS transfer-matrix diagnostics (briefing §6).

After a translation-invariant MPS is trained, its transfer matrix
``E = sum_a A^a (x) conj(A^a)`` (D^2 x D^2) encodes the correlation structure the
model actually learned. The subleading eigenvalues give correlation lengths
``xi_mu = -1/ln|lambda_mu/lambda_1|`` which we compare to the EMPIRICAL residual
correlation lengths -- the key check that the model uses the transfer mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..models.mps import MPSReadout


@dataclass
class TransferReport:
    eigenvalues: np.ndarray         # complex, sorted by |.| desc, length D^2
    magnitudes: np.ndarray          # |lambda|
    correlation_lengths: np.ndarray # xi_mu for mu>=2 (length D^2-1)
    leading: complex                # lambda_1 (should be ~ real, dominant)
    spectral_gap: float             # 1 - |lambda_2/lambda_1|


@torch.no_grad()
def mps_transfer_report(mps: MPSReadout) -> TransferReport:
    lam = mps.transfer_spectrum().detach().cpu().numpy()
    mags = np.abs(lam)
    lam1 = mags[0] if mags.size else 1.0
    xi = mps.correlation_lengths().detach().cpu().numpy()
    gap = float(1.0 - mags[1] / lam1) if mags.size > 1 else float("nan")
    return TransferReport(
        eigenvalues=lam,
        magnitudes=mags,
        correlation_lengths=xi,
        leading=complex(lam[0]) if lam.size else 0j,
        spectral_gap=gap,
    )


def dominant_correlation_length(mps: MPSReadout) -> float:
    """Longest correlation length implied by the learned transfer matrix."""
    rep = mps_transfer_report(mps)
    finite = rep.correlation_lengths[np.isfinite(rep.correlation_lengths)]
    return float(finite.max()) if finite.size else float("nan")

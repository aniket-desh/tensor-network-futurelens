r"""Eigensystem Realization (Ho-Kalman) of a matrix correlation sequence.

The clean, confound-free version of "what transfer spectrum represents the measured
residual correlations". Given the empirical matrix two-point function
``C(1), C(2), ..., C(L)`` (in a FIXED basis), a linear-Gaussian MPS / state-space
model that reproduces it is ``C(Delta) = H A^{Delta-1} G``. We recover it directly:

  * block-Hankel SVD -> singular spectrum; its numerical rank = number of modes
    (the state dimension), which an MPS represents with bond ``D`` s.t. ``D^2-1 >= rank``.
  * the realized state matrix ``A`` has eigenvalues = the correlation decay modes
    ``lambda_mu``; ``xi_mu = -1/ln|lambda_mu|`` are the correlation lengths, in the same
    basis the correlations were measured in.

This is exact for a finite sum of (matrix) exponentials and robust to moderate noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Realization:
    singular_values: np.ndarray  # block-Hankel singular values (mode strengths)
    rank: int                    # chosen state dimension (# modes)
    eigenvalues: np.ndarray      # decay modes lambda_mu (complex)
    xis: np.ndarray              # -1/ln|lambda_mu|, sorted desc by |lambda|


def _block_hankel(M: np.ndarray, r: int, shift: int) -> np.ndarray:
    """Block Hankel with blocks ``M[i+j+shift]`` for i,j in [0,r). M: [L, p, p]."""
    p = M.shape[1]
    H = np.empty((r * p, r * p))
    for i in range(r):
        for j in range(r):
            H[i * p:(i + 1) * p, j * p:(j + 1) * p] = M[i + j + shift]
    return H


def ho_kalman(
    C: np.ndarray, rank: int | None = None, rel_threshold: float = 0.02, max_rank: int = 24
) -> Realization:
    """Realize ``C(Delta)=H A^{Delta-1} G`` from ``C`` ``[L+1, p, p]`` (uses C[1:]).

    ``rank`` (state dim) chosen from the block-Hankel singular spectrum if not given
    (count of singular values above ``rel_threshold * sigma_max``, capped at ``max_rank``).
    """
    C = np.asarray(C, dtype=np.float64)
    M = C[1:]                                   # impulse response C(1..L)
    L, p, _ = M.shape
    r = L // 2                                  # block rows/cols; needs M up to 2r-1
    if r < 2:
        raise ValueError("need C up to at least Delta=4 for a realization")
    H0 = _block_hankel(M, r, shift=0)           # blocks C(1..2r-1)
    H1 = _block_hankel(M, r, shift=1)           # shifted
    U, S, Vt = np.linalg.svd(H0)
    if rank is None:
        rank = int(np.sum(S > rel_threshold * S[0]))
        rank = max(1, min(rank, max_rank, S.size))
    Un, Sn, Vtn = U[:, :rank], S[:rank], Vt[:rank]
    sq = np.sqrt(Sn)
    O = Un * sq[None, :]                         # observability  [rp, n]
    Cont = sq[:, None] * Vtn                     # controllability [n, rp]
    A = np.linalg.pinv(O) @ H1 @ np.linalg.pinv(Cont)   # [n, n]
    eig = np.linalg.eigvals(A)
    order = np.argsort(np.abs(eig))[::-1]
    eig = eig[order]
    mag = np.abs(eig).clip(1e-12, 1 - 1e-12)
    xis = -1.0 / np.log(mag)
    return Realization(singular_values=S, rank=rank, eigenvalues=eig, xis=xis)

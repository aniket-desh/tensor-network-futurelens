r"""Matrix Product State (MPS / tensor-train) probe layers.

The local core has a physical leg ``p`` and two bond legs ``D``:

    A_j^a in R^{D x D}  (one matrix per physical index a = 1..p),
    A_j(v_j) = sum_a v_{j,a} A_j^a.

We contract a length-``N`` chain of per-site feature vectors ``v_j in R^p`` into a
hidden representation, with per-site normalisation + a tracked log-scale for
numerical stability (briefing §9.2), and near-identity initialisation so the
transfer matrix has spectral radius ~1 at init.

Two variants (briefing §9.1):
  * non-translation-invariant: a separate core per site (better for finite windows).
  * translation-invariant (TI): one shared core -> the transfer-matrix spectrum is
    well-defined and is the bridge to the correlation-length theory (briefing §4, §6).

Readout modes (briefing §9.3):
  * ``"vector"``: contract a left boundary vector -> D-dim hidden.
  * ``"env"``:    carry the full D x D environment (identity start) -> D^2-dim hidden
                  (better matches transfer-matrix theory).
  * ``"scalar"``: full MPS scalar amplitude l^T A(v_1)...A(v_N) r (for Born-style scores).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

_READOUTS = ("vector", "env", "scalar")


def _near_identity_core(D: int, p: int, init_std: float, generator=None) -> Tensor:
    """Core ``[D, p, D]`` with each physical slice ~ I/sqrt(p) + noise.

    Then ``E = sum_a A^a (x) A^a ~ I (x) I`` so the chain neither explodes nor
    vanishes at init (spectral radius ~1).
    """
    base = torch.eye(D).unsqueeze(1).expand(D, p, D).clone() / math.sqrt(p)
    noise = torch.randn(D, p, D, generator=generator) * init_std
    return base + noise


class MPSReadout(nn.Module):
    """MPS contraction over ``n_sites`` feature vectors -> hidden (and optional heads).

    Args:
        p: physical (feature) dimension per site.
        D: bond dimension.
        n_sites: chain length N.
        translation_invariant: share one core across all sites if True.
        readout: one of {"vector", "env", "scalar"}.
        out_dim: if given, attach a linear head mapping hidden -> ``n_heads * out_dim``
            and return shape ``[B, n_heads, out_dim]``. If None, return the raw hidden.
        n_heads: number of output heads (e.g. one per future horizon s=1..n).
        init_std: std of the additive noise on the near-identity cores.
        const_channel: if True, prepend a constant-1 feature so the local map becomes
            ``A_j(v_j) = A_j^0 + sum_a v_{j,a} A_j^a``. This lets the MPS represent
            constants, single-site linear terms, and all higher-order interactions in
            one architecture -- without it the contraction is a pure multilinear product
            of the inputs (no additive/linear component), which biases it against tasks
            with a large linear part.
        seed: optional seed for reproducible init.

    Forward input:  ``v`` of shape ``[B, n_sites, p]``.
    Forward output: hidden ``[B, hidden_dim]`` (out_dim None) or ``[B, n_heads, out_dim]``.
    """

    def __init__(
        self,
        p: int,
        D: int,
        n_sites: int,
        translation_invariant: bool = False,
        readout: str = "vector",
        out_dim: int | None = None,
        n_heads: int = 1,
        init_std: float = 1e-2,
        const_channel: bool = False,
        seed: int | None = None,
    ):
        super().__init__()
        if readout not in _READOUTS:
            raise ValueError(f"readout must be one of {_READOUTS}, got {readout!r}")
        self.p = p                                  # input feature dim
        self.const_channel = const_channel
        self.p_eff = p + 1 if const_channel else p  # physical dim of the cores
        self.D = D
        self.n_sites = n_sites
        self.translation_invariant = translation_invariant
        self.readout = readout
        self.n_heads = n_heads

        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)

        if translation_invariant:
            core = _near_identity_core(D, self.p_eff, init_std, gen)  # [D, p_eff, D]
            self.core = nn.Parameter(core)
        else:
            cores = torch.stack(
                [_near_identity_core(D, self.p_eff, init_std, gen) for _ in range(n_sites)]
            )  # [N, D, p_eff, D]
            self.cores = nn.Parameter(cores)

        # boundary vectors only needed for vector/scalar readouts
        if readout in ("vector", "scalar"):
            self.left_boundary = nn.Parameter(torch.ones(D) / math.sqrt(D))
        if readout == "scalar":
            self.right_boundary = nn.Parameter(torch.ones(D) / math.sqrt(D))

        self.hidden_dim = {"vector": D, "env": D * D, "scalar": 1}[readout]

        self.head: nn.Linear | None = None
        if out_dim is not None:
            self.head = nn.Linear(self.hidden_dim, n_heads * out_dim)
        self.out_dim = out_dim

    # -- internals -----------------------------------------------------------
    def _site_matrices(self, v: Tensor) -> Tensor:
        """Per-site transfer matrices ``M_j(v_j) = sum_a v_{j,a} A_j^a``.

        Input ``v`` ``[B, N, p]`` -> output ``[B, N, D, D]``.
        """
        if v.ndim != 3 or v.shape[1] != self.n_sites or v.shape[2] != self.p:
            raise ValueError(
                f"expected v of shape [B, {self.n_sites}, {self.p}], got {tuple(v.shape)}"
            )
        if self.const_channel:
            ones = v.new_ones(v.shape[0], v.shape[1], 1)
            v = torch.cat([ones, v], dim=-1)  # [B, N, p+1]
        if self.translation_invariant:
            # same core at every site: [D, p, D] x [B, N, p] -> [B, N, D, D]
            return torch.einsum("dpe,bnp->bnde", self.core, v)
        # per-site cores: [N, D, p, D] x [B, N, p] -> [B, N, D, D]
        return torch.einsum("ndpe,bnp->bnde", self.cores, v)

    def contract(self, v: Tensor) -> tuple[Tensor, Tensor]:
        """Contract the chain. Returns (hidden ``[B, hidden_dim]``, log_norm ``[B]``)."""
        M = self._site_matrices(v)  # [B, N, D, D]
        B = M.shape[0]
        D = self.D

        if self.readout == "env":
            H = torch.eye(D, device=M.device, dtype=M.dtype).expand(B, D, D).clone()
        else:  # vector / scalar start from the left boundary row [B, 1, D]
            H = self.left_boundary.to(M.dtype).expand(B, 1, D).clone()

        log_norm = torch.zeros(B, device=M.device, dtype=M.dtype)
        for j in range(self.n_sites):
            H = torch.bmm(H, M[:, j])  # [B, R, D] @ [B, D, D] -> [B, R, D]
            nrm = H.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            H = H / nrm
            log_norm = log_norm + nrm.reshape(B).log()

        if self.readout == "scalar":
            r = self.right_boundary.to(M.dtype).reshape(1, D, 1).expand(B, D, 1)
            hidden = torch.bmm(H, r).reshape(B, 1)  # [B, 1]
        else:
            hidden = H.reshape(B, -1)  # [B, D] (vector) or [B, D*D] (env)
        return hidden, log_norm

    def forward(self, v: Tensor) -> Tensor:
        hidden, _ = self.contract(v)
        if self.head is None:
            return hidden
        out = self.head(hidden)  # [B, n_heads * out_dim]
        return out.reshape(out.shape[0], self.n_heads, self.out_dim)

    # -- transfer-matrix diagnostics (TI only) -------------------------------
    def transfer_matrix(self) -> Tensor:
        r"""Transfer matrix ``E = sum_a A^a (x) conj(A^a)`` of shape ``[D^2, D^2]``.

        Row index is ``(a, b)`` (both "bra" copies), column ``(c, d)``. TI only.
        """
        if not self.translation_invariant:
            raise RuntimeError(
                "transfer_matrix is only defined for a translation-invariant MPS"
            )
        A = self.core.detach()  # [D, p, D] with legs (left, phys, right); diagnostics only
        E = torch.einsum("apc,bpd->abcd", A, A.conj())  # [D, D, D, D]
        return E.reshape(self.D * self.D, self.D * self.D)

    def transfer_spectrum(self) -> Tensor:
        """Eigenvalues of the transfer matrix, sorted by descending magnitude (complex)."""
        E = self.transfer_matrix()
        lam = torch.linalg.eigvals(E)
        return lam[lam.abs().argsort(descending=True)]

    def correlation_lengths(self, eps: float = 1e-12) -> Tensor:
        r"""Correlation lengths ``xi_mu = -1/ln|lambda_mu / lambda_1|`` for mu >= 2.

        The leading eigenvalue (disconnected part) is divided out; the remaining
        ``D^2 - 1`` values are the exponential modes of the connected correlator.
        Returned sorted by descending |lambda| (i.e. longest xi first).
        """
        lam = self.transfer_spectrum()
        lam1 = lam[0].abs().clamp_min(eps)
        ratios = (lam[1:].abs() / lam1).clamp_min(eps)
        return -1.0 / torch.log(ratios.clamp_max(1.0 - 1e-12))

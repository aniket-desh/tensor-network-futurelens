"""Transfer-matrix shape + known-spectrum recovery tests (briefing §4, §13.6)."""

import numpy as np
import pytest
import torch

from tn_futurelens.models.mps import MPSReadout


def test_transfer_matrix_shape():
    D = 5
    mps = MPSReadout(p=3, D=D, n_sites=4, translation_invariant=True, seed=0)
    E = mps.transfer_matrix()
    assert E.shape == (D * D, D * D)


def test_non_ti_transfer_matrix_raises():
    mps = MPSReadout(p=3, D=4, n_sites=4, translation_invariant=False, seed=0)
    with pytest.raises(RuntimeError):
        mps.transfer_matrix()


def test_diagonal_core_spectrum():
    r"""For p=1 and A = diag(a, b), E = A (x) A has eigenvalues {a^2, ab, ab, b^2}."""
    a, b = 0.9, 0.5
    mps = MPSReadout(p=1, D=2, n_sites=2, translation_invariant=True, seed=0)
    with torch.no_grad():
        core = torch.zeros(2, 1, 2)
        core[0, 0, 0] = a
        core[1, 0, 1] = b
        mps.core.copy_(core)
    mags = mps.transfer_spectrum().abs().numpy()
    expected = np.array(sorted([a * a, a * b, a * b, b * b], reverse=True))
    np.testing.assert_allclose(np.sort(mags)[::-1], expected, atol=1e-5)


def test_correlation_length_recovery():
    r"""Leading lambda divided out; subleading ratio gives xi = -1/ln|lambda2/lambda1|."""
    a, b = 0.9, 0.5
    mps = MPSReadout(p=1, D=2, n_sites=2, translation_invariant=True, seed=0)
    with torch.no_grad():
        core = torch.zeros(2, 1, 2)
        core[0, 0, 0] = a
        core[1, 0, 1] = b
        mps.core.copy_(core)
    xi = mps.correlation_lengths().numpy()
    # lambda1 = a^2 = 0.81, next |lambda| = ab = 0.45 -> xi = -1/ln(0.45/0.81)
    expected_top = -1.0 / np.log((a * b) / (a * a))
    assert np.isclose(xi.max(), expected_top, atol=1e-4)

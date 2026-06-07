"""Ho-Kalman / ERA realization of a matrix correlation sequence."""
import numpy as np
from tn_futurelens.analysis.realization import ho_kalman


def test_ar1_single_mode():
    rho = 0.85
    C = np.stack([(rho ** d) * np.eye(4) for d in range(31)])
    r = ho_kalman(C, rel_threshold=0.02)
    assert np.isclose(np.abs(r.eigenvalues[0]), rho, atol=1e-3)
    assert np.isclose(np.sort(r.xis)[::-1][0], -1 / np.log(rho), atol=0.05)


def test_multi_mode_recovery():
    rhos = [0.5, 0.8, 0.95]
    C = np.stack([np.diag([rr ** d for rr in rhos]) for d in range(31)])
    r = ho_kalman(C, rel_threshold=0.02)
    assert r.rank == 3
    got = np.sort(np.abs(r.eigenvalues))[::-1]
    np.testing.assert_allclose(got, [0.95, 0.8, 0.5], atol=1e-3)

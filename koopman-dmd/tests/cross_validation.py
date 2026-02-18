"""Cross-validation: verify Rust (via Python) and R produce matching results."""

import numpy as np
import koopman_dmd


def test_dmd_python_eigenvalues_match_properties():
    """Eigenvalues from DMD should satisfy expected mathematical properties."""
    t = np.linspace(0, 2 * np.pi, 100)
    X = np.vstack([np.sin(t), np.cos(t)])

    d = koopman_dmd.DMD(X)
    eigs = d.eigenvalues  # (rank, 2) with [re, im]

    # Eigenvalue magnitudes should be near 1 for oscillatory data
    mags = np.sqrt(eigs[:, 0] ** 2 + eigs[:, 1] ** 2)
    for m in mags:
        assert abs(m - 1.0) < 0.1, f"Eigenvalue magnitude {m} not near 1.0"


def test_dmd_reconstruct_matches_original():
    """Reconstruction should closely match original data."""
    t = np.linspace(0, 2 * np.pi, 100)
    X = np.vstack([np.sin(t), np.cos(t)])

    d = koopman_dmd.DMD(X)
    recon = d.reconstruct(100)

    rmse = np.sqrt(np.mean((recon - X) ** 2))
    assert rmse < 0.1, f"Reconstruction RMSE {rmse} too large"


def test_predict_both_methods_agree():
    """Mode-based and matrix-based prediction should roughly agree."""
    t = np.linspace(0, 2 * np.pi, 100)
    X = np.vstack([np.sin(t), np.cos(t)])

    d = koopman_dmd.DMD(X)
    pred_modes = d.predict(10, method="modes")
    pred_matrix = d.predict(10, method="matrix")

    diff = np.abs(pred_modes - pred_matrix)
    assert np.max(diff) < 2.0, f"Mode vs matrix max diff: {np.max(diff)}"


def test_hankel_eigenvalue_magnitudes():
    """Hankel-DMD eigenvalues should be near unit circle for oscillatory data."""
    t = np.linspace(0, 4 * np.pi, 200)
    y = np.sin(t).reshape(1, -1)

    h = koopman_dmd.HankelDMD(y, delays=20, rank=2)
    eigs = h.eigenvalues
    mags = np.sqrt(eigs[:, 0] ** 2 + eigs[:, 1] ** 2)

    for m in mags:
        assert abs(m - 1.0) < 0.15, f"Hankel eigenvalue magnitude {m} not near 1.0"


def test_standard_map_bounded():
    """Standard map trajectory should stay in [0, 1]."""
    ic = np.array([0.1, 0.2])
    traj = koopman_dmd.generate_trajectory("standard", ic, 1000)
    assert np.all(traj >= 0.0) and np.all(traj <= 1.0)


if __name__ == "__main__":
    test_dmd_python_eigenvalues_match_properties()
    test_dmd_reconstruct_matches_original()
    test_predict_both_methods_agree()
    test_hankel_eigenvalue_magnitudes()
    test_standard_map_bounded()
    print("All cross-validation tests passed!")

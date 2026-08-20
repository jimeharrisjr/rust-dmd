"""Tests for koopman-dmd Python bindings."""

import numpy as np
import pytest

import koopman_dmd


# ============================================================================
# DMD tests
# ============================================================================


class TestDMD:
    """Test the DMD class."""

    @staticmethod
    def _make_signal():
        """Create a simple 2-variable oscillating signal."""
        t = np.linspace(0, 2 * np.pi, 100)
        x = np.vstack([np.sin(t), np.cos(t)])  # (2, 100)
        return x

    def test_basic(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        assert d.rank > 0
        assert d.data_dim == (2, 100)
        assert d.dt == 1.0
        assert not d.center

    def test_eigenvalues_shape(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        eigs = d.eigenvalues
        assert eigs.ndim == 2
        assert eigs.shape[1] == 2  # re, im columns
        assert eigs.shape[0] == d.rank

    def test_modes_shape(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        modes = d.modes
        assert modes.ndim == 2
        assert modes.shape[0] == 2  # n_vars
        assert modes.shape[1] == d.rank * 2  # re/im alternating

    def test_amplitudes_shape(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        amps = d.amplitudes
        assert amps.shape == (d.rank, 2)

    def test_singular_values(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        sv = d.singular_values
        assert sv.ndim == 1
        assert len(sv) == d.rank
        # Singular values should be positive and sorted descending
        assert all(sv[i] >= sv[i + 1] for i in range(len(sv) - 1))

    def test_predict(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        pred = d.predict(10)
        assert pred.shape[0] == 2  # n_vars
        assert pred.shape[1] == 10

    def test_predict_with_x0(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        x0 = np.array([0.0, 1.0])
        pred = d.predict(5, x0=x0)
        assert pred.shape == (2, 5)

    def test_predict_matrix(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        pred = d.predict(5, method="matrix")
        assert pred.shape[0] == 2

    def test_reconstruct(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        recon = d.reconstruct(100)
        assert recon.shape == (2, 100)
        # Reconstruction should be close to original
        rmse = np.sqrt(np.mean((recon - x) ** 2))
        assert rmse < 0.5

    def test_spectrum(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        spec = d.spectrum()
        assert isinstance(spec, list)
        assert len(spec) == d.rank
        for mode in spec:
            assert "magnitude" in mode
            assert "frequency" in mode
            assert "stability" in mode

    def test_stability(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        is_stable, is_unstable, is_marginal, spectral_radius = d.stability()
        assert isinstance(is_stable, bool)
        assert isinstance(spectral_radius, float)
        assert spectral_radius > 0

    def test_error(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        rmse, mae, mape, rel_err = d.error()
        assert rmse >= 0
        assert mae >= 0
        assert rel_err >= 0

    def test_dominant_modes(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        dom = d.dominant_modes(1)
        assert len(dom) == 1
        assert dom[0] < d.rank

    def test_residual(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        res_norm, res_rel = d.residual()
        assert res_norm >= 0
        assert res_rel >= 0

    def test_repr(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x)
        r = repr(d)
        assert "DMD" in r
        assert "rank=" in r

    def test_with_rank(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x, rank=1)
        assert d.rank == 1

    def test_with_centering(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x, center=True)
        assert d.center

    def test_with_dt(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x, dt=0.5)
        assert d.dt == 0.5

    def test_with_lifting(self):
        x = self._make_signal()
        d = koopman_dmd.DMD(x, lifting="polynomial", lifting_param=2)
        assert d.rank > 0

    def test_invalid_lifting(self):
        x = self._make_signal()
        with pytest.raises(ValueError, match="unknown lifting"):
            koopman_dmd.DMD(x, lifting="invalid")


# ============================================================================
# DMDc tests
# ============================================================================


class TestDMDc:
    """Test the DMDc class."""

    A0 = np.array([[0.9, 0.1], [0.0, 0.8]])
    B0 = np.array([[0.5], [1.0]])

    @classmethod
    def _make_forced(cls, m=120):
        """Simulate x_{t+1} = A0 x_t + B0 u_t with a persistently exciting input."""
        x1 = np.zeros((2, m))
        x2 = np.zeros((2, m))
        u = np.zeros((1, m))
        x = np.array([1.0, -0.5])
        for t in range(m):
            ut = np.sin(0.7 * t) + 0.5 * np.cos(2.3 * t + 1.0)
            x1[:, t] = x
            u[0, t] = ut
            x = cls.A0 @ x + cls.B0[:, 0] * ut
            x2[:, t] = x
        return x1, x2, u

    def test_recovers_a_and_b(self):
        x1, x2, u = self._make_forced()
        d = koopman_dmd.DMDc(x1, x2, u, rank_input=3)
        np.testing.assert_allclose(d.a, self.A0, atol=1e-9)
        np.testing.assert_allclose(d.b, self.B0, atol=1e-9)
        assert d.rank_input == 3
        assert d.rank_output == 2
        assert d.n_states == 2
        assert d.n_inputs == 1
        # Full-order default: basis is the identity and A~ = A.
        np.testing.assert_allclose(d.basis, np.eye(2), atol=0)
        np.testing.assert_allclose(d.a_tilde, d.a, atol=0)

    def test_eigenvalues(self):
        x1, x2, u = self._make_forced()
        d = koopman_dmd.DMDc(x1, x2, u, rank_input=3)
        eigs = d.eigenvalues
        assert eigs.shape == (2, 2)
        mags = np.sort(np.hypot(eigs[:, 0], eigs[:, 1]))
        np.testing.assert_allclose(mags, [0.8, 0.9], atol=1e-9)

    def test_known_b(self):
        x1, x2, u = self._make_forced()
        d = koopman_dmd.DMDc(x1, x2, u, rank_input=2, known_b=self.B0)
        np.testing.assert_allclose(d.a, self.A0, atol=1e-9)
        np.testing.assert_allclose(d.b, self.B0, atol=0)

    def test_autonomous_pairs(self):
        # u=None: multi-trajectory autonomous identification from pairs.
        a0 = np.array([[0.95, 0.02], [0.0, 0.85]])
        cols = []
        for start in ([1.0, 0.5], [-0.3, 1.2]):
            x = np.array(start)
            for _ in range(40):
                nxt = a0 @ x
                cols.append((x.copy(), nxt.copy()))
                x = nxt
        x1 = np.column_stack([c[0] for c in cols])
        x2 = np.column_stack([c[1] for c in cols])
        d = koopman_dmd.DMDc(x1, x2, rank_input=2)
        np.testing.assert_allclose(d.a, a0, atol=1e-9)
        assert d.n_inputs == 0

    def test_output_projection(self):
        x1, x2, u = self._make_forced()
        d = koopman_dmd.DMDc(x1, x2, u, rank_input=3, rank_output=2)
        assert d.a_tilde.shape == (2, 2)
        assert d.b_tilde.shape == (2, 1)
        assert d.basis.shape == (2, 2)
        # Basis columns are orthonormal.
        np.testing.assert_allclose(d.basis.T @ d.basis, np.eye(2), atol=1e-12)

    def test_singular_values(self):
        x1, x2, u = self._make_forced()
        d = koopman_dmd.DMDc(x1, x2, u, rank_input=3)
        sv = d.singular_values
        assert sv.shape == (3,)
        assert np.all(np.diff(sv) <= 0)  # non-increasing

    def test_predict_matches_simulation(self):
        x1, x2, u = self._make_forced()
        d = koopman_dmd.DMDc(x1, x2, u, rank_input=3)
        # Predicting with the training inputs from the first snapshot must
        # reproduce x2 (the model is exact for this data).
        pred = d.predict(u=u)
        np.testing.assert_allclose(pred, x2, atol=1e-7)

    def test_predict_zero_input(self):
        x1, x2, u = self._make_forced()
        d = koopman_dmd.DMDc(x1, x2, u, rank_input=3)
        x0 = np.array([1.0, 1.0])
        pred = d.predict(x0=x0, n_ahead=5)
        assert pred.shape == (2, 5)
        # Zero input: pure A-dynamics.
        expect = x0.copy()
        for t in range(5):
            expect = self.A0 @ expect
            np.testing.assert_allclose(pred[:, t], expect, atol=1e-7)

    def test_predict_requires_horizon(self):
        x1, x2, u = self._make_forced()
        d = koopman_dmd.DMDc(x1, x2, u, rank_input=3)
        with pytest.raises(ValueError, match="either u or n_ahead"):
            d.predict()

    def test_shape_mismatch_rejected(self):
        x1, x2, u = self._make_forced()
        with pytest.raises(ValueError):
            koopman_dmd.DMDc(x1, x2[:, :-1], u)
        with pytest.raises(ValueError):
            koopman_dmd.DMDc(x1, x2, u[:, :-1])
        with pytest.raises(ValueError):
            koopman_dmd.DMDc(x1, x2, u, known_b=np.zeros((3, 1)))

    def test_dt_validation(self):
        x1, x2, u = self._make_forced()
        with pytest.raises(ValueError, match="dt"):
            koopman_dmd.DMDc(x1, x2, u, dt=0.0)
        d = koopman_dmd.DMDc(x1, x2, u, dt=0.1)
        assert d.dt == 0.1

    def test_stability(self):
        x1, x2, u = self._make_forced()
        d = koopman_dmd.DMDc(x1, x2, u, rank_input=3)
        is_stable, is_unstable, is_marginal, spectral_radius = d.stability()
        assert is_stable
        assert not is_unstable
        np.testing.assert_allclose(spectral_radius, 0.9, atol=1e-9)

    def test_spectrum(self):
        x1, x2, u = self._make_forced()
        d = koopman_dmd.DMDc(x1, x2, u, rank_input=3)
        spec = d.spectrum()
        assert len(spec) == 2
        mags = sorted(m["magnitude"] for m in spec)
        np.testing.assert_allclose(mags, [0.8, 0.9], atol=1e-9)
        assert all(m["amplitude"] == 0.0 for m in spec)

    def test_repr(self):
        x1, x2, u = self._make_forced()
        d = koopman_dmd.DMDc(x1, x2, u, rank_input=3)
        assert "DMDc" in repr(d)


# ============================================================================
# HankelDMD tests
# ============================================================================


class TestHankelDMD:
    """Test the HankelDMD class."""

    @staticmethod
    def _make_signal():
        t = np.linspace(0, 4 * np.pi, 200)
        y = np.sin(t).reshape(1, -1)  # (1, 200)
        return y

    def test_basic(self):
        y = self._make_signal()
        h = koopman_dmd.HankelDMD(y)
        assert h.rank > 0
        assert h.delays > 0
        assert h.n_obs == 1

    def test_with_delays(self):
        y = self._make_signal()
        h = koopman_dmd.HankelDMD(y, delays=10)
        assert h.delays == 10

    def test_eigenvalues(self):
        y = self._make_signal()
        h = koopman_dmd.HankelDMD(y, delays=10)
        eigs = h.eigenvalues
        assert eigs.ndim == 2
        assert eigs.shape[1] == 2

    def test_residual(self):
        y = self._make_signal()
        h = koopman_dmd.HankelDMD(y, delays=10)
        assert isinstance(h.residual, float)

    def test_reconstruct(self):
        y = self._make_signal()
        h = koopman_dmd.HankelDMD(y, delays=10)
        recon = h.reconstruct(50)
        assert recon.shape[0] == 1  # n_obs

    def test_predict(self):
        y = self._make_signal()
        h = koopman_dmd.HankelDMD(y, delays=10)
        pred = h.predict(20)
        assert pred.shape[0] == 1

    def test_repr(self):
        y = self._make_signal()
        h = koopman_dmd.HankelDMD(y, delays=10)
        r = repr(h)
        assert "HankelDMD" in r


# ============================================================================
# GLA tests
# ============================================================================


class TestGLA:
    """Test the GLA class."""

    @staticmethod
    def _make_signal():
        t = np.linspace(0, 4 * np.pi, 200)
        y = np.vstack([np.sin(t), np.cos(t)])  # (2, 200)
        return y

    def test_basic(self):
        y = self._make_signal()
        g = koopman_dmd.GLA(y, n_eigenvalues=2)
        assert g.n_obs == 2
        assert g.n_time == 200

    def test_eigenvalues(self):
        y = self._make_signal()
        g = koopman_dmd.GLA(y, n_eigenvalues=2)
        eigs = g.eigenvalues
        assert eigs.ndim == 2
        assert eigs.shape[1] == 2

    def test_convergence(self):
        y = self._make_signal()
        g = koopman_dmd.GLA(y, n_eigenvalues=2)
        conv = g.convergence
        assert isinstance(conv, list)

    def test_residuals(self):
        y = self._make_signal()
        g = koopman_dmd.GLA(y, n_eigenvalues=2)
        res = g.residuals
        assert res.ndim == 1

    def test_predict(self):
        y = self._make_signal()
        g = koopman_dmd.GLA(y, n_eigenvalues=2)
        pred = g.predict(10)
        assert pred.shape[0] == 2

    def test_reconstruct(self):
        y = self._make_signal()
        g = koopman_dmd.GLA(y, n_eigenvalues=2)
        recon = g.reconstruct()
        assert recon.shape[0] == 2

    def test_repr(self):
        y = self._make_signal()
        g = koopman_dmd.GLA(y, n_eigenvalues=2)
        r = repr(g)
        assert "GLA" in r

    def test_with_known_eigenvalues(self):
        y = self._make_signal()
        # Provide approximate eigenvalues
        g = koopman_dmd.GLA(y, eigenvalues=[(0.99, 0.1), (0.99, -0.1)])
        assert g.eigenvalues.shape[0] == 2


# ============================================================================
# Map trajectory tests
# ============================================================================


class TestGenerateTrajectory:
    """Test the generate_trajectory function."""

    def test_standard_map(self):
        ic = np.array([0.1, 0.2])
        traj = koopman_dmd.generate_trajectory("standard", ic, 100)
        assert traj.shape == (2, 101)

    def test_standard_map_params(self):
        ic = np.array([0.1, 0.2])
        traj = koopman_dmd.generate_trajectory("standard", ic, 50, {"epsilon": 0.3})
        assert traj.shape == (2, 51)

    def test_henon_map(self):
        ic = np.array([0.0, 0.0])
        traj = koopman_dmd.generate_trajectory("henon", ic, 100, {"a": 1.4, "b": 0.3})
        assert traj.shape == (2, 101)

    def test_logistic_map(self):
        ic = np.array([0.5])
        traj = koopman_dmd.generate_trajectory("logistic", ic, 100, {"r": 3.9})
        assert traj.shape == (1, 101)
        # All values should be in [0, 1]
        assert np.all(traj >= 0) and np.all(traj <= 1)

    def test_froeschle_map(self):
        ic = np.array([0.1, 0.2, 0.3, 0.4])
        traj = koopman_dmd.generate_trajectory("froeschle", ic, 50)
        assert traj.shape == (4, 51)

    def test_extended_standard_map(self):
        ic = np.array([0.1, 0.2, 0.3])
        traj = koopman_dmd.generate_trajectory("extended_standard", ic, 50)
        assert traj.shape == (3, 51)

    def test_unknown_map(self):
        ic = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="unknown map"):
            koopman_dmd.generate_trajectory("unknown", ic, 10)


# ============================================================================
# HTA tests
# ============================================================================


class TestHarmonicTimeAverage:
    """Test the harmonic_time_average function."""

    def test_basic(self):
        ic = np.array([0.1, 0.2])
        mag, phase, hta_re, hta_im = koopman_dmd.harmonic_time_average(
            "standard", ic, "sin_pi", 0.1, 1000
        )
        assert isinstance(mag, float)
        assert isinstance(phase, float)
        assert mag >= 0

    def test_different_observables(self):
        ic = np.array([0.1, 0.2])
        for obs in ["identity", "sin_pi", "cos_pi", "sin_2pi", "cos_2pi"]:
            mag, phase, re, im = koopman_dmd.harmonic_time_average(
                "standard", ic, obs, 0.1, 500
            )
            assert mag >= 0

    def test_invalid_observable(self):
        ic = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="unknown observable"):
            koopman_dmd.harmonic_time_average("standard", ic, "invalid", 0.1, 100)


# ============================================================================
# Mesochronic compute tests
# ============================================================================


class TestMesochronicCompute:
    """Test the mesochronic_compute function."""

    def test_basic(self):
        hta, phase, x, y = koopman_dmd.mesochronic_compute(
            "standard",
            (0.0, 1.0),
            (0.0, 1.0),
            10,
            "sin_pi",
            0.1,
            100,
        )
        assert hta.shape == (10, 10)
        assert phase.shape == (10, 10)
        assert len(x) == 10
        assert len(y) == 10

    def test_with_params(self):
        hta, phase, x, y = koopman_dmd.mesochronic_compute(
            "standard",
            (0.0, 1.0),
            (0.0, 1.0),
            5,
            "cos_pi",
            0.2,
            50,
            {"epsilon": 0.3},
        )
        assert hta.shape == (5, 5)


# ============================================================================
# classify_phase_space tests
# ============================================================================


class TestClassifyPhaseSpace:
    """Test the classify_phase_space function."""

    def test_classification(self):
        mags = np.array([0.5, 0.001, 1e-6, 0.1])
        labels = koopman_dmd.classify_phase_space(mags)
        assert labels.shape == (4,)
        assert labels.dtype == np.int32

    def test_custom_thresholds(self):
        mags = np.array([0.5, 0.005, 1e-6])
        labels = koopman_dmd.classify_phase_space(
            mags, resonating_threshold=0.1, chaotic_threshold=0.001
        )
        assert labels.shape == (3,)


# ============================================================================
# hta_convergence tests
# ============================================================================


class TestHtaConvergence:
    """Test the hta_convergence function."""

    def test_basic(self):
        ic = np.array([0.1, 0.2])
        result = koopman_dmd.hta_convergence("standard", ic, "sin_pi", 0.1, 1000)
        assert isinstance(result, dict)
        assert "times" in result
        assert "hta_magnitudes" in result
        assert "dynamics_type" in result
        assert len(result["times"]) == len(result["hta_magnitudes"])


# ============================================================================
# Integration test: DMD on map trajectory
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_dmd_on_trajectory(self):
        """Run DMD on a dynamical system trajectory."""
        ic = np.array([0.1, 0.2])
        traj = koopman_dmd.generate_trajectory("standard", ic, 200)
        d = koopman_dmd.DMD(traj, rank=5)
        recon = d.reconstruct(traj.shape[1])
        assert recon.shape == traj.shape

    def test_hankel_on_logistic(self):
        """Hankel-DMD on logistic map trajectory."""
        ic = np.array([0.5])
        traj = koopman_dmd.generate_trajectory("logistic", ic, 300, {"r": 3.5})
        h = koopman_dmd.HankelDMD(traj, delays=10, rank=5)
        assert h.rank == 5

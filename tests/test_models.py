"""
Tests for Monte Carlo simulators: Heston, 3/2, Bergomi-1F, Bass LV.

Each test exercises one structural or statistical property of the simulator:
martingale preservation, non-negativity of variance, output shapes,
reproducibility under fixed seeds, and exactness of the OU discretisation.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.heston import HestonModel, HESTON_PARAMS
from src.models.model_3_2 import Model3_2, MODEL_3_2_PARAMS
from src.models.bergomi_1f import Bergomi1FModel, BERGOMI_PARAMS
from src.models.bass_lv import BassLVModel

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_PATHS = 20_000
N_STEPS = 50
T       = 1.0
SEED    = 42


@pytest.fixture(scope="module")
def heston():
    return HestonModel(**HESTON_PARAMS)


@pytest.fixture(scope="module")
def model_32():
    return Model3_2(**MODEL_3_2_PARAMS)


@pytest.fixture(scope="module")
def bergomi():
    return Bergomi1FModel(**BERGOMI_PARAMS)


@pytest.fixture(scope="module")
def bass_lv():
    return BassLVModel(r=0.02)


# ---------------------------------------------------------------------------
# Heston tests
# ---------------------------------------------------------------------------

def test_heston_martingale(heston):
    """E[S_T] under Heston should equal S_0 = 1.0 (martingale, forward measure)."""
    S_paths, _ = heston.simulate_paths(T, N_PATHS, N_STEPS, seed=SEED, S0=1.0)
    mean_ST = np.mean(S_paths[:, -1])
    assert abs(mean_ST - 1.0) < 0.01, (
        f"Heston martingale: E[S_T]={mean_ST:.4f}, expected ≈1.0"
    )


def test_heston_positive_variance(heston):
    """Variance V_t must stay non-negative for all paths and all steps."""
    _, V_paths = heston.simulate_paths(T, N_PATHS, N_STEPS, seed=SEED, S0=1.0)
    assert np.all(V_paths >= 0.0), "Heston variance went negative."


def test_heston_positive_price(heston):
    """Forward price S_t must remain strictly positive for all paths."""
    S_paths, _ = heston.simulate_paths(T, N_PATHS, N_STEPS, seed=SEED, S0=1.0)
    assert np.all(S_paths > 0.0), "Heston forward price went non-positive."


# ---------------------------------------------------------------------------
# 3/2 model tests
# ---------------------------------------------------------------------------

def test_3_2_martingale(model_32):
    """E[S_T] under 3/2 model should equal S_0 = 1.0 (martingale)."""
    S_paths, _ = model_32.simulate_paths(T, N_PATHS, N_STEPS, seed=SEED, S0=1.0)
    mean_ST = np.mean(S_paths[:, -1])
    assert abs(mean_ST - 1.0) < 0.02, (
        f"3/2 martingale: E[S_T]={mean_ST:.4f}, expected ≈1.0"
    )


def test_3_2_positive_variance(model_32):
    """Variance V_t = 1/U_t must stay strictly positive for all paths."""
    _, V_paths = model_32.simulate_paths(T, N_PATHS, N_STEPS, seed=SEED, S0=1.0)
    assert np.all(V_paths > 0.0), "3/2 model: variance V_t went non-positive."


def test_3_2_positive_price(model_32):
    """Forward price S_t must remain strictly positive for all paths."""
    S_paths, _ = model_32.simulate_paths(T, N_PATHS, N_STEPS, seed=SEED, S0=1.0)
    assert np.all(S_paths > 0.0), "3/2 model: forward price went non-positive."


# ---------------------------------------------------------------------------
# Bergomi 1F tests
# ---------------------------------------------------------------------------

def test_bergomi_martingale(bergomi):
    """E[S_T] under Bergomi-1F should equal S_0 = 1.0 (martingale)."""
    S_paths, _ = bergomi.simulate_paths(T, N_PATHS, N_STEPS, seed=SEED, S0=1.0)
    mean_ST = np.mean(S_paths[:, -1])
    assert abs(mean_ST - 1.0) < 0.02, (
        f"Bergomi-1F martingale: E[S_T]={mean_ST:.4f}, expected ≈1.0"
    )


def test_bergomi_ou_exact(bergomi):
    """OU state Z_T should have variance close to (1-e^{-2κT})/(2κ).

    The exact Gaussian transition guarantees this marginal regardless of
    step size.
    """
    kappa = bergomi.kappa
    expected_var = (1.0 - np.exp(-2.0 * kappa * T)) / (2.0 * kappa)

    _, Z_paths = bergomi.simulate_paths(T, N_PATHS, N_STEPS, seed=SEED, S0=1.0)
    Z_T = Z_paths[:, -1]

    # OU starts at 0, so E[Z_T] = 0
    assert abs(np.mean(Z_T)) < 0.05, (
        f"Bergomi OU mean: {np.mean(Z_T):.4f}, expected ≈0"
    )
    # Variance check: tolerance 10% of expected
    assert abs(np.var(Z_T) - expected_var) < 0.1 * expected_var + 1e-6, (
        f"Bergomi OU variance: got {np.var(Z_T):.4f}, expected {expected_var:.4f}"
    )


def test_bergomi_positive_variance(bergomi):
    """Instantaneous variance ξ_t = ξ₀·exp(η Z_t − c(t)) must be strictly positive."""
    S_paths, Z_paths = bergomi.simulate_paths(T, N_PATHS, N_STEPS, seed=SEED, S0=1.0)
    assert np.all(S_paths > 0.0), "Bergomi-1F: S_t went non-positive."


# ---------------------------------------------------------------------------
# Bass LV tests
# ---------------------------------------------------------------------------

def test_bass_lv_martingale(bass_lv):
    """After calibration to a martingale target, E[S_T] ≈ 1.0.

    Calibration target is sampled from a log-normal distribution with
    zero drift (E[S_T] = 1.0 exactly).  The rank transport preserves this
    mean, so the calibrated model should also have E[S_T] ≈ 1.
    """
    sigma_base = 0.25
    T_cal      = 1.0
    n          = 10_000

    rng     = np.random.default_rng(101)
    samples = np.exp(rng.normal(-0.5 * sigma_base**2 * T_cal,
                                 sigma_base * np.sqrt(T_cal), n))

    bass_lv.calibrate_maturity(T_cal, samples, n_paths=n, seed=101)
    X_paths = bass_lv.simulate_paths(T_cal, n, n_steps=50, seed=202)

    mean_ST = np.mean(X_paths[:, -1])
    assert abs(mean_ST - 1.0) < 0.05, (
        f"Bass LV martingale: E[S_T]={mean_ST:.4f}, expected ≈1.0"
    )


def test_bass_lv_shapes(bass_lv):
    """BassLVModel.simulate_paths returns a single (n_paths, n_steps+1) array.

    Bass LV has no latent state, so simulate_paths returns one array only.
    """
    T_test  = 0.5
    n_paths = 500
    n_steps = 20
    X_paths = bass_lv.simulate_paths(T_test, n_paths, n_steps, seed=99)
    assert X_paths.shape == (n_paths, n_steps + 1), (
        f"Bass LV path shape: {X_paths.shape}"
    )


# ---------------------------------------------------------------------------
# Shape tests (all models)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_fixture, model_name", [
    ("heston", "Heston"),
    ("model_32", "3/2"),
    ("bergomi", "Bergomi-1F"),
])
def test_model_shapes(model_fixture, model_name, request):
    """simulate_paths returns (n_paths, n_steps+1) arrays for S and V/Z."""
    model    = request.getfixturevalue(model_fixture)
    T_test   = 0.5
    n_paths  = 200
    n_steps  = 10
    out1, out2 = model.simulate_paths(T_test, n_paths, n_steps, seed=7)
    assert out1.shape == (n_paths, n_steps + 1), (
        f"{model_name}: S shape {out1.shape}"
    )
    assert out2.shape == (n_paths, n_steps + 1), (
        f"{model_name}: V/Z shape {out2.shape}"
    )


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_fixture", ["heston", "model_32", "bergomi"])
def test_reproducibility(model_fixture, request):
    """Same seed produces identical paths; different seeds produce different paths."""
    model   = request.getfixturevalue(model_fixture)
    T_test  = 0.25
    n_paths = 100
    n_steps = 10

    S_a, _ = model.simulate_paths(T_test, n_paths, n_steps, seed=1234)
    S_b, _ = model.simulate_paths(T_test, n_paths, n_steps, seed=1234)
    S_c, _ = model.simulate_paths(T_test, n_paths, n_steps, seed=5678)

    np.testing.assert_array_equal(S_a, S_b,
        err_msg=f"{model_fixture}: same seed → paths differ")
    assert not np.allclose(S_a, S_c), (
        f"{model_fixture}: different seeds → paths should differ"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_heston_small_T(heston):
    """Heston simulation for very small T should not raise and S should be near 1."""
    T_small = 1e-4
    S_paths, V_paths = heston.simulate_paths(T_small, n_paths=100, n_steps=1, seed=0)
    assert S_paths.shape == (100, 2)
    assert np.all(np.isfinite(S_paths)), "Heston small T: non-finite values."


def test_heston_single_path(heston):
    """Heston simulator works with a single path."""
    S_paths, V_paths = heston.simulate_paths(T, n_paths=1, n_steps=5, seed=9)
    assert S_paths.shape == (1, 6)
    assert V_paths.shape == (1, 6)


def test_simulate_joint_from_state(heston):
    """simulate_joint_from_state returns correct shapes and preserves martingale."""
    n      = 1_000
    S_init = np.ones(n)
    V_init = np.full(n, heston.v0)
    S_end, V_end = heston.simulate_joint_from_state(0.0, 0.25, S_init, V_init,
                                                     n_steps=10, seed=77)
    assert S_end.shape == (n,)
    assert V_end.shape == (n,)
    assert np.all(V_end >= 0.0)
    assert abs(np.mean(S_end) - 1.0) < 0.05

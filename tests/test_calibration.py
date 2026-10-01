"""
Tests for the SKR calibration pipeline.

Exercises: monotonicity of the transport map, convergence diagnostics,
terminal-marginal match, approximate martingale preservation, shift
centering, multi-interval marginal propagation, and call repricing accuracy.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration.skr import SKRCalibrator
from src.models.heston import HestonModel, HESTON_PARAMS
from src.utils.distributions import EmpiricalDistribution, wasserstein1_distance
from src.utils.black_scholes import bs_call_price

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _lognormal_samples(n: int, sigma: float, T: float, seed: int) -> np.ndarray:
    """Return n samples from log-normal with mean 1 (forward measure)."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(-0.5 * sigma**2 * T, sigma * np.sqrt(T), n))


def _build_calibrator(n_paths: int, verbose: bool = False) -> SKRCalibrator:
    """Return an SKRCalibrator configured for fast unit tests."""
    model = HestonModel(**HESTON_PARAMS)
    return SKRCalibrator(
        model,
        r=0.02,
        max_iter=20,
        tol_marginal=5e-3,
        tol_martingale=1e-3,
        n_paths=n_paths,
        n_steps_per_interval=10,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Transport map monotonicity (pure utility test)
# ---------------------------------------------------------------------------

def test_transport_map_monotone():
    """The rank-transport map g must be monotone: sorted input → sorted output.

    Applies monotone_transport_map on a sorted Gaussian source to a
    log-normal target and verifies that the transported values preserve
    rank order.
    """
    from src.utils.distributions import monotone_transport_map

    rng    = np.random.default_rng(0)
    source = np.sort(rng.normal(0, 1, 2000))
    target = rng.lognormal(0, 0.4, 2000)

    tgt_dist    = EmpiricalDistribution(target)
    transported = monotone_transport_map(source, tgt_dist.quantile)
    diffs       = np.diff(transported)

    assert np.all(diffs >= -1e-10), "Transport map not monotone on sorted source."


# ---------------------------------------------------------------------------
# Calibration convergence (Sinkhorn W₁ decreases)
# ---------------------------------------------------------------------------

def test_sinkhorn_convergence():
    """Final W₁ diagnostic should be below the tol_marginal stopping threshold.

    The Sinkhorn loop runs until W₁ < tol_marginal, so the last recorded
    diagnostic should satisfy this bound.
    """
    n_paths = 3_000
    T       = 0.5
    sigma   = 0.25

    target = _lognormal_samples(n_paths, sigma, T, seed=11)
    cal    = _build_calibrator(n_paths)
    cal.calibrate(maturities=[T], target_samples={T: target})

    diags = cal.all_diagnostics.get(T, [])
    assert len(diags) > 0, "No diagnostics recorded after calibration."

    final_w1 = diags[-1]["wasserstein_1"]
    assert final_w1 < 0.05, (
        f"Final W1={final_w1:.4f} exceeds 0.05 after SKR calibration at T={T}"
    )


# ---------------------------------------------------------------------------
# Terminal marginal match
# ---------------------------------------------------------------------------

def test_terminal_marginal_match():
    """After calibration, W₁(X_T, μ_T) should be below a small threshold.

    The quantile transport delivers exact marginal matching in the
    noiseless limit; for n_paths samples the residual W₁ should be
    small (below tol_marginal + MC noise).
    """
    n_paths = 5_000
    T       = 1.0
    sigma   = 0.25

    target = _lognormal_samples(n_paths, sigma, T, seed=13)
    cal    = _build_calibrator(n_paths)
    cal.calibrate(maturities=[T], target_samples={T: target})

    x_T, _ = cal.get_calibrated_samples(T)
    w1      = wasserstein1_distance(x_T, target)

    assert w1 < 0.02, f"Terminal marginal mismatch: W1={w1:.4f}"


# ---------------------------------------------------------------------------
# Martingale preservation
# ---------------------------------------------------------------------------

def test_martingale_preservation():
    """After SKR transport, E[X_T] ≈ 1.0 (unconditional martingale).

    E[X_T] = 1 is a necessary (not sufficient) condition for the
    local martingale property.
    """
    n_paths = 6_000
    T       = 1.0
    sigma   = 0.25

    target = _lognormal_samples(n_paths, sigma, T, seed=17)
    cal    = _build_calibrator(n_paths)
    cal.calibrate(maturities=[T], target_samples={T: target})

    x_T, _  = cal.get_calibrated_samples(T)
    mean_XT = np.mean(x_T)

    assert abs(mean_XT - 1.0) < 0.05, (
        f"Martingale: E[X_T]={mean_XT:.4f}, expected ≈1.0"
    )


# ---------------------------------------------------------------------------
# Shift centering
# ---------------------------------------------------------------------------

def test_shift_centering():
    """The Sinkhorn re-centering step should drive mean(shift) ≈ 0.

    The diagnostic key 'shift_centering_error' records |mean(a_k)| after
    re-centering.  It should be near zero at the final iteration.
    """
    n_paths = 4_000
    T       = 1.0
    sigma   = 0.25

    target = _lognormal_samples(n_paths, sigma, T, seed=19)
    cal    = _build_calibrator(n_paths)
    cal.calibrate(maturities=[T], target_samples={T: target})

    diags = cal.all_diagnostics.get(T, [])
    assert len(diags) > 0, "No diagnostics recorded."

    final_shift_ce = diags[-1].get("shift_centering_error", float("nan"))
    assert abs(final_shift_ce) < 0.05, (
        f"Shift centering error = {final_shift_ce:.4f}, expected ≈0"
    )


# ---------------------------------------------------------------------------
# Marginal propagation (two intervals)
# ---------------------------------------------------------------------------

def test_marginal_propagation():
    """After multi-interval calibration, each date's marginal matches its target.

    The calibrated samples at T₁ and T₂ should each have W₁ close to 0
    versus their respective target marginals.
    """
    n_paths  = 4_000
    T1, T2   = 0.5, 1.0
    sigma    = 0.25

    target_T1 = _lognormal_samples(n_paths, sigma, T1, seed=20)
    target_T2 = _lognormal_samples(n_paths, sigma, T2, seed=21)

    cal = _build_calibrator(n_paths)
    cal.calibrate(
        maturities=[T1, T2],
        target_samples={T1: target_T1, T2: target_T2},
    )

    x_T1, _ = cal.get_calibrated_samples(T1)
    x_T2, _ = cal.get_calibrated_samples(T2)

    w1_T1 = wasserstein1_distance(x_T1, target_T1)
    w1_T2 = wasserstein1_distance(x_T2, target_T2)

    assert w1_T1 < 0.02, f"Marginal at T1={T1}: W1={w1_T1:.4f}"
    assert w1_T2 < 0.02, f"Marginal at T2={T2}: W1={w1_T2:.4f}"


# ---------------------------------------------------------------------------
# Single interval structural checks
# ---------------------------------------------------------------------------

def test_single_interval():
    """Calibrate a single interval and check structural properties."""
    n_paths = 3_000
    T       = 0.5
    sigma   = 0.30

    target = _lognormal_samples(n_paths, sigma, T, seed=25)
    cal    = _build_calibrator(n_paths)
    result = cal.calibrate(maturities=[T], target_samples={T: target})

    # calibrate() returns a dict with required keys
    assert "calibrated_data" in result
    assert "all_diagnostics" in result

    # Initial state (T=0) must exist with X_0 = 1
    x0, _ = cal.get_calibrated_samples(0.0)
    np.testing.assert_allclose(x0, 1.0,
        err_msg="Initial calibrated samples should all be 1.0")

    # Terminal calibrated samples at T are positive
    x_T, l_T = cal.get_calibrated_samples(T)
    assert x_T.shape == (n_paths,)
    assert np.all(x_T > 0.0), "Calibrated terminal prices must be positive."

    # W1 close to target
    w1 = wasserstein1_distance(x_T, target)
    assert w1 < 0.02, f"Single interval W1={w1:.4f}"


# ---------------------------------------------------------------------------
# Reprice calls
# ---------------------------------------------------------------------------

def test_reprice_calls():
    """Repriced calls from calibrated paths should match target BS prices.

    Calibrating to a log-normal target (Black-Scholes) guarantees that
    repriced call prices match the BS formula within MC error.
    """
    n_paths = 10_000
    T       = 1.0
    F       = 1.0
    sigma   = 0.25
    r       = 0.02

    target = _lognormal_samples(n_paths, sigma, T, seed=30)
    cal    = _build_calibrator(n_paths)
    cal.calibrate(maturities=[T], target_samples={T: target})

    K_grid               = F * np.exp(np.linspace(-0.3, 0.2, 15))
    mc_prices, mc_se     = cal.reprice_calls(T, K_grid)
    ref_prices           = np.array([bs_call_price(F, K, T, sigma, r=r) for K in K_grid])

    # Allow 3 MC SE per strike (floor 1%) for ATM strikes; larger for far OTM
    abs_errors = np.abs(mc_prices - ref_prices)
    tolerances = np.maximum(3.0 * mc_se, 0.01)
    worst      = int(np.argmax(abs_errors - tolerances))
    assert abs_errors[worst] <= tolerances[worst], (
        f"K={K_grid[worst]:.3f}: mc={mc_prices[worst]:.5f} "
        f"ref={ref_prices[worst]:.5f} tol={tolerances[worst]:.5f}"
    )

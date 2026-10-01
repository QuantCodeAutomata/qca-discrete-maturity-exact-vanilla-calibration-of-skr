"""
Tests for exotic product pricers: reverse cliquet and memory autocallable.

Each test exercises a specific structural, edge-case, or numerical property
of the payoff computation.  All tests use real code paths on synthetic
deterministic or pseudo-random path arrays — no mocking.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exotics.reverse_cliquet import (
    compute_reverse_cliquet_payoff,
    price_reverse_cliquet,
    sensitivity_reverse_cliquet,
)
from src.exotics.memory_autocallable import (
    price_memory_autocallable,
    compute_autocall_statistics,
    sensitivity_autocallable,
)

# ---------------------------------------------------------------------------
# Default product parameters
# ---------------------------------------------------------------------------

N_NOTIONAL = 100.0
C0         = 0.50
L_BASE     = 0.075
R_RATE     = 0.02
T_FINAL    = 3.0   # 12 quarters
N_QUARTERS = 12
N_DATES    = N_QUARTERS + 1  # including t=0

AC_C        = 0.05
AC_B        = 0.60
AC_AUTOCALL = 1.00
AC_B_P      = 0.60
AC_S0       = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_path(n_paths: int, n_dates: int = N_DATES, S0: float = 1.0) -> np.ndarray:
    """Flat path: S_t = S0 at all dates (zero returns)."""
    return np.full((n_paths, n_dates), S0)


def _down_path(n_paths: int, n_dates: int = N_DATES, factor: float = 0.5) -> np.ndarray:
    """Exponentially decaying path: S_{t_j} = S0 * factor^j."""
    S = np.ones((n_paths, n_dates))
    for j in range(1, n_dates):
        S[:, j] = S[:, j - 1] * factor
    return S


def _above_barrier_path(n_paths: int, n_dates: int, level: float) -> np.ndarray:
    """Path that stays at 'level' for all dates after t=0."""
    S = np.ones((n_paths, n_dates))
    S[:, 1:] = level
    return S


# ---------------------------------------------------------------------------
# Reverse cliquet — payoff bounds
# ---------------------------------------------------------------------------

def test_reverse_cliquet_payoff_bounds():
    """Payoff must lie in [N, N*(1+C0)] for all paths and all L values.

    The floor arises when aggregate loss ≥ C0 (all coupon depleted).
    The ceiling arises when aggregate loss = 0 (no market falls).
    """
    rng    = np.random.default_rng(0)
    # Simulate random paths around 1.0
    S_paths = np.exp(rng.normal(0, 0.02, (5_000, N_DATES)).cumsum(axis=1))
    S_paths = np.hstack([np.ones((5_000, 1)), S_paths])[:, :N_DATES]

    for L in [0.01, 0.05, 0.10, 0.20]:
        payoffs = compute_reverse_cliquet_payoff(S_paths, N=N_NOTIONAL, C0=C0, L=L)
        assert np.all(payoffs >= N_NOTIONAL - 1e-9), (
            f"Payoff below floor N={N_NOTIONAL} for L={L}."
        )
        assert np.all(payoffs <= N_NOTIONAL * (1.0 + C0) + 1e-9), (
            f"Payoff above ceiling N*(1+C0)={N_NOTIONAL*(1+C0)} for L={L}."
        )


# ---------------------------------------------------------------------------
# Reverse cliquet — trivial (flat) path
# ---------------------------------------------------------------------------

def test_reverse_cliquet_trivial():
    """Flat path (all returns = 0) → payoff = N*(1+C0) = 150.

    When every return R_j = 0, each local loss = min(0, L) = 0, so the
    aggregate loss is 0 and the full coupon budget C0 is retained.
    """
    S_paths = _flat_path(n_paths=100)
    payoffs = compute_reverse_cliquet_payoff(S_paths, N=N_NOTIONAL, C0=C0, L=L_BASE)
    expected = N_NOTIONAL * (1.0 + C0)
    np.testing.assert_allclose(payoffs, expected, atol=1e-10,
        err_msg=f"Flat path payoff: got {payoffs[0]:.4f}, expected {expected:.4f}")


# ---------------------------------------------------------------------------
# Reverse cliquet — extreme down path
# ---------------------------------------------------------------------------

def test_reverse_cliquet_max_loss():
    """Extreme down path (large falls each quarter) → payoff = N.

    When each |R_j| >> L, the cap binds at L every quarter, so
    agg_loss = n_quarters * L.  With n_quarters=12 and L=0.075,
    agg_loss = 0.9 > C0 = 0.5, so residual = 0 and payoff = N.
    """
    # Returns of −30 % each quarter, much larger than L=0.075
    S_paths = _down_path(n_paths=100, factor=0.70)
    payoffs = compute_reverse_cliquet_payoff(S_paths, N=N_NOTIONAL, C0=C0, L=L_BASE)
    np.testing.assert_allclose(payoffs, N_NOTIONAL, atol=1e-10,
        err_msg=f"Extreme down path: payoff={payoffs[0]:.4f}, expected {N_NOTIONAL}")


# ---------------------------------------------------------------------------
# Reverse cliquet — exact payoff on known path
# ---------------------------------------------------------------------------

def test_payoff_with_known_path_reverse_cliquet():
    """Verify exact payoff computation on a synthetic path with known answer.

    Path: 12 quarters of constant −5 % returns.
    With L=0.03: local_loss = min(0.05, 0.03)=0.03 each quarter
    agg_loss = 12 × 0.03 = 0.36
    residual = max(0.5 − 0.36, 0) = 0.14
    payoff = 100 × (1 + 0.14) = 114.0
    """
    n_paths = 1
    S = np.ones((n_paths, N_DATES))
    for j in range(1, N_DATES):
        S[:, j] = S[:, j - 1] * 0.95   # −5 % each quarter

    payoffs = compute_reverse_cliquet_payoff(S, N=N_NOTIONAL, C0=C0, L=0.03)
    expected = N_NOTIONAL * (1.0 + 0.14)   # = 114.0
    np.testing.assert_allclose(payoffs[0], expected, rtol=1e-6,
        err_msg=f"Known path payoff: got {payoffs[0]:.4f}, expected {expected}")


# ---------------------------------------------------------------------------
# Reverse cliquet — MC standard error
# ---------------------------------------------------------------------------

def test_mc_se_positive():
    """Monte Carlo standard error must be strictly positive.

    SE > 0 iff there is genuine variance in the payoffs across paths.
    """
    rng     = np.random.default_rng(1)
    S_paths = np.exp(rng.normal(0, 0.04, (2_000, N_DATES)).cumsum(axis=1))
    S_paths = np.hstack([np.ones((2_000, 1)), S_paths])[:, :N_DATES]

    price, mc_se = price_reverse_cliquet(
        S_paths, r=R_RATE, T=T_FINAL, N=N_NOTIONAL, C0=C0, L=L_BASE
    )
    assert mc_se > 0.0, f"MC standard error should be positive; got {mc_se}."
    assert np.isfinite(price), f"Price is not finite: {price}"


# ---------------------------------------------------------------------------
# Reverse cliquet — sensitivity monotone in L
# ---------------------------------------------------------------------------

def test_sensitivity_monotone():
    """Reverse cliquet price is monotone *decreasing* in L.

    Increasing L allows more coupon depletion per quarter, reducing the
    expected residual coupon and hence the product price.
    """
    rng     = np.random.default_rng(2)
    S_paths = np.exp(rng.normal(0, 0.04, (5_000, N_DATES)).cumsum(axis=1))
    S_paths = np.hstack([np.ones((5_000, 1)), S_paths])[:, :N_DATES]

    L_grid = np.array([0.01, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20])
    prices, _ = sensitivity_reverse_cliquet(
        S_paths, L_grid, r=R_RATE, T=T_FINAL, N=N_NOTIONAL, C0=C0
    )

    diffs = np.diff(prices)
    assert np.all(diffs <= 1e-6), (
        f"Reverse cliquet price should decrease as L increases; got diffs: {diffs}"
    )


# ---------------------------------------------------------------------------
# Memory autocallable — early redemption (Q1 autocall)
# ---------------------------------------------------------------------------

def test_autocallable_early_redemption():
    """When all paths autocall in Q1, price = exp(-r*0.25)*(N + c*N).

    With a single quarterly period that immediately autocalls: the path
    is at autocall_barrier × S0 at t = 0.25, pending = 0, so the payment
    is N + (0+1)*c*N = N*(1+c) discounted by exp(-r*0.25).
    """
    n_paths = 1_000
    # 2-date paths: t=0 (S=1) and t=0.25 (S=1.1 ≥ autocall_barrier=1.0)
    S_paths = np.column_stack([
        np.ones(n_paths),
        np.full(n_paths, 1.1),
    ])   # shape (1000, 2)

    price, mc_se = price_memory_autocallable(
        S_paths, r=R_RATE, N=N_NOTIONAL, c=AC_C,
        b=AC_B, autocall_barrier=AC_AUTOCALL,
        b_P=AC_B_P, S0=AC_S0,
    )
    expected = np.exp(-R_RATE * 0.25) * N_NOTIONAL * (1.0 + AC_C)
    np.testing.assert_allclose(price, expected, rtol=1e-8,
        err_msg=f"Early redemption price: got {price:.4f}, expected {expected:.4f}")
    assert mc_se < 1e-10, f"SE should be 0 for deterministic paths; got {mc_se}"


# ---------------------------------------------------------------------------
# Memory autocallable — coupon memory accumulation
# ---------------------------------------------------------------------------

def test_autocallable_memory():
    """Memory coupon: missing Q1 coupon is paid in Q2 when condition is met.

    Constructed path (4 dates: t = 0, 0.25, 0.50, 0.75):
      t=0.25: S=0.5 < b*S0=0.60  → no coupon; pending=1
      t=0.50: S=0.7 ≥ 0.60, < 1.0 → coupon met: pay (1+1)*c*N=10; pending=0
      t=0.75: final → S=0.9 ≥ b_P*S0=0.60 → pay N=100

    PV = 10*exp(-r*0.50) + 100*exp(-r*0.75)
    """
    n = 1_000
    S_paths = np.column_stack([
        np.ones(n),               # t=0
        np.full(n, 0.5),          # t=0.25: below coupon barrier
        np.full(n, 0.7),          # t=0.50: above coupon, below autocall
        np.full(n, 0.9),          # t=0.75: terminal, above b_P
    ])

    price, mc_se = price_memory_autocallable(
        S_paths, r=R_RATE, N=N_NOTIONAL, c=AC_C,
        b=AC_B, autocall_barrier=AC_AUTOCALL,
        b_P=AC_B_P, S0=AC_S0,
    )

    # Memory coupon at t=0.50: 2*c*N = 10
    # Terminal at t=0.75: N = 100
    expected = (
        2 * AC_C * N_NOTIONAL * np.exp(-R_RATE * 0.50)
        + N_NOTIONAL * np.exp(-R_RATE * 0.75)
    )
    np.testing.assert_allclose(price, expected, rtol=1e-8,
        err_msg=f"Memory coupon price: got {price:.4f}, expected {expected:.4f}")

    # Verify memory effect: price should be higher than without memory
    # Without memory (c → c * 1 always): coupon paid = 1*c*N at t=0.50
    no_memory_expected = (
        AC_C * N_NOTIONAL * np.exp(-R_RATE * 0.50)
        + N_NOTIONAL * np.exp(-R_RATE * 0.75)
    )
    assert price > no_memory_expected, (
        "Memory should increase the price vs no-memory case."
    )


# ---------------------------------------------------------------------------
# Memory autocallable — terminal protection
# ---------------------------------------------------------------------------

def test_autocallable_terminal_protection():
    """Path always below b_P at maturity → payoff = N * S_T / S0.

    With a 3-quarter path (4 dates) where the price stays at 0.3 throughout
    (below both b and b_P = 0.6), no coupons are ever paid and the terminal
    payment is N × 0.3 / 1.0 = 30 discounted at exp(-r*0.75).
    """
    n = 1_000
    level = 0.30   # below b=0.60 and b_P=0.60
    S_paths = np.column_stack([
        np.ones(n),               # t=0
        np.full(n, level),        # t=0.25
        np.full(n, level),        # t=0.50
        np.full(n, level),        # t=0.75 (final)
    ])

    price, mc_se = price_memory_autocallable(
        S_paths, r=R_RATE, N=N_NOTIONAL, c=AC_C,
        b=AC_B, autocall_barrier=AC_AUTOCALL,
        b_P=AC_B_P, S0=AC_S0,
    )
    expected = N_NOTIONAL * level / AC_S0 * np.exp(-R_RATE * 0.75)
    np.testing.assert_allclose(price, expected, rtol=1e-8,
        err_msg=f"Terminal protection: got {price:.4f}, expected {expected:.4f}")


# ---------------------------------------------------------------------------
# Memory autocallable — autocall statistics
# ---------------------------------------------------------------------------

def test_autocall_statistics_shape_and_bounds():
    """compute_autocall_statistics returns arrays with correct shapes and values in [0,1]."""
    rng = np.random.default_rng(3)
    S_paths = np.exp(rng.normal(0, 0.04, (2_000, N_DATES)).cumsum(axis=1))
    S_paths = np.hstack([np.ones((2_000, 1)), S_paths])[:, :N_DATES]

    stats = compute_autocall_statistics(
        S_paths, b=AC_B, autocall_barrier=AC_AUTOCALL,
        b_P=AC_B_P, S0=AC_S0,
    )

    ac_prob   = stats["autocall_prob_per_quarter"]
    coup_freq = stats["coupon_hit_freq_per_quarter"]
    tb_freq   = stats["terminal_barrier_breach_freq"]

    assert ac_prob.shape   == (N_QUARTERS,), f"Autocall prob shape {ac_prob.shape}"
    assert coup_freq.shape == (N_QUARTERS,), f"Coupon freq shape {coup_freq.shape}"

    assert np.all(ac_prob   >= 0.0) and np.all(ac_prob   <= 1.0)
    assert np.all(coup_freq >= 0.0) and np.all(coup_freq <= 1.0)
    assert 0.0 <= tb_freq <= 1.0


def test_autocall_statistics_always_above():
    """When S always stays above autocall barrier: P(autocall Q1) = 1."""
    n = 500
    # All paths above autocall barrier from Q1 onwards
    S_paths = _above_barrier_path(n, N_DATES, level=1.1)

    stats = compute_autocall_statistics(
        S_paths, b=AC_B, autocall_barrier=AC_AUTOCALL,
        b_P=AC_B_P, S0=AC_S0,
    )
    assert stats["autocall_prob_per_quarter"][0] == pytest.approx(1.0), (
        "All paths above barrier: Q1 autocall prob should be 1.0"
    )
    # After Q1 autocall, no further paths are active → prob = 0 for j > 1
    assert np.all(stats["autocall_prob_per_quarter"][1:] == 0.0), (
        "After Q1 autocall, subsequent autocall prob should be 0."
    )


# ---------------------------------------------------------------------------
# Sensitivity — autocallable price varies with coupon barrier b
# ---------------------------------------------------------------------------

def test_sensitivity_autocallable_shape():
    """sensitivity_autocallable returns arrays of the correct shape."""
    rng = np.random.default_rng(4)
    S_paths = np.exp(rng.normal(0, 0.04, (2_000, N_DATES)).cumsum(axis=1))
    S_paths = np.hstack([np.ones((2_000, 1)), S_paths])[:, :N_DATES]

    b_grid         = np.array([0.40, 0.60, 0.80])
    prices, mc_ses = sensitivity_autocallable(
        S_paths, b_grid, r=R_RATE, N=N_NOTIONAL, c=AC_C,
        autocall_barrier=AC_AUTOCALL, b_P=AC_B_P, S0=AC_S0,
    )

    assert prices.shape  == (3,), f"Prices shape: {prices.shape}"
    assert mc_ses.shape  == (3,), f"SEs shape: {mc_ses.shape}"
    assert np.all(mc_ses >= 0.0), "Standard errors must be non-negative."


def test_sensitivity_autocallable_decreasing_b():
    """Autocallable price should be non-increasing in the coupon barrier b.

    Raising the coupon barrier b makes it harder to earn coupons, reducing
    the expected coupon income → lower price.
    """
    rng = np.random.default_rng(5)
    S_paths = np.exp(rng.normal(-0.002, 0.04, (5_000, N_DATES)).cumsum(axis=1))
    S_paths = np.hstack([np.ones((5_000, 1)), S_paths])[:, :N_DATES]

    # Use a wide b_grid to see the monotone effect clearly
    b_grid         = np.array([0.20, 0.40, 0.60, 0.80, 0.95])
    prices, mc_ses = sensitivity_autocallable(
        S_paths, b_grid, r=R_RATE, N=N_NOTIONAL, c=AC_C,
        autocall_barrier=AC_AUTOCALL, b_P=AC_B_P, S0=AC_S0,
    )

    # Allow 2× MC SE as tolerance for each comparison
    ci = 2.0 * mc_ses
    for i in range(len(b_grid) - 1):
        assert prices[i] >= prices[i + 1] - (ci[i] + ci[i + 1]), (
            f"Price not monotone at b={b_grid[i]:.2f}: "
            f"p[i]={prices[i]:.3f}, p[i+1]={prices[i+1]:.3f}"
        )


# ---------------------------------------------------------------------------
# Price and SE positivity
# ---------------------------------------------------------------------------

def test_price_and_se_are_finite_and_positive():
    """Price and SE for random paths should be finite and positive."""
    rng = np.random.default_rng(6)
    S_paths = np.exp(rng.normal(0, 0.03, (3_000, N_DATES)).cumsum(axis=1))
    S_paths = np.hstack([np.ones((3_000, 1)), S_paths])[:, :N_DATES]

    rc_p, rc_se = price_reverse_cliquet(
        S_paths, r=R_RATE, T=T_FINAL, N=N_NOTIONAL, C0=C0, L=L_BASE
    )
    ac_p, ac_se = price_memory_autocallable(
        S_paths, r=R_RATE, N=N_NOTIONAL, c=AC_C,
        b=AC_B, autocall_barrier=AC_AUTOCALL,
        b_P=AC_B_P, S0=AC_S0,
    )

    assert np.isfinite(rc_p) and rc_p > 0
    assert np.isfinite(rc_se) and rc_se > 0
    assert np.isfinite(ac_p) and ac_p > 0
    assert np.isfinite(ac_se) and ac_se > 0

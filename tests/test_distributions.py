"""
Tests for distribution utilities in src/utils/distributions.py.

Covers: EmpiricalDistribution (CDF and quantile), monotone transport map,
Breeden-Litzenberger density extraction, and the 1-Wasserstein distance.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.distributions import (
    EmpiricalDistribution,
    monotone_transport_map,
    compute_cdf_from_call_prices,
    wasserstein1_distance,
)
from src.utils.black_scholes import bs_call_price


# ---------------------------------------------------------------------------
# EmpiricalDistribution — CDF
# ---------------------------------------------------------------------------

def test_empirical_cdf_monotone():
    """Empirical CDF must be non-decreasing on a sorted evaluation grid."""
    rng     = np.random.default_rng(0)
    samples = rng.normal(0, 1, 1000)
    dist    = EmpiricalDistribution(samples)

    x_grid  = np.linspace(-4, 4, 200)
    cdf_vals = dist.cdf(x_grid)

    diffs = np.diff(cdf_vals)
    assert np.all(diffs >= -1e-12), (
        "Empirical CDF is not non-decreasing; max negative diff: "
        f"{diffs.min():.2e}"
    )


def test_empirical_cdf_bounds():
    """Empirical CDF must lie in [0, 1] everywhere."""
    rng     = np.random.default_rng(1)
    samples = rng.uniform(-5, 5, 500)
    dist    = EmpiricalDistribution(samples)

    x_grid   = np.linspace(-10, 10, 300)
    cdf_vals = dist.cdf(x_grid)

    assert np.all(cdf_vals >= 0.0), "CDF has negative values."
    assert np.all(cdf_vals <= 1.0), "CDF exceeds 1."


# ---------------------------------------------------------------------------
# EmpiricalDistribution — quantile
# ---------------------------------------------------------------------------

def test_empirical_quantile_inverse():
    """Quantile function must be an approximate inverse of the CDF.

    For a sample x drawn from the distribution, F(Q(F(x))) ≈ F(x) up to
    the interpolation error introduced by the finite sample.
    """
    rng     = np.random.default_rng(2)
    samples = rng.normal(0, 1, 2000)
    dist    = EmpiricalDistribution(samples)

    # Evaluate CDF at the sorted samples
    x_eval = dist.samples()
    p_vals = dist.cdf(x_eval)

    # Invert back via quantile
    x_recon = dist.quantile(p_vals)

    # The reconstruction should be close to the original sorted samples
    np.testing.assert_allclose(x_recon, x_eval, atol=0.05,
        err_msg="Quantile is not an approximate inverse of the CDF.")


def test_empirical_quantile_monotone():
    """Quantile function (inverse CDF) must be non-decreasing."""
    rng     = np.random.default_rng(3)
    samples = rng.exponential(1, 500)
    dist    = EmpiricalDistribution(samples)

    p_grid  = np.linspace(0.01, 0.99, 300)
    q_vals  = dist.quantile(p_grid)
    diffs   = np.diff(q_vals)

    assert np.all(diffs >= -1e-10), (
        "Quantile function is not non-decreasing."
    )


# ---------------------------------------------------------------------------
# Monotone transport map
# ---------------------------------------------------------------------------

def test_transport_map_pushforward():
    """Transported samples should follow the target distribution.

    After applying the rank transport g = Q_ν ∘ F_μ, the transported
    empirical distribution should be close to ν (target) in W₁.
    """
    rng     = np.random.default_rng(4)
    source  = rng.normal(0, 1, 5000)
    target  = rng.normal(2, 0.5, 5000)   # different mean and scale

    tgt_dist     = EmpiricalDistribution(target)
    transported  = monotone_transport_map(source, tgt_dist.quantile)

    # The transported samples should be close to the target in W1
    w1 = wasserstein1_distance(transported, target)
    assert w1 < 0.05, (
        f"Transport pushforward: W1={w1:.4f}, expected < 0.05 "
        f"(source mean={source.mean():.2f}, target mean={target.mean():.2f})"
    )


def test_transport_map_monotone():
    """The transport map g must be monotone increasing.

    Rank preserving maps turn sorted inputs into sorted outputs.
    """
    rng    = np.random.default_rng(5)
    source = np.sort(rng.normal(0, 1, 2000))
    target = rng.lognormal(0, 0.5, 2000)

    tgt_dist    = EmpiricalDistribution(target)
    transported = monotone_transport_map(source, tgt_dist.quantile)

    # transported[i] should be non-decreasing since source is sorted
    diffs = np.diff(transported)
    assert np.all(diffs >= -1e-10), (
        "Transport map is not monotone increasing on sorted source."
    )


# ---------------------------------------------------------------------------
# Wasserstein distance
# ---------------------------------------------------------------------------

def test_wasserstein_zero():
    """W₁(μ, μ) = 0 when both inputs are the same sample set."""
    rng     = np.random.default_rng(6)
    samples = rng.normal(3, 2, 1000)
    w1      = wasserstein1_distance(samples, samples)
    assert w1 < 1e-12, f"W1(μ, μ) = {w1:.2e}, expected 0."


def test_wasserstein_positive():
    """W₁ > 0 for two clearly different distributions."""
    rng  = np.random.default_rng(7)
    mu_A = rng.normal(0, 1, 2000)
    mu_B = rng.normal(10, 1, 2000)   # shifted by 10
    w1   = wasserstein1_distance(mu_A, mu_B)
    assert w1 > 9.0, f"W1 between N(0,1) and N(10,1) should be ≈10; got {w1:.2f}."


def test_wasserstein_triangle():
    """Triangle inequality: W₁(μ, ν) ≤ W₁(μ, π) + W₁(π, ν)."""
    rng = np.random.default_rng(8)
    mu  = rng.normal(0, 1, 3000)
    nu  = rng.normal(3, 1, 3000)
    pi  = rng.normal(1.5, 1, 3000)

    w_mn = wasserstein1_distance(mu, nu)
    w_mp = wasserstein1_distance(mu, pi)
    w_pn = wasserstein1_distance(pi, nu)

    assert w_mn <= w_mp + w_pn + 1e-9, (
        f"Triangle inequality violated: W(μ,ν)={w_mn:.4f} > "
        f"W(μ,π)+W(π,ν)={w_mp+w_pn:.4f}"
    )


def test_wasserstein_unequal_length():
    """W₁ should work correctly even when the two sample arrays differ in length."""
    rng  = np.random.default_rng(9)
    mu   = rng.normal(0, 1, 1000)
    nu   = rng.normal(0, 1, 3000)   # same distribution, different sample size
    w1   = wasserstein1_distance(mu, nu)
    assert w1 < 0.1, (
        f"W1 between same-distribution samples of different size: {w1:.4f}"
    )


# ---------------------------------------------------------------------------
# Breeden-Litzenberger
# ---------------------------------------------------------------------------

def test_breeden_litzenberger_density_integrates_to_one():
    """Risk-neutral density from Breeden-Litzenberger should integrate to ≈ 1."""
    F     = 1.0
    T     = 1.0
    sigma = 0.25
    r     = 0.0

    K_grid = F * np.exp(np.linspace(-2.0, 1.0, 300))
    C_grid = np.array([bs_call_price(F, K, T, sigma, r=r) for K in K_grid])

    x_grid, density, cdf_vals = compute_cdf_from_call_prices(K_grid, C_grid, F, T)

    integral = np.trapz(density, x_grid)
    assert abs(integral - 1.0) < 0.05, (
        f"Breeden-Litzenberger density integrates to {integral:.4f}, expected ≈1."
    )


def test_breeden_litzenberger_cdf_bounds():
    """Extracted CDF must be non-decreasing and lie in [0, 1]."""
    F     = 1.0
    T     = 0.5
    sigma = 0.20
    r     = 0.0

    K_grid = F * np.exp(np.linspace(-1.5, 0.8, 200))
    C_grid = np.array([bs_call_price(F, K, T, sigma, r=r) for K in K_grid])

    _, _, cdf_vals = compute_cdf_from_call_prices(K_grid, C_grid, F, T)

    assert np.all(cdf_vals >= -1e-10), "CDF has negative values."
    assert np.all(cdf_vals <= 1.0 + 1e-10), "CDF exceeds 1."
    diffs = np.diff(cdf_vals)
    assert np.all(diffs >= -1e-10), "CDF is not non-decreasing."


def test_breeden_litzenberger_mean():
    """For a log-normal (BS) distribution, E[S] should approximately equal F.

    Under risk-neutral measure with r=0, E[S_T] = F.  The Breeden-
    Litzenberger density should reflect this via ∫ x p(x) dx ≈ F.
    """
    F     = 1.0
    T     = 1.0
    sigma = 0.25
    r     = 0.0

    K_grid = F * np.exp(np.linspace(-2.5, 1.2, 500))
    C_grid = np.array([bs_call_price(F, K, T, sigma, r=r) for K in K_grid])

    x_grid, density, _ = compute_cdf_from_call_prices(K_grid, C_grid, F, T)
    mean_x = np.trapz(density * x_grid, x_grid)

    assert abs(mean_x - F) < 0.05, (
        f"Breeden-Litzenberger mean: {mean_x:.4f}, expected ≈F={F}."
    )


# ---------------------------------------------------------------------------
# Convex order
# ---------------------------------------------------------------------------

def test_convex_order():
    """μ ≤_cx ν (convex order) iff ∫ (x−K)⁺ dμ ≤ ∫ (x−K)⁺ dν for all K.

    A scaled version (σ₁ ≤ σ₂ for log-normals with same mean) satisfies
    this.  The call price of the less volatile measure should be ≤ that of
    the more volatile one at every strike.
    """
    F      = 1.0
    T      = 1.0
    r      = 0.0
    n      = 10_000
    rng    = np.random.default_rng(20)

    sigma1, sigma2 = 0.15, 0.35   # σ₁ < σ₂ → μ₁ ≤_cx μ₂

    # Sample from both log-normals (same mean F=1)
    S1 = np.exp(rng.normal(-0.5 * sigma1**2 * T, sigma1 * np.sqrt(T), n))
    S2 = np.exp(rng.normal(-0.5 * sigma2**2 * T, sigma2 * np.sqrt(T), n))

    K_grid = F * np.exp(np.linspace(-0.5, 0.5, 30))

    calls1 = np.array([np.mean(np.maximum(S1 - K, 0)) for K in K_grid])
    calls2 = np.array([np.mean(np.maximum(S2 - K, 0)) for K in K_grid])

    # For all strikes, calls1 ≤ calls2 (with MC tolerance)
    assert np.all(calls1 <= calls2 + 0.01), (
        "Convex order: less volatile distribution does not have lower call prices."
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_sample_distribution():
    """EmpiricalDistribution with a single sample should not raise."""
    dist = EmpiricalDistribution(np.array([2.5]))
    assert dist.cdf(np.array([2.5])) == pytest.approx(0.5)
    assert dist.quantile(np.array([0.5])) == pytest.approx(2.5)


def test_transport_map_identity():
    """Transport from a distribution to itself should be the identity (approx)."""
    rng     = np.random.default_rng(30)
    samples = rng.normal(0, 1, 1000)
    dist    = EmpiricalDistribution(samples)
    result  = monotone_transport_map(samples, dist.quantile)

    # The transported values should be close to the originals
    np.testing.assert_allclose(
        np.sort(result), np.sort(samples), atol=1e-8,
        err_msg="Identity transport: sorted transported ≠ sorted source."
    )


def test_wasserstein_symmetric():
    """W₁(μ, ν) = W₁(ν, μ)."""
    rng = np.random.default_rng(31)
    mu  = rng.normal(0, 1, 500)
    nu  = rng.normal(1, 2, 700)
    w_mn = wasserstein1_distance(mu, nu)
    w_nm = wasserstein1_distance(nu, mu)
    assert abs(w_mn - w_nm) < 1e-10, (
        f"W1 not symmetric: W(μ,ν)={w_mn:.6f}, W(ν,μ)={w_nm:.6f}"
    )

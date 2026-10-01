"""
Extended SSVI (eSSVI) smile parameterisation for SPX as of 23 June 2020.

The eSSVI total-variance surface is

    w(k, θ_T) = (θ_T / 2) * [1 + ρ φ(θ_T) k + √( (φ(θ_T) k + ρ)² + (1−ρ²) )]

where
    k = ln(K/F)         log-moneyness,
    φ(θ)  = η / [θ^γ (1+θ)^{1-γ}]    (Gatheral–Jacquier power law),

and (ρ, η, γ) are model constants.  At k=0 (ATM) the formula collapses to
w(0, θ_T) = θ_T so θ_T equals the ATM total variance σ_{ATM}² T.

References
----------
Gatheral, J. and Jacquier, A. (2014), "Arbitrage-free SVI volatility surfaces",
Quantitative Finance, 14(1), 59–71.
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on the path so that src.utils can be imported.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy.interpolate import interp1d

from src.utils.black_scholes import bs_call_price, bs_implied_vol
from src.utils.distributions import compute_cdf_from_call_prices

# ---------------------------------------------------------------------------
# Calibration maturities
# ---------------------------------------------------------------------------

CALIBRATION_MATURITIES: list = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]

# ---------------------------------------------------------------------------
# eSSVI parameters — SPX 23 June 2020 snapshot
# ---------------------------------------------------------------------------
#
# Global shape parameters shared across all maturities.
# ATM implied vols (approximately):
#   T=0.25 → 27.5 %,  T=0.5 → 26.5 %,  T=1.0 → 25.5 %
#   T=2.0  → 24.5 %,  T=3.0 → 24.0 %
#
# Per-maturity ATM total variance: θ_T = σ_ATM² × T.

_RHO   = -0.70
_ETA   = 1.20
_GAMMA = 0.40

# θ_T values (ATM total variance = σ_ATM² × T)
_ATM_VOLS = {0.25: 0.275, 0.5: 0.265, 1.0: 0.255, 2.0: 0.245, 3.0: 0.240}
_THETA_T  = {T: (_ATM_VOLS[T] ** 2) * T for T in _ATM_VOLS}


# ---------------------------------------------------------------------------
# Core eSSVI functions
# ---------------------------------------------------------------------------

def _phi(theta: float, eta: float, gamma: float) -> float:
    """Power-law skew function φ(θ) = η / [θ^γ (1+θ)^{1-γ}].

    Parameters
    ----------
    theta : float
        ATM total variance θ_T > 0.
    eta : float
        Vol-of-vol-like amplitude parameter.
    gamma : float
        Power parameter in (0, 1).

    Returns
    -------
    float
        Value of φ(θ).
    """
    return eta / (theta ** gamma * (1.0 + theta) ** (1.0 - gamma))


def essvi_total_variance(
    k: np.ndarray,
    T: float,
    theta_t: float,
    rho: float,
    eta: float,
    gamma: float,
) -> np.ndarray:
    """Compute the eSSVI total variance w(k, T) on a log-moneyness grid.

    Parameters
    ----------
    k : np.ndarray
        Log-moneyness grid k = ln(K/F).
    T : float
        Maturity (years).  Not used in the formula itself but retained
        for signature consistency.
    theta_t : float
        ATM total variance at maturity T: θ_T = σ_ATM² T.
    rho : float
        Correlation parameter ρ ∈ (−1, 1).
    eta : float
        Amplitude η > 0.
    gamma : float
        Power parameter γ ∈ (0, 1).

    Returns
    -------
    np.ndarray
        Total variance w(k) ≥ 0, same shape as *k*.
    """
    k = np.asarray(k, dtype=float)
    phi_val = _phi(theta_t, eta, gamma)
    psi = phi_val * k + rho
    w = (theta_t / 2.0) * (1.0 + rho * phi_val * k + np.sqrt(psi ** 2 + 1.0 - rho ** 2))
    return w


def essvi_implied_vol(
    k: np.ndarray,
    T: float,
    theta_t: float,
    rho: float,
    eta: float,
    gamma: float,
) -> np.ndarray:
    """Compute the eSSVI implied volatility IV(k, T) = √(w(k,T) / T).

    Parameters
    ----------
    k : np.ndarray
        Log-moneyness grid.
    T : float
        Maturity in years (must be > 0).
    theta_t : float
        ATM total variance θ_T.
    rho : float
        Correlation ρ.
    eta : float
        Amplitude η.
    gamma : float
        Power γ.

    Returns
    -------
    np.ndarray
        Implied volatility IV(k, T) ≥ 0, same shape as *k*.
    """
    w = essvi_total_variance(k, T, theta_t, rho, eta, gamma)
    return np.sqrt(np.maximum(w, 0.0) / T)


# ---------------------------------------------------------------------------
# Parameter accessors
# ---------------------------------------------------------------------------

def get_essvi_params() -> dict:
    """Return eSSVI parameters for the SPX 23 June 2020 snapshot.

    The global shape parameters (ρ, η, γ) are shared across all maturities;
    the per-maturity ATM total variance θ_T is stored in the nested dict
    ``theta_t``.

    Returns
    -------
    dict
        Keys:

        ``rho``      – correlation ρ = −0.70,
        ``eta``      – amplitude η = 1.20,
        ``gamma``    – power γ = 0.40,
        ``theta_t``  – dict {T: θ_T} for T ∈ {0.25, 0.5, 1.0, 2.0, 3.0}.
    """
    return {
        "rho":     _RHO,
        "eta":     _ETA,
        "gamma":   _GAMMA,
        "theta_t": dict(_THETA_T),
    }


# ---------------------------------------------------------------------------
# Call-price grid
# ---------------------------------------------------------------------------

def get_call_prices(
    T: float,
    K_grid: np.ndarray,
    F: float = 1.0,
    r: float = 0.02,
) -> np.ndarray:
    """Return Black-Scholes call prices computed from the eSSVI smile.

    For each strike K in ``K_grid`` the log-moneyness k = ln(K/F) is
    computed, the eSSVI implied vol is looked up, and the Black-Scholes
    price is evaluated at that vol.

    Parameters
    ----------
    T : float
        Maturity in years.  Must be one of the calibrated maturities
        {0.25, 0.5, 1.0, 2.0, 3.0}.
    K_grid : np.ndarray
        Strike grid (strictly positive, same units as *F*).
    F : float, optional
        Forward price.  Defaults to 1.0 (normalised coordinates).
    r : float, optional
        Discount rate for Black-Scholes pricing.  Defaults to 0.02.

    Returns
    -------
    np.ndarray
        Call prices of the same shape as ``K_grid``.

    Raises
    ------
    KeyError
        If *T* is not a calibrated maturity.
    """
    if T not in _THETA_T:
        raise KeyError(
            f"Maturity T={T} not calibrated.  Available: {sorted(_THETA_T.keys())}"
        )
    theta_t = _THETA_T[T]
    K_grid = np.asarray(K_grid, dtype=float)
    k_grid = np.log(K_grid / F)
    iv_grid = essvi_implied_vol(k_grid, T, theta_t, _RHO, _ETA, _GAMMA)
    prices = np.array(
        [bs_call_price(F, K, T, sig, r=r) for K, sig in zip(K_grid, iv_grid)]
    )
    return prices


# ---------------------------------------------------------------------------
# Full-surface accessor
# ---------------------------------------------------------------------------

def get_all_smiles(
    F: float = 1.0,
    r: float = 0.02,
    n_strikes: int = 200,
    k_min: float = -1.5,
    k_max: float = 0.5,
) -> dict:
    """Return eSSVI smile data for all calibrated maturities.

    For each maturity a log-moneyness grid is created, call prices and
    implied vols are computed from the eSSVI parameterisation, and the
    results are stored in a nested dict.

    Parameters
    ----------
    F : float, optional
        Forward price.  Defaults to 1.0.
    r : float, optional
        Discount rate.  Defaults to 0.02.
    n_strikes : int, optional
        Number of strike grid points per maturity.  Defaults to 200.
    k_min : float, optional
        Lower bound of log-moneyness grid.  Defaults to −1.5.
    k_max : float, optional
        Upper bound of log-moneyness grid.  Defaults to 0.5.

    Returns
    -------
    dict
        ``{T: {"K_grid": ..., "C_grid": ..., "iv_grid": ...}}``
        for each T in {0.25, 0.5, 1.0, 2.0, 3.0}.
    """
    smiles = {}
    k_grid = np.linspace(k_min, k_max, n_strikes)
    for T, theta_t in _THETA_T.items():
        K_grid = F * np.exp(k_grid)
        iv_grid = essvi_implied_vol(k_grid, T, theta_t, _RHO, _ETA, _GAMMA)
        C_grid = np.array(
            [bs_call_price(F, K, T, sig, r=r) for K, sig in zip(K_grid, iv_grid)]
        )
        smiles[T] = {"K_grid": K_grid, "C_grid": C_grid, "iv_grid": iv_grid}
    return smiles


# ---------------------------------------------------------------------------
# Target marginal sampling via inverse-CDF
# ---------------------------------------------------------------------------

def get_target_marginals(
    n_samples: int = 100_000,
    seed: int = 42,
    F: float = 1.0,
    r: float = 0.0,
    n_strikes: int = 500,
    k_min: float = -2.0,
    k_max: float = 1.0,
) -> dict:
    """Sample forward-normalised marginals from the eSSVI smile at each maturity.

    The algorithm is:
    1. Build a fine strike grid K on [F·e^{k_min}, F·e^{k_max}].
    2. Compute eSSVI call prices on that grid.
    3. Apply Breeden-Litzenberger (``compute_cdf_from_call_prices``) to
       extract the risk-neutral CDF in x = K/F coordinates.
    4. Draw n_samples uniform samples U ~ U(0, 1) and apply the
       empirical inverse-CDF to obtain samples from the target marginal.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples per maturity.  Defaults to 100 000.
    seed : int, optional
        Random seed for reproducibility.  Defaults to 42.
    F : float, optional
        Forward price (normalisation).  Defaults to 1.0.
    r : float, optional
        Discount rate for call pricing.  Use 0.0 for undiscounted prices
        (consistent with forward measure).  Defaults to 0.0.
    n_strikes : int, optional
        Number of points on the fine strike grid.  Defaults to 500.
    k_min : float, optional
        Lower log-moneyness bound.  Defaults to −2.0.
    k_max : float, optional
        Upper log-moneyness bound.  Defaults to 1.0.

    Returns
    -------
    dict
        ``{T: samples_array}`` where ``samples_array`` has shape
        ``(n_samples,)`` and contains forward-normalised stock prices
        X_T = S_T / F sampled from the eSSVI risk-neutral density.
    """
    rng = np.random.default_rng(seed)
    marginals: dict = {}

    k_fine = np.linspace(k_min, k_max, n_strikes)
    K_fine = F * np.exp(k_fine)

    for T, theta_t in _THETA_T.items():
        # Step 1–2: call prices on fine grid
        iv_fine = essvi_implied_vol(k_fine, T, theta_t, _RHO, _ETA, _GAMMA)
        C_fine = np.array(
            [bs_call_price(F, K, T, sig, r=r) for K, sig in zip(K_fine, iv_fine)]
        )

        # Step 3: Breeden-Litzenberger CDF
        x_grid, density, cdf_vals = compute_cdf_from_call_prices(K_fine, C_fine, F, T)

        # Clamp and normalise CDF to ensure strict monotonicity for inversion
        cdf_vals = np.clip(cdf_vals, 0.0, 1.0)

        # Build inverse CDF (quantile function) via interpolation
        # Ensure the CDF is strictly increasing before passing to interp1d
        # by de-duplicating equal consecutive values
        _, unique_idx = np.unique(cdf_vals, return_index=True)
        x_unique = x_grid[unique_idx]
        cdf_unique = cdf_vals[unique_idx]

        quantile_fn = interp1d(
            cdf_unique,
            x_unique,
            kind="linear",
            bounds_error=False,
            fill_value=(x_unique[0], x_unique[-1]),
        )

        # Step 4: inverse-CDF sampling
        u = rng.uniform(0.0, 1.0, size=n_samples)
        samples = quantile_fn(u)
        marginals[T] = np.asarray(samples, dtype=float)

    return marginals

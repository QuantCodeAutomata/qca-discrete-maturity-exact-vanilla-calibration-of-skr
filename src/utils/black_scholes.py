"""
Black-Scholes pricing and implied-volatility utilities.

All functions operate in *forward-normalised* coordinates where the forward
price F plays the role of the spot (no drift term under the risk-neutral
forward measure).  The default discount rate ``r=0.0`` reflects the
convention used throughout this codebase; pass a non-zero ``r`` when
pricing discounted (spot) call prices.
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Core pricing
# ---------------------------------------------------------------------------

def bs_call_price(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Compute the Black-Scholes (Black-76) European call price.

    In forward-normalised coordinates the formula is the standard
    Black-76 result discounted at rate *r*:

        C = e^{-rT} [ F N(d1) - K N(d2) ]

    where
        d1 = [ln(F/K) + ½ σ² T] / (σ √T),
        d2 = d1 − σ √T.

    Parameters
    ----------
    F : float
        Forward price (or current spot when the model is drift-free).
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    sigma : float
        Annualised implied/Black volatility (must be > 0).
    r : float, optional
        Continuously-compounded discount rate.  Defaults to 0.0 (pure
        forward pricing, no discounting).

    Returns
    -------
    float
        Call option price.  Returns the intrinsic value max(F-K, 0) when
        ``T <= 0`` or ``sigma <= 0``.
    """
    if T <= 0.0 or sigma <= 0.0:
        return float(max(F - K, 0.0))

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return float(np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2)))


def bs_vega(F: float, K: float, T: float, sigma: float) -> float:
    """Compute the Black-Scholes vega (∂C/∂σ) of a European call.

    Vega is the same for calls and puts.  In forward-normalised
    coordinates (r = 0):

        vega = F √T φ(d1)

    where φ is the standard normal PDF and d1 is defined as in
    :func:`bs_call_price`.

    Parameters
    ----------
    F : float
        Forward price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    sigma : float
        Implied volatility (must be > 0).

    Returns
    -------
    float
        Vega of the option.  Returns 0.0 when ``T <= 0`` or ``sigma <= 0``.
    """
    if T <= 0.0 or sigma <= 0.0:
        return 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    return float(F * sqrt_T * norm.pdf(d1))


def bs_call_from_vol(vol: float, F: float, K: float, T: float) -> float:
    """Convenience wrapper: Black-Scholes call price with r=0.

    Equivalent to ``bs_call_price(F, K, T, sigma=vol, r=0.0)``.

    Parameters
    ----------
    vol : float
        Annualised Black volatility.
    F : float
        Forward price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.

    Returns
    -------
    float
        Undiscounted (r = 0) call price.
    """
    return bs_call_price(F, K, T, sigma=vol, r=0.0)


# ---------------------------------------------------------------------------
# Implied volatility inversion
# ---------------------------------------------------------------------------

def bs_implied_vol(
    C: float,
    F: float,
    K: float,
    T: float,
    r: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Invert the Black-Scholes formula via Newton-Raphson to obtain implied vol.

    The iteration starts from the Brenner-Subrahmanyam (1988) ATM
    approximation and then refines with Newton-Raphson steps:

        σ_{n+1} = σ_n − (BS(σ_n) − C) / vega(σ_n)

    A bisection fallback is used when the Newton step would leave the
    valid domain.

    Parameters
    ----------
    C : float
        Observed (or model) call price.
    F : float
        Forward price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float, optional
        Discount rate used in :func:`bs_call_price`.  Defaults to 0.0.
    tol : float, optional
        Convergence tolerance on |BS(σ) − C|.  Defaults to 1e-8.
    max_iter : int, optional
        Maximum Newton-Raphson iterations.  Defaults to 100.

    Returns
    -------
    float
        Implied volatility.  Returns ``np.nan`` if the price is outside
        arbitrage bounds or convergence is not achieved.

    Raises
    ------
    ValueError
        If ``T <= 0``.
    """
    if T <= 0.0:
        raise ValueError("T must be positive for implied-vol inversion.")

    discount = np.exp(-r * T)
    intrinsic = max(discount * (F - K), 0.0)

    # Arbitrage bounds check
    if C < intrinsic - tol or C > discount * F + tol:
        return np.nan

    # Clamp to arbitrage bounds (numerical noise)
    C_clamped = np.clip(C, intrinsic + 1e-12, discount * F - 1e-12)

    # Initial guess: Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2.0 * np.pi / T) * C_clamped / F

    # Newton-Raphson
    for _ in range(max_iter):
        price = bs_call_price(F, K, T, sigma, r=r)
        diff = price - C_clamped
        if abs(diff) < tol:
            return float(sigma)
        vega = bs_vega(F, K, T, sigma)
        if vega < 1e-14:
            break
        sigma -= diff / vega
        sigma = max(sigma, 1e-9)  # keep positive

    # Final check
    if abs(bs_call_price(F, K, T, sigma, r=r) - C_clamped) < tol:
        return float(sigma)
    return np.nan

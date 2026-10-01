"""
Reverse cliquet structured product — pricer and sensitivity analysis.

Payoff at maturity T:

    Π_T = N · [1 + (C₀ − Σ_j min((−R_j)⁺, L))⁺]

where:
    R_j = S_{t_j} / S_{t_{j-1}} − 1   (quarterly returns, j = 1 … n)
    (−R_j)⁺ = max(−R_j, 0)             (down-return, i.e. loss per period)
    L                                   (local loss cap per quarter)
    C₀                                  (aggregate coupon budget)
    N                                   (notional)

Interpretation
--------------
At inception the investor is credited a "coupon budget" of C₀ × N.
Each quarter, if the underlying falls by |R_j|, up to L of the budget
is depleted.  The product pays back the notional N plus any remaining
coupon budget, floored at N.

- Payoff range: [N, N(1 + C₀)].
- Flat path (R_j = 0): aggregate loss = 0 → payoff = N(1 + C₀).
- Extreme down (|R_j| ≥ L each quarter, n × L ≥ C₀): payoff = N.
- Price is monotone *decreasing* in L (larger cap → more depletion → lower price).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Payoff computation
# ---------------------------------------------------------------------------

def compute_reverse_cliquet_payoff(
    S_paths: np.ndarray,
    N:       float = 100.0,
    C0:      float = 0.50,
    L:       float = 0.075,
    t_grid:  Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute reverse cliquet payoffs for a Monte Carlo path array.

    Parameters
    ----------
    S_paths : np.ndarray, shape (n_paths, n_dates)
        Spot prices at quarterly observation dates.  The first column
        is t = 0 (initial price) and subsequent columns are the
        quarterly fixing dates t_1, t_2, …, t_N.
    N : float, optional
        Notional.  Defaults to 100.0.
    C0 : float, optional
        Aggregate coupon budget (as a fraction of N).  Defaults to 0.50.
    L : float, optional
        Local loss cap per quarterly period.  Defaults to 0.075 (7.5 %).
    t_grid : np.ndarray, optional
        Quarterly dates (informational; not used in the computation).
        Defaults to ``np.linspace(0, 3, 13)`` (twelve quarters).

    Returns
    -------
    payoffs : np.ndarray, shape (n_paths,)
        Undiscounted payoff for each path.
    """
    S_paths = np.asarray(S_paths, dtype=float)

    # Quarterly returns  R_j = S_j / S_{j-1} - 1
    S_prev = np.maximum(S_paths[:, :-1], 1e-14)   # (n_paths, n_quarters)
    S_curr = S_paths[:, 1:]                        # (n_paths, n_quarters)
    R      = S_curr / S_prev - 1.0                 # signed quarterly return

    # Per-period loss: min(max(-R_j, 0), L)
    local_loss  = np.minimum(np.maximum(-R, 0.0), L)   # (n_paths, n_quarters)

    # Aggregate loss and residual coupon
    agg_loss    = local_loss.sum(axis=1)                # (n_paths,)
    residual    = np.maximum(C0 - agg_loss, 0.0)

    return N * (1.0 + residual)


# ---------------------------------------------------------------------------
# Monte Carlo pricer
# ---------------------------------------------------------------------------

def price_reverse_cliquet(
    S_paths: np.ndarray,
    r:       float = 0.02,
    T:       float = 3.0,
    N:       float = 100.0,
    C0:      float = 0.50,
    L:       float = 0.075,
) -> Tuple[float, float]:
    """Price a reverse cliquet by discounted Monte Carlo.

    Parameters
    ----------
    S_paths : np.ndarray, shape (n_paths, n_dates)
        Simulated spot paths at quarterly dates (including t = 0).
    r : float, optional
        Continuously-compounded risk-free rate.  Defaults to 0.02.
    T : float, optional
        Product maturity in years.  Defaults to 3.0.
    N : float, optional
        Notional.  Defaults to 100.0.
    C0 : float, optional
        Aggregate coupon budget.  Defaults to 0.50.
    L : float, optional
        Local loss cap per quarter.  Defaults to 0.075.

    Returns
    -------
    price : float
        Monte Carlo estimate of the present value.
    mc_se : float
        Monte Carlo standard error of the price estimate.
    """
    payoffs  = compute_reverse_cliquet_payoff(S_paths, N=N, C0=C0, L=L)
    disc     = np.exp(-r * T)
    pv       = disc * payoffs
    price    = float(np.mean(pv))
    mc_se    = float(np.std(pv) / np.sqrt(len(pv)))
    return price, mc_se


# ---------------------------------------------------------------------------
# Sensitivity to local loss cap L
# ---------------------------------------------------------------------------

def sensitivity_reverse_cliquet(
    S_paths: np.ndarray,
    L_grid:  np.ndarray,
    r:       float = 0.02,
    T:       float = 3.0,
    N:       float = 100.0,
    C0:      float = 0.50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sensitivity of reverse cliquet price to the local loss cap L.

    Prices the same path array at each value of *L_grid*.  Because
    the same random paths are reused the price curve has low noise
    (control-variate-like variance reduction across L values).

    Price is *monotone decreasing* in L: a tighter cap (smaller L)
    limits coupon depletion and raises the product value.

    Parameters
    ----------
    S_paths : np.ndarray, shape (n_paths, n_dates)
        Simulated quarterly paths.
    L_grid : np.ndarray
        Grid of local loss cap values to sweep.
    r : float, optional
        Risk-free rate.  Defaults to 0.02.
    T : float, optional
        Maturity.  Defaults to 3.0.
    N : float, optional
        Notional.  Defaults to 100.0.
    C0 : float, optional
        Coupon budget.  Defaults to 0.50.

    Returns
    -------
    prices : np.ndarray, shape (len(L_grid),)
        Estimated prices for each L in *L_grid*.
    mc_ses : np.ndarray, shape (len(L_grid),)
        Monte Carlo standard errors.
    """
    L_grid = np.asarray(L_grid, dtype=float)
    prices  = np.empty(len(L_grid))
    mc_ses  = np.empty(len(L_grid))

    for k, L in enumerate(L_grid):
        prices[k], mc_ses[k] = price_reverse_cliquet(S_paths, r=r, T=T, N=N, C0=C0, L=L)

    return prices, mc_ses

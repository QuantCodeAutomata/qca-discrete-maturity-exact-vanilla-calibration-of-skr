"""
Memory autocallable structured product — pricer and analytics.

Product mechanics (quarterly observation dates t_j = j × 0.25, j = 1 … n)
---------------------------------------------------------------------------

At each quarterly observation date t_j:

    1. **Autocall check** (first priority):
       If S_{t_j} ≥ autocall_barrier × S₀:
           → Autocall triggered.
           → Pay  N + (pending_j + 1) × c × N  (return of notional plus
             all accumulated coupons including the current period).
           → Product terminates.

    2. **Coupon check** (if not autocalled):
       If S_{t_j} ≥ b × S₀:
           → Coupon condition met.
           → Pay  (pending_j + 1) × c × N  (current coupon plus all
             coupons that had been deferred by the memory feature).
           → Reset pending_j = 0.
       Else:
           → Coupon condition not met.
           → pending_j ← pending_j + 1  (defer coupon to the future).

    3. **At maturity** (t_n = T, if not yet autocalled):
       If S_T ≥ autocall_barrier × S₀:
           → Autocall at maturity: pay N + (pending_n + 1) × c × N.
       Else:
           → Terminal protection:
               if S_T ≥ b_P × S₀:  pay N
               else:                 pay N × S_T / S₀

Notes
-----
- The autocall barrier is typically at par (autocall_barrier = 1.0 ≥ b),
  so autocall also satisfies the coupon condition; the +1 in the coupon
  count reflects the current observation.
- Pending coupons that have never been triggered by maturity (neither via
  autocall nor coupon condition) are *forfeited* — the terminal payment
  is purely capital-based.
- All cash flows are discounted at the continuous rate r.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Main pricer
# ---------------------------------------------------------------------------

def price_memory_autocallable(
    S_paths:         np.ndarray,
    r:               float = 0.02,
    N:               float = 100.0,
    c:               float = 0.05,
    b:               float = 0.60,
    autocall_barrier: float = 1.00,
    b_P:             float = 0.60,
    S0:              float = 1.0,
) -> Tuple[float, float]:
    """Price a memory autocallable structured product by Monte Carlo.

    Parameters
    ----------
    S_paths : np.ndarray, shape (n_paths, n_dates)
        Simulated spot paths at quarterly observation dates (including
        the initial date t = 0 as the first column).
    r : float, optional
        Continuously-compounded risk-free rate.  Defaults to 0.02.
    N : float, optional
        Notional.  Defaults to 100.0.
    c : float, optional
        Quarterly coupon rate (as a fraction of N per period).
        Defaults to 0.05 (5 %).
    b : float, optional
        Coupon barrier (fraction of S₀).  Defaults to 0.60.
    autocall_barrier : float, optional
        Autocall trigger level (fraction of S₀).  Defaults to 1.00.
    b_P : float, optional
        Terminal capital-protection barrier (fraction of S₀).
        Defaults to 0.60.
    S0 : float, optional
        Initial spot price used to normalise barriers.  Defaults to 1.0.

    Returns
    -------
    price : float
        Monte Carlo estimate of the present value.
    mc_se : float
        Monte Carlo standard error.
    """
    S_paths = np.asarray(S_paths, dtype=float)
    n_paths, n_dates = S_paths.shape
    n_quarters = n_dates - 1

    # Time grid: quarterly spacing (t_j = j * dt where dt = (t_N - t_0) / n_quarters)
    T_final = (n_quarters) * 0.25   # total maturity in years
    t_grid  = np.array([j * 0.25 for j in range(n_dates)])

    pv      = np.zeros(n_paths)                 # accumulated present values
    active  = np.ones(n_paths, dtype=bool)      # True = not yet terminated
    pending = np.zeros(n_paths, dtype=np.int64) # deferred coupon periods

    autocall_lv = autocall_barrier * S0
    coupon_lv   = b * S0
    protect_lv  = b_P * S0

    for j in range(1, n_dates):
        t_j     = t_grid[j]
        disc_j  = np.exp(-r * t_j)
        S_j     = S_paths[:, j]
        is_last = (j == n_dates - 1)

        # ---- Autocall check ----
        auto = active & (S_j >= autocall_lv)
        if auto.any():
            total_coup        = (pending[auto] + 1) * c * N
            pv[auto]         += disc_j * (N + total_coup)
            active[auto]      = False
            pending[auto]     = 0

        if is_last:
            # Terminal cash flow for paths still active
            term = active.copy()
            if term.any():
                S_T = S_j[term]
                terminal_pay = np.where(S_T >= protect_lv, N, N * S_T / S0)
                pv[term]    += disc_j * terminal_pay
            active[term] = False

        else:
            # ---- Coupon check (only on non-terminal dates) ----
            coup = active & (S_j >= coupon_lv)
            if coup.any():
                total_coup    = (pending[coup] + 1) * c * N
                pv[coup]     += disc_j * total_coup
                pending[coup] = 0

            # Defer coupon for active paths that missed both barriers
            missed = active & (S_j < coupon_lv)
            pending[missed] += 1

    price = float(np.mean(pv))
    mc_se = float(np.std(pv) / np.sqrt(n_paths))
    return price, mc_se


# ---------------------------------------------------------------------------
# Barrier-sensitivity sweep
# ---------------------------------------------------------------------------

def sensitivity_autocallable(
    S_paths:          np.ndarray,
    b_grid:           np.ndarray,
    r:                float = 0.02,
    N:                float = 100.0,
    c:                float = 0.05,
    autocall_barrier: float = 1.00,
    b_P:              float = 0.60,
    S0:               float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sensitivity of the autocallable price to the coupon barrier b.

    The same Monte Carlo paths are repriced at each b in *b_grid*.

    Parameters
    ----------
    S_paths : np.ndarray, shape (n_paths, n_dates)
        Simulated quarterly paths.
    b_grid : np.ndarray
        Grid of coupon barrier levels (fractions of S₀) to sweep.
    r : float, optional
        Risk-free rate.  Defaults to 0.02.
    N : float, optional
        Notional.  Defaults to 100.0.
    c : float, optional
        Quarterly coupon rate.  Defaults to 0.05.
    autocall_barrier : float, optional
        Autocall trigger.  Defaults to 1.00.
    b_P : float, optional
        Terminal protection barrier.  Defaults to 0.60.
    S0 : float, optional
        Initial spot.  Defaults to 1.0.

    Returns
    -------
    prices : np.ndarray, shape (len(b_grid),)
        Price estimates.
    mc_ses : np.ndarray, shape (len(b_grid),)
        Monte Carlo standard errors.
    """
    b_grid = np.asarray(b_grid, dtype=float)
    prices = np.empty(len(b_grid))
    mc_ses = np.empty(len(b_grid))

    for k, b_val in enumerate(b_grid):
        prices[k], mc_ses[k] = price_memory_autocallable(
            S_paths,
            r=r, N=N, c=c,
            b=b_val,
            autocall_barrier=autocall_barrier,
            b_P=b_P, S0=S0,
        )

    return prices, mc_ses


# ---------------------------------------------------------------------------
# Path-wise statistics
# ---------------------------------------------------------------------------

def compute_autocall_statistics(
    S_paths:          np.ndarray,
    b:                float = 0.60,
    autocall_barrier: float = 1.00,
    b_P:              float = 0.60,
    S0:               float = 1.0,
) -> Dict[str, object]:
    """Compute path-wise statistics for the memory autocallable.

    Tracks, for each quarterly observation date:

    - **Autocall probability**: fraction of paths that autocall at that quarter
      (conditional on being active).
    - **Coupon hit frequency**: fraction of *all* paths where the coupon
      condition is met at that quarter (regardless of active status).
    - **Terminal barrier breach**: fraction of all paths where S_T < b_P × S₀.

    Parameters
    ----------
    S_paths : np.ndarray, shape (n_paths, n_dates)
        Simulated quarterly paths (first column = t = 0).
    b : float, optional
        Coupon barrier.  Defaults to 0.60.
    autocall_barrier : float, optional
        Autocall trigger.  Defaults to 1.00.
    b_P : float, optional
        Terminal protection barrier.  Defaults to 0.60.
    S0 : float, optional
        Initial spot.  Defaults to 1.0.

    Returns
    -------
    dict with keys:

    ``'autocall_prob_per_quarter'`` : np.ndarray, shape (n_quarters,)
        Fraction of *active* paths autocalled at each quarter.
    ``'coupon_hit_freq_per_quarter'`` : np.ndarray, shape (n_quarters,)
        Fraction of *all* paths meeting the coupon condition at each quarter.
    ``'terminal_barrier_breach_freq'`` : float
        Fraction of *all* paths where S_T < b_P × S₀.
    """
    S_paths = np.asarray(S_paths, dtype=float)
    n_paths, n_dates = S_paths.shape
    n_quarters = n_dates - 1

    autocall_lv = autocall_barrier * S0
    coupon_lv   = b * S0
    protect_lv  = b_P * S0

    autocall_prob   = np.zeros(n_quarters)
    coupon_hit_freq = np.zeros(n_quarters)

    active = np.ones(n_paths, dtype=bool)

    for j in range(1, n_dates):
        S_j = S_paths[:, j]

        # Autocall probability (fraction of active paths)
        n_active = active.sum()
        if n_active > 0:
            auto = active & (S_j >= autocall_lv)
            autocall_prob[j - 1] = auto.sum() / n_active
            active[auto] = False

        # Coupon hit frequency (fraction of all paths)
        coupon_hit_freq[j - 1] = (S_j >= coupon_lv).mean()

    S_T = S_paths[:, -1]
    terminal_breach = float((S_T < protect_lv).mean())

    return {
        "autocall_prob_per_quarter":   autocall_prob,
        "coupon_hit_freq_per_quarter": coupon_hit_freq,
        "terminal_barrier_breach_freq": terminal_breach,
    }

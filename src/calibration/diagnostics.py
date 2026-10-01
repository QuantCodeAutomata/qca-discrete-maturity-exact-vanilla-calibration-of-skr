"""
Calibration diagnostics — implied-vol repricing and smile-comparison plots.

All plots use the non-interactive Agg backend and are written to disk.

Provides
--------
- ``compute_implied_vol_from_mc``  : convert MC call prices to an IV smile.
- ``plot_smile_comparison``        : overlay target and model IV smiles.
- ``plot_convergence_diagnostics`` : Sinkhorn convergence curves.
- ``plot_transport_maps``          : monotone transport maps g per interval.
- ``print_calibration_summary``    : per-maturity calibration summary table.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from src.utils.black_scholes import bs_implied_vol


# ---------------------------------------------------------------------------
# Implied vol from MC call prices
# ---------------------------------------------------------------------------

def compute_implied_vol_from_mc(
    T:              float,
    K_grid:         np.ndarray,
    call_prices_mc: np.ndarray,
    F:              float = 1.0,
    r:              float = 0.02,
) -> np.ndarray:
    """Convert MC call prices to implied vols via Black-Scholes inversion.

    For each strike K in ``K_grid`` and the corresponding MC call price,
    the Black-Scholes implied volatility is found via Newton-Raphson.

    Parameters
    ----------
    T : float
        Maturity in years.
    K_grid : np.ndarray
        Strike grid.
    call_prices_mc : np.ndarray
        MC call prices corresponding to ``K_grid``.
    F : float, optional
        Forward price.  Defaults to 1.0.
    r : float, optional
        Discount rate.  Defaults to 0.02.

    Returns
    -------
    iv_grid : np.ndarray, shape (len(K_grid),)
        Implied volatility for each strike.  Entries are ``np.nan``
        where inversion failed.
    """
    K_grid         = np.asarray(K_grid,         dtype=float)
    call_prices_mc = np.asarray(call_prices_mc, dtype=float)
    iv_grid        = np.empty(len(K_grid))

    for j, (K, price) in enumerate(zip(K_grid, call_prices_mc)):
        iv_grid[j] = bs_implied_vol(float(price), F, float(K), T, r=r)

    return iv_grid


# ---------------------------------------------------------------------------
# Smile-comparison plot
# ---------------------------------------------------------------------------

def plot_smile_comparison(
    maturities:    List[float],
    target_smiles: Dict,
    model_smiles:  Dict,
    save_path:     str,
    title:         str = 'SKR Vanilla Smile Repricing',
) -> None:
    """Plot target vs calibrated model smiles at all maturities.

    Creates a grid of sub-plots — one per maturity — overlaying the target
    eSSVI smile with each model's repriced smile.  Comparable to paper
    Figure 5.

    Parameters
    ----------
    maturities : list of float
        Calibration maturities to plot.
    target_smiles : dict {T: {'K_grid': ..., 'iv_grid': ...}}
        Target smile data (e.g. eSSVI).  Keys must include ``'K_grid'``
        (strike grid) and ``'iv_grid'`` (implied vols).
    model_smiles : dict {model_name: {T: {'K_grid': ..., 'iv_grid': ...}}}
        Model-repriced smile data for each model and maturity.
    save_path : str
        File path to save the figure.
    title : str, optional
        Figure suptitle.  Defaults to ``'SKR Vanilla Smile Repricing'``.
    """
    n_mats  = len(maturities)
    n_cols  = min(3, n_mats)
    n_rows  = (n_mats + n_cols - 1) // n_cols
    ls_list = ['-', '--', '-.', ':']

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=13)

    model_names = list(model_smiles.keys())

    for i, T in enumerate(maturities):
        row, col = divmod(i, n_cols)
        ax       = axes[row][col]

        # Target smile
        if T in target_smiles:
            tgt = target_smiles[T]
            K_t = np.asarray(tgt.get('K_grid', []), dtype=float)
            iv_t = np.asarray(tgt.get('iv_grid', []), dtype=float)
            if K_t.size > 0 and iv_t.size > 0:
                k_plot = np.log(K_t) if K_t.min() > 0 else K_t
                ax.plot(k_plot, iv_t * 100.0, 'k-', lw=2, label='Target', zorder=5)

        # Model smiles
        for m_idx, m_name in enumerate(model_names):
            m_data = model_smiles[m_name]
            if T not in m_data:
                continue
            m_smile = m_data[T]
            K_m  = np.asarray(m_smile.get('K_grid', []), dtype=float)
            iv_m = np.asarray(m_smile.get('iv_grid', []), dtype=float)
            if K_m.size == 0 or iv_m.size == 0:
                continue
            k_plot = np.log(K_m) if K_m.min() > 0 else K_m
            valid  = ~np.isnan(iv_m)
            ax.plot(
                k_plot[valid], iv_m[valid] * 100.0,
                ls_list[m_idx % len(ls_list)],
                lw=1.5, label=m_name,
            )

        ax.set_title(f'T = {T:.2f}y', fontsize=10)
        ax.set_xlabel('log-moneyness  k = ln(K/F)')
        ax.set_ylabel('IV (%)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused sub-plots
    for i in range(n_mats, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convergence diagnostics plot
# ---------------------------------------------------------------------------

def plot_convergence_diagnostics(
    diagnostics_history: List[Dict],
    interval_idx:        int,
    save_path:           str,
) -> None:
    """Plot convergence of Sinkhorn iterations for one calibration interval.

    Produces a two-panel figure:

    - Left panel:  Wasserstein-1 distance vs. iteration (log scale).
    - Right panel: Absolute martingale residual vs. iteration (log scale).

    Parameters
    ----------
    diagnostics_history : list of dict
        Per-iteration diagnostics from
        :meth:`SKRCalibrator._calibrate_interval`.  Each dict must contain
        at least ``'wasserstein_1'`` and ``'martingale_residual'``.
    interval_idx : int
        Interval index used in the figure title.
    save_path : str
        File path to save the figure.
    """
    iters     = np.arange(1, len(diagnostics_history) + 1)
    w1_vals   = [d.get('wasserstein_1',       float('nan')) for d in diagnostics_history]
    mart_vals = [abs(d.get('martingale_residual', float('nan'))) for d in diagnostics_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f'SKR Sinkhorn Convergence — Interval {interval_idx}', fontsize=11
    )

    ax1.semilogy(iters, w1_vals, 'b-o', markersize=3, lw=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Wasserstein-1 distance')
    ax1.set_title('Marginal match (W₁)')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(iters, mart_vals, 'r-s', markersize=3, lw=1.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('|Martingale residual|')
    ax2.set_title('Martingale constraint')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Transport map plot
# ---------------------------------------------------------------------------

def plot_transport_maps(
    transport_maps: Dict,
    save_path:      str,
) -> None:
    """Plot monotone transport maps g for each interval and model.

    Parameters
    ----------
    transport_maps : dict
        Nested dict ``{interval_idx: {model_name: (y_grid, g_values)}}``.
        Each leaf is a 2-tuple of 1-D arrays.
    save_path : str
        File path to save the figure.
    """
    intervals   = sorted(transport_maps.keys())
    n_intervals = len(intervals)
    if n_intervals == 0:
        return

    n_cols  = min(3, n_intervals)
    n_rows  = (n_intervals + n_cols - 1) // n_cols
    ls_list = ['-', '--', '-.', ':']

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    fig.suptitle('Monotone Transport Maps g', fontsize=12)

    for i, interval in enumerate(intervals):
        row, col = divmod(i, n_cols)
        ax       = axes[row][col]
        models   = transport_maps[interval]

        for m_idx, (m_name, g_data) in enumerate(models.items()):
            y_grid, g_vals = g_data
            ax.plot(
                y_grid, g_vals,
                ls_list[m_idx % len(ls_list)],
                lw=1.5, label=m_name,
            )

        ax.set_title(f'Interval {interval}')
        ax.set_xlabel('y  (shifted reference)')
        ax.set_ylabel('g(y)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(n_intervals, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Calibration summary
# ---------------------------------------------------------------------------

def print_calibration_summary(
    all_diagnostics: Dict,
    model_name:      str,
) -> None:
    """Print interval-by-interval calibration summary table.

    For each calibrated maturity, reports: number of Sinkhorn iterations,
    final Wasserstein-1 distance, and final martingale residual.

    Parameters
    ----------
    all_diagnostics : dict {T: list of diagnostic dicts}
        Full diagnostics returned by :meth:`SKRCalibrator.calibrate`
        (``result['all_diagnostics']``).
    model_name : str
        Model label printed in the table header.
    """
    header = (
        f"  {'Maturity':>10}  {'N_iter':>7}  "
        f"{'W1 (final)':>12}  {'Mart. res.':>12}"
    )
    sep = '-' * len(header)

    print(f"\n{'=' * len(header)}")
    print(f"  SKR Calibration Summary — {model_name}")
    print(sep)
    print(header)
    print(sep)

    for T in sorted(all_diagnostics.keys()):
        hist = all_diagnostics[T]
        if not hist:
            continue
        n_iter = len(hist)
        final  = hist[-1]
        w1     = final.get('wasserstein_1',      float('nan'))
        mart   = final.get('martingale_residual', float('nan'))
        print(f"  T={T:<8.3f}  {n_iter:>7d}  {w1:>12.6f}  {mart:>12.6f}")

    print(sep)

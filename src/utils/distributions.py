"""
Distribution utilities for marginal-law handling in the SKR calibration
pipeline.

Includes an empirical distribution class, monotone-transport tools, a
Breeden-Litzenberger density extractor, and the 1-Wasserstein distance.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid


# ---------------------------------------------------------------------------
# Empirical distribution
# ---------------------------------------------------------------------------

class EmpiricalDistribution:
    """Empirical CDF and quantile function built from a finite sample.

    The class sorts the input samples once on construction and exposes
    vectorised CDF and quantile (inverse-CDF) methods backed by linear
    interpolation.

    Parameters
    ----------
    samples : np.ndarray
        1-D array of real-valued observations.  Need not be sorted.
    """

    def __init__(self, samples: np.ndarray) -> None:
        """Sort and store samples; pre-compute probability grid.

        Parameters
        ----------
        samples : np.ndarray
            Raw sample array of shape ``(n,)``.
        """
        self._sorted = np.sort(np.asarray(samples, dtype=float).ravel())
        n = len(self._sorted)
        # Midpoint probability grid: p_i = (i + 0.5) / n avoids 0 and 1
        self._probs = (np.arange(n) + 0.5) / n

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the empirical CDF at each point in *x*.

        Uses linear interpolation between the empirical jump points.
        Values outside the sample range are clamped to [0, 1].

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the CDF.

        Returns
        -------
        np.ndarray
            CDF values in [0, 1] of the same shape as *x*.
        """
        x = np.asarray(x, dtype=float)
        cdf_fn = interp1d(
            self._sorted,
            self._probs,
            kind="linear",
            bounds_error=False,
            fill_value=(0.0, 1.0),
        )
        return cdf_fn(x)

    def quantile(self, p: np.ndarray) -> np.ndarray:
        """Evaluate the empirical quantile (inverse-CDF) at probabilities *p*.

        Uses linear interpolation between the empirical quantile points.
        Probabilities outside [0, 1] are clamped.

        Parameters
        ----------
        p : np.ndarray
            Probability values in [0, 1].

        Returns
        -------
        np.ndarray
            Quantile values of the same shape as *p*.
        """
        p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        q_fn = interp1d(
            self._probs,
            self._sorted,
            kind="linear",
            bounds_error=False,
            fill_value=(self._sorted[0], self._sorted[-1]),
        )
        return q_fn(p)

    def mean(self) -> float:
        """Return the sample mean.

        Returns
        -------
        float
        """
        return float(self._sorted.mean())

    def samples(self) -> np.ndarray:
        """Return the sorted sample array.

        Returns
        -------
        np.ndarray
            1-D sorted array of shape ``(n,)``.
        """
        return self._sorted.copy()


# ---------------------------------------------------------------------------
# Monotone transport
# ---------------------------------------------------------------------------

def monotone_transport_map(
    source_samples: np.ndarray,
    target_quantile_fn,
) -> np.ndarray:
    """Compute the (unique) monotone increasing transport map g: source → target.

    The optimal transport map between two one-dimensional distributions μ
    (source) and ν (target) under any strictly convex cost is the monotone
    rearrangement:

        g = Q_ν ∘ F_μ,

    i.e. the composition of the source CDF with the target quantile
    function.  This function evaluates *g* at each point in
    ``source_samples``.

    Parameters
    ----------
    source_samples : np.ndarray
        1-D array of samples drawn from the source distribution μ.
    target_quantile_fn : callable
        Callable that maps an array of probabilities p ∈ [0, 1] to the
        corresponding quantiles of the target distribution ν.  Typically
        obtained from :func:`compute_target_quantile_fn` or the
        ``quantile`` method of :class:`EmpiricalDistribution`.

    Returns
    -------
    np.ndarray
        Array of the same shape as ``source_samples`` containing g(xᵢ)
        for each source sample xᵢ.  The result is the push-forward of the
        source empirical measure under g, matching the target marginal.
    """
    source = np.asarray(source_samples, dtype=float).ravel()
    source_dist = EmpiricalDistribution(source)
    # Evaluate source CDF at each original (unsorted) sample
    probs = source_dist.cdf(source)
    return np.asarray(target_quantile_fn(probs), dtype=float)


def compute_target_quantile_fn(target_samples: np.ndarray):
    """Build a callable quantile function from a finite target sample.

    Parameters
    ----------
    target_samples : np.ndarray
        1-D array of samples from the target distribution.

    Returns
    -------
    callable
        Function ``q(p)`` mapping probabilities in [0, 1] to quantile
        values, backed by linear interpolation over the sorted sample.
    """
    dist = EmpiricalDistribution(target_samples)
    return dist.quantile


# ---------------------------------------------------------------------------
# Breeden-Litzenberger
# ---------------------------------------------------------------------------

def compute_cdf_from_call_prices(
    K_grid: np.ndarray,
    C_grid: np.ndarray,
    F: float,
    T: float,
) -> tuple:
    """Extract the risk-neutral density and CDF from a call-price surface.

    Implements the Breeden-Litzenberger (1978) formula:

        p(K) = e^{rT} ∂²C/∂K²,

    here evaluated with r = 0 (forward-normalised pricing where C already
    contains the discount factor).  The second derivative is approximated
    by central finite differences on the *K*-grid, followed by a density
    normalisation step to enforce ∫ p dx = 1.

    Coordinates are converted to forward-normalised log-space: x = K / F.

    Parameters
    ----------
    K_grid : np.ndarray
        Strike grid (must be strictly increasing, length ≥ 5).
    C_grid : np.ndarray
        Call prices corresponding to ``K_grid`` (same length).
    F : float
        Forward price used to normalise strikes to x = K / F.
    T : float
        Time to expiry (used for annotation; not needed in r=0 formula).

    Returns
    -------
    tuple
        ``(x_grid, density, cdf)`` where:

        - ``x_grid`` is the forward-normalised strike grid x = K / F of
          shape ``(m,)`` (interior points only, excluding boundary);
        - ``density`` is the risk-neutral probability density p(x) of
          shape ``(m,)``;
        - ``cdf`` is the cumulative distribution function F(x) = P(X ≤ x)
          of shape ``(m,)``.
    """
    K_grid = np.asarray(K_grid, dtype=float)
    C_grid = np.asarray(C_grid, dtype=float)

    # Second derivative via central finite differences on interior points
    dK = np.diff(K_grid)
    # Approximate second derivative at each interior node
    d2C_dK2 = np.zeros(len(K_grid))
    for i in range(1, len(K_grid) - 1):
        dK_left = K_grid[i] - K_grid[i - 1]
        dK_right = K_grid[i + 1] - K_grid[i]
        dK_avg = 0.5 * (dK_left + dK_right)
        d2C_dK2[i] = (
            (C_grid[i + 1] - C_grid[i]) / dK_right
            - (C_grid[i] - C_grid[i - 1]) / dK_left
        ) / dK_avg

    # Use interior points only
    interior = slice(1, len(K_grid) - 1)
    K_int = K_grid[interior]
    density_raw = np.maximum(d2C_dK2[interior], 0.0)

    # Normalise: ∫ p(K) dK = 1
    norm_const = np.trapz(density_raw, K_int)
    if norm_const > 1e-14:
        density_raw = density_raw / norm_const

    # Convert to forward-normalised coordinates x = K / F
    x_grid = K_int / F
    # Density in x-space: p_x(x) = p_K(K) * F  (change of variables K=xF)
    density = density_raw * F

    # Re-normalise in x-space (should already be 1 but guard against numerics)
    norm_x = np.trapz(density, x_grid)
    if norm_x > 1e-14:
        density = density / norm_x

    # CDF via cumulative trapezoid
    cdf_vals = cumulative_trapezoid(density, x_grid, initial=0.0)
    cdf_vals = np.clip(cdf_vals, 0.0, 1.0)

    return x_grid, density, cdf_vals


# ---------------------------------------------------------------------------
# Wasserstein distance
# ---------------------------------------------------------------------------

def wasserstein1_distance(
    samples1: np.ndarray,
    samples2: np.ndarray,
) -> float:
    """Compute the 1-Wasserstein (earth-mover) distance between two sample sets.

    For one-dimensional distributions the 1-Wasserstein distance admits the
    closed-form representation

        W₁(μ, ν) = ∫₀¹ |Q_μ(p) − Q_ν(p)| dp,

    which, for empirical measures, is computed by sorting both sample
    arrays, interpolating the quantile functions onto a common uniform
    grid, and integrating the absolute quantile difference.

    Parameters
    ----------
    samples1 : np.ndarray
        1-D array of samples from distribution μ.
    samples2 : np.ndarray
        1-D array of samples from distribution ν.  Need not be the same
        length as ``samples1``.

    Returns
    -------
    float
        1-Wasserstein distance W₁(μ, ν) ≥ 0.
    """
    s1 = np.sort(np.asarray(samples1, dtype=float).ravel())
    s2 = np.sort(np.asarray(samples2, dtype=float).ravel())

    n1, n2 = len(s1), len(s2)
    p1 = (np.arange(n1) + 0.5) / n1
    p2 = (np.arange(n2) + 0.5) / n2

    # Common probability grid for integration
    p_common = np.linspace(0.0, 1.0, max(n1, n2, 1000))

    q1 = interp1d(p1, s1, kind="linear", bounds_error=False,
                  fill_value=(s1[0], s1[-1]))(p_common)
    q2 = interp1d(p2, s2, kind="linear", bounds_error=False,
                  fill_value=(s2[0], s2[-1]))(p_common)

    return float(np.trapz(np.abs(q1 - q2), p_common))

"""
Bass local-volatility model — calibration and Monte Carlo simulator.

The Bass LV model is a Markov martingale (local volatility) constructed
via optimal transport to match a given sequence of target marginals
μ₁, …, μ_N at discrete maturities T₁ < … < T_N.

Construction
------------
At each maturity T_i the monotone Bass map is:
    g_i = F_{μ_i}^{-1} ∘ Φ_{N(0, T_i)}

where Φ_{N(0, T_i)} is the CDF of N(0, T_i) and F_{μ_i}^{-1} is the
quantile function of the target marginal.  Thus X_{T_i} = g_i(W_{T_i})
where W_t is a standard Brownian motion.

Between calibration maturities, price paths are constructed by simulating
the Brownian motion at all time steps and linearly interpolating the
transported X values.

Reference
---------
Bass, R. F. (1983), "Skorokhod embedding via stochastic integrals",
in *Seminar on Probability XVII*, Lecture Notes in Mathematics 986.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, List


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class BassLVModel:
    """Bass Local Volatility model.

    At each maturity T_i, uses monotone map g_i from N(0, T_i)-distributed
    W_{T_i} to target marginal μ_i.  Between calibration maturities,
    interpolates.  Since this is a local-vol-type model, the state is just
    (X_t) without latent factors.

    Parameters
    ----------
    r : float, optional
        Discount rate.  Defaults to 0.02.
    """

    def __init__(self, r: float = 0.02) -> None:
        self.r                    = float(r)
        self.calibrated_maps:     Dict[float, Dict[str, np.ndarray]] = {}
        self.calibrated_maturities: List[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_map(self, T: float, w: np.ndarray) -> np.ndarray:
        """Apply calibrated transport map g_T(w) via linear interpolation.

        Parameters
        ----------
        T : float
            Calibration maturity whose map to apply.
        w : np.ndarray
            Brownian-motion values at time T.

        Returns
        -------
        np.ndarray
            Transported values g_T(w).
        """
        m = self.calibrated_maps[T]
        return np.interp(w, m['w_sorted'], m['x_sorted'])

    def _invert_map(self, T: float, x: np.ndarray) -> np.ndarray:
        """Invert calibrated transport map g_T^{-1}(x).

        Parameters
        ----------
        T : float
            Calibration maturity whose inverse map to apply.
        x : np.ndarray
            Price values.

        Returns
        -------
        np.ndarray
            Latent Brownian-motion values w = g_T^{-1}(x).
        """
        m = self.calibrated_maps[T]
        return np.interp(x, m['x_sorted'], m['w_sorted'])

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_maturity(
        self,
        T:              float,
        target_samples: np.ndarray,
        n_paths:        int = 100_000,
        seed:           int = 42,
    ) -> dict:
        """Calibrate Bass map at maturity T.

        The Bass map g_T = F_{μ}^{-1} ∘ Φ_{N(0, T)} is computed by drawing
        n_paths reference samples W_T ~ N(0, T) and aligning their sorted
        order with the sorted target samples.

        Parameters
        ----------
        T : float
            Maturity to calibrate.
        target_samples : np.ndarray
            Samples from the target marginal μ_T.
        n_paths : int, optional
            Number of reference BM paths.  Defaults to 100 000.
        seed : int, optional
            Random seed.  Defaults to 42.

        Returns
        -------
        dict
            Calibration info: T, n_paths, map_min, map_max.
        """
        rng   = np.random.default_rng(seed)
        w_ref = rng.normal(0.0, np.sqrt(T), n_paths)

        w_sorted = np.sort(w_ref)

        tgt   = np.asarray(target_samples, dtype=float)
        n_tgt = len(tgt)
        if n_tgt == n_paths:
            x_sorted = np.sort(tgt)
        else:
            # Re-sample target to n_paths equi-spaced quantiles
            p_src = (np.arange(n_paths) + 0.5) / n_paths
            p_tgt = (np.arange(n_tgt)   + 0.5) / n_tgt
            x_sorted = np.interp(p_src, p_tgt, np.sort(tgt))

        self.calibrated_maps[T] = {'w_sorted': w_sorted, 'x_sorted': x_sorted}
        if T not in self.calibrated_maturities:
            self.calibrated_maturities.append(T)
            self.calibrated_maturities.sort()

        return {
            'T':       T,
            'n_paths': n_paths,
            'map_min': float(x_sorted[0]),
            'map_max': float(x_sorted[-1]),
        }

    # ------------------------------------------------------------------
    # Path simulation
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        T:       float,
        n_paths: int,
        n_steps: int,
        seed:    int = 42,
    ) -> np.ndarray:
        """Simulate Bass LV paths up to maturity T at n_steps+1 time points.

        Generates a Brownian motion at all time steps.  At each calibration
        maturity, the transport map g_T is applied.  Between calibration
        maturities, X values are linearly interpolated in time (Brownian
        bridge interpolation of the latent W paths).

        Parameters
        ----------
        T : float
            Final maturity.
        n_paths : int
            Number of Monte Carlo paths.
        n_steps : int
            Number of time steps; total time points = n_steps + 1.
        seed : int, optional
            Random seed.  Defaults to 42.

        Returns
        -------
        X_paths : np.ndarray, shape (n_paths, n_steps+1)
            Simulated price paths.

        Note
        ----
        For Bass LV there is no latent state — simpler than SKR models.
        """
        rng    = np.random.default_rng(seed)
        t_grid = np.linspace(0.0, T, n_steps + 1)
        dt     = T / n_steps

        # Full Brownian motion at all time steps
        dW = rng.normal(0.0, np.sqrt(dt), (n_paths, n_steps))
        W  = np.zeros((n_paths, n_steps + 1))
        W[:, 1:] = np.cumsum(dW, axis=1)

        X = np.zeros((n_paths, n_steps + 1))
        X[:, 0] = 1.0  # S_0 = 1

        # Restrict to calibrated maturities within [0, T]
        mats = sorted(m for m in self.calibrated_maturities if m <= T + 1e-10)

        # First pass: apply transport map at each calibration maturity
        mat_to_col: Dict[float, int] = {}
        for T_cal in mats:
            j = int(np.argmin(np.abs(t_grid - T_cal)))
            X[:, j]      = self._apply_map(T_cal, W[:, j])
            mat_to_col[T_cal] = j

        # Second pass: linear-in-time interpolation for non-calibration steps
        # Boundaries: t=0 (X=1) plus each calibration maturity
        boundaries = [0.0] + mats
        for j in range(1, n_steps + 1):
            t = t_grid[j]
            if any(np.isclose(t, m) for m in mats):
                continue  # already set above

            b_idx = int(np.searchsorted(boundaries, t, side='right')) - 1
            b_idx = np.clip(b_idx, 0, len(boundaries) - 1)
            T_lo  = boundaries[b_idx]
            j_lo  = 0 if T_lo == 0.0 else mat_to_col[T_lo]
            X_lo  = np.ones(n_paths) if T_lo == 0.0 else X[:, j_lo]

            if b_idx + 1 < len(boundaries):
                T_hi = boundaries[b_idx + 1]
                X_hi = X[:, mat_to_col[T_hi]]
            else:
                T_hi = T_lo + 1.0   # extrapolate flat
                X_hi = X_lo

            alpha   = (t - T_lo) / max(T_hi - T_lo, 1e-14)
            X[:, j] = (1.0 - alpha) * X_lo + alpha * X_hi

        return X

    def simulate_joint_from_state(
        self,
        T_start:  float,
        T_end:    float,
        x_states: np.ndarray,
        seed:     int = 42,
    ) -> np.ndarray:
        """Simulate from given x_states at T_start to T_end.

        Inverts the calibrated transport map at T_start to recover the
        latent Brownian values, simulates a BM increment over
        [T_start, T_end], then applies the calibrated map at T_end.

        Parameters
        ----------
        T_start : float
            Starting time (should be a calibrated maturity).
        T_end : float
            Ending time (should be a calibrated maturity).
        x_states : np.ndarray, shape (n_states,)
            X values at T_start.
        seed : int, optional
            Random seed.  Defaults to 42.

        Returns
        -------
        X_end : np.ndarray, shape (n_states,)
            X values at T_end.
        """
        rng      = np.random.default_rng(seed)
        x_states = np.asarray(x_states, dtype=float).ravel()
        n_states = len(x_states)

        # Recover latent BM values at T_start
        if T_start in self.calibrated_maps:
            w_start = self._invert_map(T_start, x_states)
        else:
            # No calibrated map — use log as rough approximation
            w_start = np.log(np.maximum(x_states, 1e-10))

        # BM increment over [T_start, T_end]
        dW    = rng.normal(0.0, np.sqrt(T_end - T_start), n_states)
        w_end = w_start + dW

        # Apply transport map at T_end
        if T_end in self.calibrated_maps:
            x_end = self._apply_map(T_end, w_end)
        else:
            x_end = w_end  # fallback: return raw BM

        return x_end

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price_call_mc(
        self,
        T:       float,
        K_grid:  np.ndarray,
        n_paths: int = 100_000,
        n_steps: int = 50,
        seed:    int = 42,
    ) -> np.ndarray:
        """Price European calls at maturity T via Monte Carlo.

        Parameters
        ----------
        T : float
            Maturity.  Should be a calibrated maturity.
        K_grid : np.ndarray
            Strike grid.
        n_paths : int, optional
            Number of MC paths.  Defaults to 100 000.
        n_steps : int, optional
            Number of time steps.  Defaults to 50.
        seed : int, optional
            Random seed.  Defaults to 42.

        Returns
        -------
        np.ndarray
            Discounted call prices of shape (len(K_grid),).
        """
        K_grid = np.asarray(K_grid, dtype=float)
        paths  = self.simulate_paths(T, n_paths, n_steps, seed=seed)
        S_T    = paths[:, -1]
        disc   = np.exp(-self.r * T)
        return np.array(
            [disc * float(np.mean(np.maximum(S_T - K, 0.0))) for K in K_grid]
        )

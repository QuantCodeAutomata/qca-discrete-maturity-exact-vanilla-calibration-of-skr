"""
SKR (Sequential Kernel Reconstruction) calibration.

Calibrates a process X to match prescribed marginals μ₀, …, μ_N at
maturities T₀, …, T_N while preserving reference model dynamics.

Algorithm
---------
For each interval [T_i, T_{i+1}]:

  1.  Draw one reference transition per path:
          Y_j^{i+1} ~ γ_{R_j^i}  (= Law(X_{T_{i+1}} | R_{T_i} = R_j^i))

  2.  Iterate the martingale Sinkhorn fixed-point until convergence:

        a.  Shifted samples:  z_j = a_j + Y_j^{i+1}

        b.  Transport map:    g_k = F_{μ_{i+1}}^{-1} ∘ F_{ρ^a}
                              (monotone rearrangement of z to target)

        c.  Bin states (X_{T_i}, L_{T_i}) into n_bins groups.
            Within each bin b, approximate the conditional expectation:
                m_b(a) = E[g_k(a + Y) | R_0 ∈ bin b]
                       ≈ (1/|b|) Σ_{j ∈ b}  g_k(a + Y_j)
            and solve for each path's shift a_j such that m_{b(j)}(a_j) = X_j.

        d.  Recenter: a_{k+1} ← a_{k+1} − mean(a_{k+1})

        e.  Convergence: check W₁(g_k(a+Y), μ_{i+1}) < tol_marginal
                         and |E[X] − E[κ_i]| < tol_martingale.

  3.  Calibrated terminal prices:  X_{T_{i+1}} = g(a + Y)
      Propagated latent state:     L_{T_{i+1}} = L obtained in step 1.

Reference
---------
Sequential Kernel Reconstruction (SKR) paper (2024).
# Custom – Context7 found no library equivalent (SKR paper)
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from typing import Callable, Dict, List, Optional, Tuple
import time

try:
    from sklearn.cluster import KMeans
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# SKR calibrator
# ---------------------------------------------------------------------------

class SKRCalibrator:
    """Sequential Kernel Reconstruction (SKR) calibration.

    Calibrates a process X to match prescribed marginals μ₀, …, μ_N at
    maturities T₀, …, T_N while preserving reference model dynamics.

    Based on the martingale Sinkhorn fixed-point algorithm.
    # Custom – Context7 found no library equivalent (SKR paper)

    Parameters
    ----------
    reference_model : HestonModel, Model3_2, or Bergomi1FModel
        Reference stochastic-volatility model.  Must implement
        ``simulate_joint_from_state(T_start, T_end, x_states,
        aux_states, n_steps, seed)``.
    r : float, optional
        Discount rate.  Defaults to 0.02.
    max_iter : int, optional
        Maximum Sinkhorn iterations per interval.  Defaults to 100.
    tol_marginal : float, optional
        Wasserstein-1 convergence tolerance.  Defaults to 1e-3.
    tol_martingale : float, optional
        Martingale residual convergence tolerance.  Defaults to 1e-4.
    n_paths : int, optional
        Number of Monte Carlo paths.  Defaults to 100 000.
    n_steps_per_interval : int, optional
        Number of inner simulation steps per calibration interval.
        Defaults to 20.
    verbose : bool, optional
        Print per-iteration convergence info.  Defaults to True.
    """

    def __init__(
        self,
        reference_model,
        r:                    float = 0.02,
        max_iter:             int   = 100,
        tol_marginal:         float = 1e-3,
        tol_martingale:       float = 1e-4,
        n_paths:              int   = 100_000,
        n_steps_per_interval: int   = 20,
        verbose:              bool  = True,
    ) -> None:
        self.reference_model      = reference_model
        self.r                    = float(r)
        self.max_iter             = int(max_iter)
        self.tol_marginal         = float(tol_marginal)
        self.tol_martingale       = float(tol_martingale)
        self.n_paths              = int(n_paths)
        self.n_steps_per_interval = int(n_steps_per_interval)
        self.verbose              = bool(verbose)

        self.calibrated_data: Dict[float, Dict[str, np.ndarray]] = {}
        self.all_diagnostics: Dict[float, List[Dict]]            = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_initial_latent(self, n_paths: int) -> np.ndarray:
        """Return the reference model's initial auxiliary state.

        For Heston / 3/2 models the initial latent state is the initial
        variance ``v0``; for Bergomi-1F the OU process starts at 0.

        Parameters
        ----------
        n_paths : int
            Number of paths.

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        if hasattr(self.reference_model, 'v0'):
            return np.full(n_paths, float(self.reference_model.v0))
        return np.zeros(n_paths, dtype=float)

    def _bin_states(
        self,
        x_states: np.ndarray,
        l_states: np.ndarray,
        n_bins:   int = 50,
    ) -> np.ndarray:
        """Assign each path to a bin based on its (x, l) state.

        Uses k-means clustering in the normalised 2-D (x, l) state space,
        or 1-D quantile binning when the latent state is degenerate.

        Parameters
        ----------
        x_states : np.ndarray, shape (n_paths,)
            Current price-state samples.
        l_states : np.ndarray, shape (n_paths,)
            Current latent-state samples.
        n_bins : int, optional
            Target number of bins.  Actual number is
            min(n_bins, n_paths // 20).  Defaults to 50.

        Returns
        -------
        labels : np.ndarray of int, shape (n_paths,)
            Bin label for each path (0-based).
        """
        n     = len(x_states)
        x_std = float(np.std(x_states))
        l_std = float(np.std(l_states))

        # Single bin when all states are identical
        if x_std < 1e-10 and l_std < 1e-10:
            return np.zeros(n, dtype=int)

        n_bins_actual = min(n_bins, max(1, n // 20))

        if l_std < 1e-10 or not _SKLEARN_AVAILABLE:
            # 1-D quantile binning on x only
            percentiles = np.percentile(
                x_states, np.linspace(0.0, 100.0, n_bins_actual + 1)
            )
            labels = np.digitize(x_states, percentiles[1:-1]).astype(int)
        else:
            # 2-D k-means, dimensions normalised to unit std
            x_norm   = x_states / x_std
            l_norm   = l_states / l_std
            features = np.column_stack([x_norm, l_norm])
            kmeans   = KMeans(n_clusters=n_bins_actual, n_init=5, random_state=42)
            labels   = kmeans.fit_predict(features).astype(int)

        return labels

    # ------------------------------------------------------------------
    # Main calibration entry point
    # ------------------------------------------------------------------

    def calibrate(
        self,
        maturities:     List[float],
        target_samples: Dict[float, np.ndarray],
    ) -> Dict:
        """Run full sequential calibration across all intervals.

        Parameters
        ----------
        maturities : list of float
            Calibration maturities T₁, …, T_N (T₀ = 0 is implicit).
        target_samples : dict {T: np.ndarray}
            Target marginal samples at each maturity.

        Returns
        -------
        dict
            Keys:

            ``'calibrated_data'`` – {T: {'x_samples', 'l_samples'}} at
            every calibrated maturity including T₀ = 0.

            ``'all_diagnostics'`` – {T: list of per-iteration diagnostic
            dicts} from every Sinkhorn run.

            ``'results'`` – {T: full return value of
            :meth:`_calibrate_interval`} for each maturity.
        """
        t_grid = [0.0] + sorted(maturities)
        n      = self.n_paths

        # Initialise state at T₀ = 0
        kappa_x = np.ones(n, dtype=float)
        kappa_l = self._get_initial_latent(n)

        self.calibrated_data = {
            0.0: {'x_samples': kappa_x.copy(), 'l_samples': kappa_l.copy()}
        }
        self.all_diagnostics = {}
        results: Dict        = {}

        for idx, (T_i, T_next) in enumerate(zip(t_grid[:-1], t_grid[1:])):
            if T_next not in target_samples:
                continue

            if self.verbose:
                print(f"\n[SKR] Interval [{T_i:.4f}, {T_next:.4f}]  "
                      f"(interval {idx + 1}/{len(t_grid) - 1})")

            t0     = time.time()
            result = self._calibrate_interval(
                T_i, T_next, kappa_x, kappa_l,
                target_samples[T_next], idx,
            )
            elapsed = time.time() - t0

            kappa_x = result['x_end_calibrated']
            kappa_l = result['l_end']

            self.calibrated_data[T_next] = {
                'x_samples': kappa_x.copy(),
                'l_samples': kappa_l.copy(),
            }
            self.all_diagnostics[T_next] = result['diagnostics']
            results[T_next]              = result

            if self.verbose:
                final = result['diagnostics'][-1] if result['diagnostics'] else {}
                w1   = final.get('wasserstein_1',       float('nan'))
                mart = final.get('martingale_residual',  float('nan'))
                nitr = len(result['diagnostics'])
                print(f"[SKR] Done in {elapsed:.1f}s  |  "
                      f"iters={nitr}  W1={w1:.4f}  mart={mart:.2e}")

        return {
            'calibrated_data': self.calibrated_data,
            'all_diagnostics': self.all_diagnostics,
            'results':         results,
        }

    # ------------------------------------------------------------------
    # One-interval calibration
    # ------------------------------------------------------------------

    def _calibrate_interval(
        self,
        T_i:            float,
        T_next:         float,
        kappa_i_x:      np.ndarray,
        kappa_i_l:      np.ndarray,
        target_samples: np.ndarray,
        interval_idx:   int,
    ) -> Dict:
        """Solve the one-period bridge problem for interval [T_i, T_next].

        Implements the martingale Sinkhorn fixed-point:
            a_{k+1}(x, l) = (g_k * γ_{(x,l)})^{-1}(x)

        where γ_{(x,l)} = Law(Y_{T_next} | R_{T_i} = (x, l)) is the
        reference conditional law, approximated by binning states into
        groups (see :meth:`_bin_states`).

        Parameters
        ----------
        T_i, T_next : float
            Interval endpoints.
        kappa_i_x, kappa_i_l : np.ndarray, shape (n_paths,)
            Joint-state samples (X, L) at T_i.
        target_samples : np.ndarray
            Samples from target marginal μ_{i+1} at T_next.
        interval_idx : int
            Interval index, used for deterministic seeding.

        Returns
        -------
        dict
            Keys:

            ``'shift_fn'``         – callable a(x, l) → shift value.
            ``'transport_g'``      – (y_grid, g_values) arrays.
            ``'x_end_calibrated'`` – calibrated terminal X samples.
            ``'l_end'``            – propagated latent-state samples.
            ``'diagnostics'``      – list of per-iteration diagnostic dicts.
        """
        n_paths = len(kappa_i_x)
        seed    = interval_idx * 7919 + 42  # deterministic, varied per interval

        # ---- 1. Simulate one reference transition per path ----
        y_ref, l_end = self._simulate_reference_transitions(
            T_i, T_next, kappa_i_x, kappa_i_l, seed=seed,
        )

        # ---- 2. Bin states into groups ----
        labels        = self._bin_states(kappa_i_x, kappa_i_l, n_bins=50)
        unique_labels = np.unique(labels)

        # ---- 3. Sinkhorn iterations ----
        a:                   np.ndarray = np.zeros(n_paths)
        diagnostics_history: List[Dict] = []

        for k in range(self.max_iter):
            # a. Shifted reference samples (ρ^a)
            z = a + y_ref

            # b. Transport map g_k = F_{target}^{-1} ∘ F_{ρ^a}
            y_grid, g_values = self._update_transport_map(z, target_samples)

            # c. Solve for new shifts within each bin (vectorised)
            a_new = np.empty(n_paths)
            for label in unique_labels:
                bin_mask          = labels == label
                gamma_b           = y_ref[bin_mask]
                x_targets_b       = kappa_i_x[bin_mask]
                a_new[bin_mask]   = self._solve_shift_batch(
                    x_targets_b, g_values, y_grid, gamma_b,
                )

            # d. Recenter: enforce E[a] = 0 (martingale centering)
            a_new -= float(np.mean(a_new))

            # e. Diagnostics
            x_end = np.interp(a_new + y_ref, y_grid, g_values)
            diag  = self._compute_diagnostics(
                x_end, target_samples, a_new, kappa_i_x,
            )
            diagnostics_history.append(diag)

            if self.verbose and (k == 0 or (k + 1) % 10 == 0):
                print(f"  iter {k + 1:3d}: "
                      f"W1={diag['wasserstein_1']:.4f}  "
                      f"mart={diag['martingale_residual']:.2e}  "
                      f"shift_ce={diag['shift_centering_error']:.2e}")

            # f. Convergence check
            converged = (
                diag['wasserstein_1']           < self.tol_marginal
                and abs(diag['martingale_residual']) < self.tol_martingale
            )
            if converged:
                if self.verbose:
                    print(f"  Converged at iteration {k + 1}.")
                break

            a = a_new

        # Final calibrated terminal prices using last transport map
        y_grid_f, g_f = self._update_transport_map(a + y_ref, target_samples)
        x_end_calibrated = np.interp(a + y_ref, y_grid_f, g_f)

        shift_fn = self._build_shift_fn(kappa_i_x, kappa_i_l, a)

        return {
            'shift_fn':         shift_fn,
            'transport_g':      (y_grid_f, g_f),
            'x_end_calibrated': x_end_calibrated,
            'l_end':            l_end,
            'diagnostics':      diagnostics_history,
        }

    # ------------------------------------------------------------------
    # Reference transition simulation
    # ------------------------------------------------------------------

    def _simulate_reference_transitions(
        self,
        T_i:      float,
        T_next:   float,
        x_states: np.ndarray,
        l_states: np.ndarray,
        seed:     Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate reference model transitions from (x, l) states at T_i to T_next.

        Calls the reference model's ``simulate_joint_from_state`` method to
        draw one sample from the conditional law γ_{(x,l)} for each path.

        Parameters
        ----------
        T_i, T_next : float
            Interval endpoints.
        x_states : np.ndarray, shape (n_states,)
            Starting price values.
        l_states : np.ndarray, shape (n_states,)
            Starting latent-state values.
        seed : int, optional
            Random seed.

        Returns
        -------
        y_next : np.ndarray, shape (n_states,)
            X values at T_next under the reference model.
        l_next : np.ndarray, shape (n_states,)
            Latent-state values at T_next.
        """
        y_next, l_next = self.reference_model.simulate_joint_from_state(
            T_i, T_next,
            np.asarray(x_states, dtype=float),
            np.asarray(l_states, dtype=float),
            n_steps=self.n_steps_per_interval,
            seed=seed,
        )
        return np.asarray(y_next, dtype=float), np.asarray(l_next, dtype=float)

    # ------------------------------------------------------------------
    # Conditional mean computation
    # ------------------------------------------------------------------

    def _compute_conditional_mean(
        self,
        g:             np.ndarray,
        y_grid:        np.ndarray,
        a:             float,
        gamma_samples: np.ndarray,
    ) -> float:
        """Compute E[g(a + Y₁) | R₀ = r] via Monte Carlo using gamma_samples.

        Parameters
        ----------
        g : np.ndarray
            Transport map values on y_grid.
        y_grid : np.ndarray
            Domain grid for g.
        a : float
            Shift value.
        gamma_samples : np.ndarray
            Conditional transition samples Y₁ | R₀ = r.

        Returns
        -------
        float
            Monte Carlo estimate of the conditional mean.
        """
        g_vals = np.interp(float(a) + gamma_samples, y_grid, g)
        return float(np.mean(g_vals))

    # ------------------------------------------------------------------
    # Single-path shift solve via Brent's method
    # ------------------------------------------------------------------

    def _solve_shift(
        self,
        x_target:      float,
        g:             np.ndarray,
        y_support:     np.ndarray,
        gamma_samples: np.ndarray,
    ) -> float:
        """Solve for shift a such that E[g(a + Y) | gamma_samples] = x_target.

        Uses Brent's method (``scipy.optimize.brentq``) to find the unique
        root of f(a) = E[g(a + Y)] − x_target.

        Parameters
        ----------
        x_target : float
            Target conditional mean (current X state value).
        g : np.ndarray
            Transport map values on y_support.
        y_support : np.ndarray
            Domain grid for g.
        gamma_samples : np.ndarray
            Conditional Y samples for this bin.

        Returns
        -------
        float
            Shift value a.
        """
        g_min = float(g[0])
        g_max = float(g[-1])
        x_c   = float(np.clip(x_target, g_min + 1e-10, g_max - 1e-10))

        a_lo = float(y_support[0]  - np.max(gamma_samples))
        a_hi = float(y_support[-1] - np.min(gamma_samples))

        def f(a_val: float) -> float:
            return self._compute_conditional_mean(g, y_support, a_val, gamma_samples) - x_c

        try:
            return float(optimize.brentq(f, a_lo, a_hi, xtol=1e-6, maxiter=50))
        except (ValueError, RuntimeError):
            return 0.0

    # ------------------------------------------------------------------
    # Batch vectorised shift solve
    # ------------------------------------------------------------------

    def _solve_shift_batch(
        self,
        x_targets:     np.ndarray,
        g_values:      np.ndarray,
        y_grid:        np.ndarray,
        gamma_samples: np.ndarray,
        n_search:      int = 300,
    ) -> np.ndarray:
        """Vectorised shift solve for all x_targets sharing the same gamma_samples.

        Precomputes f(a) = E[g(a + γ)] on a fine a-grid, then uses
        ``np.interp`` to invert it for every x_target.  This is O(n_search ×
        n_gamma) instead of n_targets individual ``brentq`` calls.

        Parameters
        ----------
        x_targets : np.ndarray, shape (n_b,)
            Target X values for all paths in this bin.
        g_values, y_grid : np.ndarray
            Transport map values and their domain.
        gamma_samples : np.ndarray, shape (n_gamma,)
            Reference transitions for paths in this bin.
        n_search : int, optional
            Resolution of the a-search grid.  Defaults to 300.

        Returns
        -------
        a_result : np.ndarray, shape (n_b,)
            Shift values.
        """
        if len(gamma_samples) == 0 or len(x_targets) == 0:
            return np.zeros(len(x_targets))

        g_min = float(g_values[0])
        g_max = float(g_values[-1])
        a_lo  = float(y_grid[0]  - np.max(gamma_samples))
        a_hi  = float(y_grid[-1] - np.min(gamma_samples))

        a_search = np.linspace(a_lo, a_hi, n_search)          # (n_search,)

        # Vectorised: shifted[i,j] = a_search[i] + gamma_samples[j]
        shifted      = a_search[:, None] + gamma_samples[None, :]  # (n_search, n_gamma)
        g_on_shifted = np.interp(
            shifted.ravel(), y_grid, g_values
        ).reshape(n_search, -1)
        f_vals = g_on_shifted.mean(axis=1)                    # (n_search,) monotone ↑

        # Invert f at each x_target by interpolation
        x_clipped = np.clip(x_targets, g_min + 1e-10, g_max - 1e-10)
        return np.interp(x_clipped, f_vals, a_search)

    # ------------------------------------------------------------------
    # Transport map construction
    # ------------------------------------------------------------------

    def _update_transport_map(
        self,
        rho_a_samples:  np.ndarray,
        target_samples: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute increasing transport map g = F_{ν}^{-1} ∘ F_{ρ^a}.

        The i-th sorted rho_a sample is mapped to the i-th sorted target
        sample (rank-based quantile transport, monotone increasing).

        Parameters
        ----------
        rho_a_samples : np.ndarray
            Shifted reference samples a + Y.
        target_samples : np.ndarray
            Target marginal samples ν.

        Returns
        -------
        y_grid : np.ndarray
            Sorted ρ^a values (domain of g).
        g_values : np.ndarray
            Corresponding target quantile values (codomain of g).
        """
        n_src = len(rho_a_samples)
        n_tgt = len(target_samples)

        y_grid = np.sort(rho_a_samples)

        if n_src == n_tgt:
            g_values = np.sort(target_samples)
        else:
            # Re-sample target quantiles to match source size
            p_src    = (np.arange(n_src) + 0.5) / n_src
            p_tgt    = (np.arange(n_tgt) + 0.5) / n_tgt
            g_values = np.interp(p_src, p_tgt, np.sort(target_samples))

        return y_grid, g_values

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _compute_diagnostics(
        self,
        x_end:          np.ndarray,
        target_samples: np.ndarray,
        shifts:         np.ndarray,
        kappa_x:        np.ndarray,
    ) -> Dict:
        """Compute convergence diagnostics for one Sinkhorn iteration.

        Metrics
        -------
        ``wasserstein_1``         – W₁(x_end, target), empirical 1-Wasserstein.
        ``martingale_residual``   – |E[X_{T+1}] − mean(κ_x)|, martingale error.
        ``shift_centering_error`` – |mean(shifts)|, post-centering residual.

        Parameters
        ----------
        x_end : np.ndarray
            Calibrated terminal price samples.
        target_samples : np.ndarray
            Target marginal samples.
        shifts : np.ndarray
            Shift array a for each path.
        kappa_x : np.ndarray
            Starting X values κ_i at T_i.

        Returns
        -------
        dict
        """
        # W₁ via sorted quantile difference (closed form for empirical measures)
        s1 = np.sort(x_end)
        s2 = np.sort(target_samples)
        n1, n2   = len(s1), len(s2)
        p_common = np.linspace(0.0, 1.0, max(n1, n2))
        q1 = np.interp(p_common, (np.arange(n1) + 0.5) / n1, s1)
        q2 = np.interp(p_common, (np.arange(n2) + 0.5) / n2, s2)
        w1 = float(np.trapz(np.abs(q1 - q2), p_common))

        mart_res = float(np.mean(x_end) - np.mean(kappa_x))
        shift_ce = float(np.mean(shifts))

        return {
            'wasserstein_1':         w1,
            'martingale_residual':   mart_res,
            'shift_centering_error': shift_ce,
        }

    # ------------------------------------------------------------------
    # Shift function builder
    # ------------------------------------------------------------------

    def _build_shift_fn(
        self,
        kappa_x: np.ndarray,
        kappa_l: np.ndarray,
        a:       np.ndarray,
    ) -> Callable:
        """Build a callable shift function from the calibrated (x, l) → a mapping.

        For 1-D states (or when sklearn is unavailable) uses sorted linear
        interpolation on x.  For 2-D (x, l) states uses nearest-neighbour
        interpolation.

        Parameters
        ----------
        kappa_x, kappa_l : np.ndarray, shape (n_paths,)
            State cloud at T_i.
        a : np.ndarray, shape (n_paths,)
            Calibrated shift values.

        Returns
        -------
        Callable
            ``shift_fn(x, l)`` returning a shift array of the same shape
            as x.
        """
        l_std = float(np.std(kappa_l))

        if l_std < 1e-10 or not _SKLEARN_AVAILABLE:
            # 1-D interpolation on x
            sort_idx = np.argsort(kappa_x)
            x_s, a_s = kappa_x[sort_idx], a[sort_idx]

            def shift_fn(x_new, l_new=None):
                return np.interp(np.asarray(x_new, dtype=float), x_s, a_s)
        else:
            from scipy.interpolate import NearestNDInterpolator
            points   = np.column_stack([kappa_x, kappa_l])
            nn_interp = NearestNDInterpolator(points, a)

            def shift_fn(x_new, l_new):
                pts = np.column_stack([np.asarray(x_new), np.asarray(l_new)])
                return nn_interp(pts)

        return shift_fn

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_calibrated_samples(
        self,
        maturity: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x_samples, l_samples) at the given calibration maturity.

        Parameters
        ----------
        maturity : float
            A maturity previously passed to :meth:`calibrate`.

        Returns
        -------
        x_samples : np.ndarray
        l_samples : np.ndarray
        """
        data = self.calibrated_data[maturity]
        return data['x_samples'].copy(), data['l_samples'].copy()

    def reprice_calls(
        self,
        T:      float,
        K_grid: np.ndarray,
        F:      float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reprice European calls at maturity T using calibrated samples.

        Parameters
        ----------
        T : float
            A maturity previously passed to :meth:`calibrate`.
        K_grid : np.ndarray
            Strike grid.
        F : float, optional
            Forward price for discounting context.  Defaults to 1.0.

        Returns
        -------
        call_prices : np.ndarray, shape (len(K_grid),)
            Discounted call prices.
        mc_se : np.ndarray, shape (len(K_grid),)
            Monte Carlo standard errors.
        """
        if T not in self.calibrated_data:
            raise KeyError(
                f"Maturity T={T} not calibrated. "
                f"Available: {sorted(self.calibrated_data.keys())}"
            )

        S_T    = self.calibrated_data[T]['x_samples']
        K_grid = np.asarray(K_grid, dtype=float)
        disc   = np.exp(-self.r * T)
        n_p    = len(S_T)

        call_prices = np.empty(len(K_grid))
        mc_se       = np.empty(len(K_grid))

        for j, K in enumerate(K_grid):
            payoffs        = np.maximum(S_T - K, 0.0)
            call_prices[j] = disc * float(np.mean(payoffs))
            mc_se[j]       = disc * float(np.std(payoffs)) / np.sqrt(n_p)

        return call_prices, mc_se

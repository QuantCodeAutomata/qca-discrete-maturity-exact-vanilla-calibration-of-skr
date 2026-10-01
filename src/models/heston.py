"""
Heston stochastic-volatility model — Monte Carlo simulator.

Dynamics (risk-neutral forward measure, no drift):
    dS_t = S_t √V_t dW_t
    dV_t = κ(θ − V_t) dt + η √V_t dB_t
    corr(dW, dB) = ρ

The variance process is simulated with the Quadratic Exponential (QE)
scheme of Andersen (2008), which avoids negativity of V without absorbing
it at zero.  The log-price is updated via the exact conditional-moment
formula that preserves the martingale property.

Reference
---------
Andersen, L. B. G. (2008), "Efficient simulation of the Heston
stochastic volatility model", Journal of Computational Finance, 11(3).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple, List, Dict

# ---------------------------------------------------------------------------
# Default parameters (from paper)
# ---------------------------------------------------------------------------

HESTON_PARAMS: Dict[str, float] = {
    "kappa": 1.2484,
    "theta": 0.0988,
    "eta":   1.1175,
    "rho":   -0.8038,
    "v0":    0.1020,
}


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class HestonModel:
    """Heston (1993) stochastic-volatility model, QE Monte Carlo.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed of the variance process.
    theta : float
        Long-run mean variance (θ in the SDE).
    eta : float
        Vol-of-vol (σ in the SDE).
    rho : float
        Instantaneous correlation between the asset and variance
        Brownian motions.
    v0 : float
        Initial variance V_0.
    r : float, optional
        Continuously-compounded discount rate used for discounting
        option payoffs.  The SDE itself has no drift (forward measure).
        Defaults to 0.02.
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        eta: float,
        rho: float,
        v0: float,
        r: float = 0.02,
    ) -> None:
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.eta   = float(eta)
        self.rho   = float(rho)
        self.v0    = float(v0)
        self.r     = float(r)

    # ------------------------------------------------------------------
    # Internal QE step
    # ------------------------------------------------------------------

    def _qe_variance_step(
        self,
        V_curr: np.ndarray,
        dt: float,
        rng: np.random.Generator,
        psi_c: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One QE variance step.

        Parameters
        ----------
        V_curr : np.ndarray
            Current variance values, shape ``(n,)``.
        dt : float
            Time step size.
        rng : np.random.Generator
            Random number generator.
        psi_c : float, optional
            Threshold separating the quadratic (Case 1, ψ ≤ ψ_c) and
            exponential (Case 2, ψ > ψ_c) regimes.  Default is 1.5.

        Returns
        -------
        V_new : np.ndarray
            Updated variance values.
        U_V : np.ndarray
            Uniform draws used internally (needed to recover the
            driving Brownian increment for the stock update).
        """
        kappa, theta, eta = self.kappa, self.theta, self.eta
        n = len(V_curr)

        e = np.exp(-kappa * dt)
        # Conditional mean and variance
        m  = theta + (V_curr - theta) * e
        s2 = (V_curr * eta**2 * e * (1.0 - e) / kappa
              + theta * eta**2 * (1.0 - e)**2 / (2.0 * kappa))
        psi = s2 / np.maximum(m**2, 1e-14)

        U_V   = rng.uniform(0.0, 1.0, n)
        V_new = np.zeros(n)

        # ---- Case 1: ψ ≤ ψ_c  (quadratic exponential) ----
        mask1 = psi <= psi_c
        if np.any(mask1):
            psi1  = psi[mask1]
            m1    = m[mask1]
            inv_p = 2.0 / psi1
            b2    = np.maximum(inv_p - 1.0 + np.sqrt(inv_p) * np.sqrt(inv_p - 1.0), 0.0)
            a     = m1 / (1.0 + b2)
            Z_V   = norm.ppf(np.clip(U_V[mask1], 1e-14, 1.0 - 1e-14))
            V_new[mask1] = a * (np.sqrt(b2) + Z_V) ** 2

        # ---- Case 2: ψ > ψ_c  (exponential) ----
        mask2 = ~mask1
        if np.any(mask2):
            psi2 = psi[mask2]
            m2   = m[mask2]
            p    = (psi2 - 1.0) / (psi2 + 1.0)
            beta = 2.0 / (m2 * (psi2 + 1.0))
            U2   = U_V[mask2]
            V_new[mask2] = np.where(
                U2 <= p,
                0.0,
                np.log(
                    np.maximum(1.0 - p, 1e-14) / np.maximum(1.0 - U2, 1e-14)
                ) / np.maximum(beta, 1e-14),
            )

        return np.maximum(V_new, 0.0), U_V

    def _log_price_step(
        self,
        V_curr: np.ndarray,
        V_new: np.ndarray,
        dt: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Compute the log-price increment given V_curr and V_new.

        Uses the exact conditional-moment formula (Andersen 2008, Eq. 44):

            Δ ln S = (ρ/η)(V_new − V_curr − κθ·dt) + (κρ/η − ½) I_V
                   + √( (1−ρ²) I_V ) · Z

        where I_V ≈ dt·(V_curr + V_new)/2 (trapezoidal quadrature).

        Parameters
        ----------
        V_curr, V_new : np.ndarray
            Variance at start and end of the step.
        dt : float
            Time step.
        rng : np.random.Generator
            Random number generator for the independent Brownian
            increment Z.

        Returns
        -------
        np.ndarray
            Log-price increment Δ ln S.
        """
        kappa, theta, eta, rho = self.kappa, self.theta, self.eta, self.rho
        I_V    = np.maximum(dt * (V_curr + V_new) * 0.5, 0.0)
        Z      = rng.standard_normal(len(V_curr))
        dlogS  = ((rho / eta) * (V_new - V_curr - kappa * theta * dt)
                  + (kappa * rho / eta - 0.5) * I_V
                  + np.sqrt(np.maximum((1.0 - rho**2) * I_V, 0.0)) * Z)
        return dlogS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
        S0: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate Heston paths using the QE scheme (Andersen 2008).

        The asset is forward-normalised (S_0 = ``S0``, no drift in the SDE).
        The variance process is simulated with the QE scheme, and the
        log-price is updated via the exact conditional-moment increment to
        preserve the martingale property.

        Parameters
        ----------
        T : float
            Total simulation horizon in years.
        n_paths : int
            Number of Monte Carlo paths.
        n_steps : int
            Number of time steps.
        seed : int, optional
            Random seed.
        S0 : float, optional
            Initial forward price.  Defaults to 1.0.

        Returns
        -------
        S_paths : np.ndarray, shape (n_paths, n_steps+1)
            Simulated forward prices.
        V_paths : np.ndarray, shape (n_paths, n_steps+1)
            Simulated variance paths.
        """
        rng = np.random.default_rng(seed)
        dt  = T / n_steps

        S = np.full(n_paths, float(S0))
        V = np.full(n_paths, self.v0)

        S_paths = np.empty((n_paths, n_steps + 1))
        V_paths = np.empty((n_paths, n_steps + 1))
        S_paths[:, 0] = S
        V_paths[:, 0] = V

        for i in range(n_steps):
            V_new, _ = self._qe_variance_step(V, dt, rng)
            dlogS     = self._log_price_step(V, V_new, dt, rng)
            S         = S * np.exp(dlogS)
            V         = V_new
            S_paths[:, i + 1] = S
            V_paths[:, i + 1] = V

        return S_paths, V_paths

    def simulate_joint_from_state(
        self,
        T_start: float,
        T_end: float,
        x_states: np.ndarray,
        v_states: np.ndarray,
        n_steps: int = 20,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate Heston from given (S, V) states at T_start to T_end.

        Used in the SKR calibration loop to propagate particles
        κ_i = (S_i, V_i) forward in time.

        Parameters
        ----------
        T_start : float
            Starting time (used only to label the interval; the SDE has
            no explicit time dependence in the Heston model).
        T_end : float
            Ending time.  The time horizon is T = T_end − T_start.
        x_states : np.ndarray, shape (n_states,)
            Initial forward-price values S at T_start.
        v_states : np.ndarray, shape (n_states,)
            Initial variance values V at T_start.
        n_steps : int, optional
            Number of inner time steps.  Defaults to 20.
        seed : int, optional
            Random seed.

        Returns
        -------
        X_end : np.ndarray, shape (n_states,)
            Forward prices at T_end.
        V_end : np.ndarray, shape (n_states,)
            Variance values at T_end.
        """
        x_states = np.asarray(x_states, dtype=float).ravel()
        v_states = np.asarray(v_states, dtype=float).ravel()
        n_states = len(x_states)

        rng = np.random.default_rng(seed)
        T   = T_end - T_start
        dt  = T / n_steps

        S = x_states.copy()
        V = v_states.copy()

        for _ in range(n_steps):
            V_new, _ = self._qe_variance_step(V, dt, rng)
            dlogS     = self._log_price_step(V, V_new, dt, rng)
            S         = S * np.exp(dlogS)
            V         = V_new

        return S, V

    def price_call_mc(
        self,
        T: float,
        K_grid: np.ndarray,
        n_paths: int = 100_000,
        n_steps: int = 50,
        seed: int = 42,
    ) -> np.ndarray:
        """Price European calls via Monte Carlo.

        Parameters
        ----------
        T : float
            Maturity in years.
        K_grid : np.ndarray
            Strike grid.
        n_paths : int, optional
            Number of paths.  Defaults to 100 000.
        n_steps : int, optional
            Number of time steps.  Defaults to 50.
        seed : int, optional
            Random seed.  Defaults to 42.

        Returns
        -------
        np.ndarray
            Call prices of shape ``(len(K_grid),)``, discounted at rate
            ``self.r``.
        """
        K_grid = np.asarray(K_grid, dtype=float)
        S_paths, _ = self.simulate_paths(T, n_paths, n_steps, seed=seed, S0=1.0)
        S_T        = S_paths[:, -1]
        discount   = np.exp(-self.r * T)
        prices     = np.array(
            [discount * np.mean(np.maximum(S_T - K, 0.0)) for K in K_grid]
        )
        return prices

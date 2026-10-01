"""
3/2 stochastic-volatility model — Monte Carlo simulator.

Dynamics (risk-neutral forward measure, no drift):
    dS_t = S_t √V_t dW_t
    dV_t = κ V_t (θ − V_t) dt + η V_t^{3/2} dB_t
    corr(dW, dB) = ρ

The variance V_t is simulated via the transformation U_t = 1/V_t, which
satisfies an exact CIR process:

    dU_t = κ_U (θ_U − U_t) dt + η √U_t dW_U_t

where
    κ_U = κθ,    θ_U = (κ + η²)/(κθ),    dW_U = −dB.

The CIR process for U is discretised with the Quadratic Exponential (QE)
scheme (Andersen 2008) to ensure U > 0 (equivalently V > 0) without
artificial reflection.  The log-price is updated with an Euler scheme
using explicitly generated correlated Brownian increments.

References
----------
Drimus, G. G. (2012), "Options on realized variance by transform methods:
a non-affine stochastic volatility model", Quantitative Finance, 12(11).
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

MODEL_3_2_PARAMS: Dict[str, float] = {
    "kappa": 0.3600,
    "theta": 5.0050,
    "eta":   7.6870,
    "rho":   -1.0000,
    "v0":    0.0877,
}


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class Model3_2:
    """3/2 stochastic-volatility model, reciprocal-CIR / QE Monte Carlo.

    The variance SDE is:
        dV_t = κ V_t (θ − V_t) dt + η V_t^{3/2} dB_t.

    The transformation U_t = 1/V_t satisfies the CIR SDE:
        dU_t = κθ (θ_U − U_t) dt + η √U_t dW_U_t,
        θ_U = (κ + η²) / (κθ),    dW_U = −dB,
    which is simulated with the QE scheme to guarantee U_t > 0.

    Parameters
    ----------
    kappa : float
        Mean-reversion coefficient for the variance-of-variance.
    theta : float
        Long-run mean of V_t (θ in the SDE above).
    eta : float
        Vol-of-vol.
    rho : float
        Correlation between the asset and variance Brownian motions.
    v0 : float
        Initial variance V_0 > 0.
    r : float, optional
        Discount rate for option pricing.  Defaults to 0.02.
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

        # CIR parameters for U = 1/V
        # dU = kappa_cir*(theta_cir - U)*dt + eta*sqrt(U)*dW_U
        self._kappa_cir = float(kappa) * float(theta)
        self._theta_cir = (float(kappa) + float(eta) ** 2) / (float(kappa) * float(theta))

    # ------------------------------------------------------------------
    # QE step for the reciprocal CIR  (U = 1/V)
    # ------------------------------------------------------------------

    def _qe_cir_step(
        self,
        U_curr: np.ndarray,
        dt: float,
        rng: np.random.Generator,
        psi_c: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One QE step for the CIR process U_t = 1/V_t.

        Parameters
        ----------
        U_curr : np.ndarray
            Current U values (must be > 0).
        dt : float
            Time step.
        rng : np.random.Generator
            Random number generator.
        psi_c : float, optional
            QE threshold (default 1.5).

        Returns
        -------
        U_new : np.ndarray
            Updated U values.
        Z_U : np.ndarray
            Standard-normal draws used in Case 1 (set to 0 for Case 2).
            Multiplied by ``sqrt(dt)`` gives the approximate dW_U increment.
        """
        kappa_c = self._kappa_cir
        theta_c = self._theta_cir
        eta     = self.eta
        n       = len(U_curr)

        e  = np.exp(-kappa_c * dt)
        m  = theta_c + (U_curr - theta_c) * e
        s2 = (U_curr * eta**2 * e * (1.0 - e) / kappa_c
              + theta_c * eta**2 * (1.0 - e)**2 / (2.0 * kappa_c))
        psi = s2 / np.maximum(m**2, 1e-14)

        U_V   = rng.uniform(0.0, 1.0, n)
        U_new = np.zeros(n)
        Z_U   = np.zeros(n)

        # Case 1: ψ ≤ ψ_c
        mask1 = psi <= psi_c
        if np.any(mask1):
            psi1  = psi[mask1]
            m1    = m[mask1]
            inv_p = 2.0 / psi1
            b2    = np.maximum(inv_p - 1.0 + np.sqrt(inv_p) * np.sqrt(inv_p - 1.0), 0.0)
            a     = m1 / (1.0 + b2)
            zv    = norm.ppf(np.clip(U_V[mask1], 1e-14, 1.0 - 1e-14))
            U_new[mask1] = a * (np.sqrt(b2) + zv) ** 2
            Z_U[mask1]   = zv  # approximate dW_U / sqrt(dt)

        # Case 2: ψ > ψ_c
        mask2 = ~mask1
        if np.any(mask2):
            psi2 = psi[mask2]
            m2   = m[mask2]
            p    = (psi2 - 1.0) / (psi2 + 1.0)
            beta = 2.0 / (m2 * (psi2 + 1.0))
            U2   = U_V[mask2]
            U_new[mask2] = np.where(
                U2 <= p,
                0.0,
                np.log(
                    np.maximum(1.0 - p, 1e-14) / np.maximum(1.0 - U2, 1e-14)
                ) / np.maximum(beta, 1e-14),
            )
            # Approximate Z_U for Case 2 using a fresh draw
            Z_U[mask2] = rng.standard_normal(int(mask2.sum()))

        return np.maximum(U_new, 1e-14), Z_U

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _simulate_block(
        self,
        S_init: np.ndarray,
        V_init: np.ndarray,
        dt: float,
        n_steps: int,
        rng: np.random.Generator,
        store: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Simulate one block of n_steps time steps.

        Uses the reciprocal-CIR QE scheme for U = 1/V together with a
        correlated Euler update for the log-price:

            ln(S_{t+dt}/S_t) = −I_V/2 + √I_V · [ρ Z_B + √(1−ρ²) Z_W]

        where I_V ≈ dt·(V_t + V_{t+dt})/2 and Z_B is the standard-normal
        driving the variance (note: dW_U = −dB so Z_B = −Z_U).

        Parameters
        ----------
        S_init, V_init : np.ndarray
            Initial prices and variances.
        dt, n_steps : float, int
            Step size and count.
        rng : np.random.Generator
        store : bool
            If ``True``, store and return full path arrays.

        Returns
        -------
        S_end, V_end : np.ndarray
        S_paths, V_paths : np.ndarray or None  (only non-None when store=True)
        """
        n      = len(S_init)
        rho    = self.rho
        S      = S_init.copy()
        V      = V_init.copy()

        if store:
            S_out = np.empty((n, n_steps + 1))
            V_out = np.empty((n, n_steps + 1))
            S_out[:, 0] = S
            V_out[:, 0] = V
        else:
            S_out = V_out = None

        for i in range(n_steps):
            # Simulate U = 1/V with QE-CIR
            U_curr      = 1.0 / np.maximum(V, 1e-10)
            U_new, Z_U  = self._qe_cir_step(U_curr, dt, rng)
            V_new       = 1.0 / np.maximum(U_new, 1e-14)

            # dB = -dW_U  =>  Z_B = -Z_U
            Z_B    = -Z_U
            Z_W    = rng.standard_normal(n)

            # Left-endpoint Euler for I_V = dt * V_t ensures
            # E[exp(dlogS) | S_t, V_t] = 1 (discrete martingale property),
            # because I_V is then independent of the noise Z_B and Z_W.
            I_V    = np.maximum(dt * V, 0.0)
            dlogS  = (-0.5 * I_V
                      + np.sqrt(I_V) * (rho * Z_B + np.sqrt(max(1.0 - rho**2, 0.0)) * Z_W))

            S = S * np.exp(dlogS)
            V = V_new

            if store:
                S_out[:, i + 1] = S
                V_out[:, i + 1] = V

        return S, V, S_out, V_out

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
        """Simulate 3/2 model paths via reciprocal-CIR QE + correlated Euler.

        The instantaneous variance V_t is obtained as V_t = 1/U_t where U_t
        follows a CIR process simulated with the QE scheme.  The log-price
        is updated with an Euler step that uses the same Brownian increment
        as the variance (dB = −dW_U) to maintain the correlation ρ.

        Parameters
        ----------
        T : float
            Simulation horizon in years.
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
            Simulated forward-price paths.
        V_paths : np.ndarray, shape (n_paths, n_steps+1)
            Simulated variance paths.
        """
        rng = np.random.default_rng(seed)
        dt  = T / n_steps

        S_init = np.full(n_paths, float(S0))
        V_init = np.full(n_paths, self.v0)

        _, _, S_paths, V_paths = self._simulate_block(
            S_init, V_init, dt, n_steps, rng, store=True
        )
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
        """Simulate from given (S, V) states at T_start to T_end.

        Used in the SKR calibration loop to propagate particles.  The
        3/2 model SDE has no explicit time dependence, so T_start is used
        only to compute the time horizon T = T_end − T_start.

        Parameters
        ----------
        T_start : float
            Start time.
        T_end : float
            End time.
        x_states : np.ndarray, shape (n_states,)
            Initial forward-price values at T_start.
        v_states : np.ndarray, shape (n_states,)
            Initial variance values at T_start.
        n_steps : int, optional
            Number of inner steps.  Defaults to 20.
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

        rng = np.random.default_rng(seed)
        T   = T_end - T_start
        dt  = T / n_steps

        S_end, V_end, _, _ = self._simulate_block(
            x_states, v_states, dt, n_steps, rng, store=False
        )
        return S_end, V_end

    def price_call_mc(
        self,
        T: float,
        K_grid: np.ndarray,
        n_paths: int = 100_000,
        n_steps: int = 100,
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
            Number of time steps.  Defaults to 100.
        seed : int, optional
            Random seed.  Defaults to 42.

        Returns
        -------
        np.ndarray
            Call prices of shape ``(len(K_grid),)``, discounted at rate
            ``self.r``.
        """
        K_grid     = np.asarray(K_grid, dtype=float)
        S_paths, _ = self.simulate_paths(T, n_paths, n_steps, seed=seed, S0=1.0)
        S_T        = S_paths[:, -1]
        discount   = np.exp(-self.r * T)
        prices     = np.array(
            [discount * np.mean(np.maximum(S_T - K, 0.0)) for K in K_grid]
        )
        return prices

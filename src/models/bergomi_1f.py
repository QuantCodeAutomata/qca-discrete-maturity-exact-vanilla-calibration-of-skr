"""
Bergomi one-factor rough-forward-variance model — Monte Carlo simulator.

Dynamics (risk-neutral forward measure):
    dS_t = S_t √V_t dW_t
    V_t  = ξ₀(t) · exp(η Z_t − c(t))
    dZ_t = −κ Z_t dt + dB_t,    Z_0 = 0
    c(t) = η²/(4κ) · (1 − e^{−2κt})   (centering so E[V_t] = ξ₀(t))
    corr(dW, dB) = ρ

The OU process Z_t is discretised with the **exact** transition:
    Z_{t+h} = Z_t e^{−κh} + σ_h ε_B,    σ_h = √((1−e^{−2κh})/(2κ)),
which preserves the correct marginal distribution of Z at any step size h.

The log-price is updated with a correlated Euler step:
    Δ(ln S) = −I_V/2 + √I_V · (ρ ε_B + √(1−ρ²) ε_W),
    I_V ≈ h (V_t + V_{t+h}) / 2,
where ε_B is the same unit-normal that drives the OU step, giving the
exact instantaneous correlation ρ between the asset and variance
Brownian motions.

Reference
---------
Bergomi, L. (2016), *Stochastic Volatility Modelling*, Chapman & Hall.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Dict

# ---------------------------------------------------------------------------
# Default parameters (from paper)
# ---------------------------------------------------------------------------

BERGOMI_PARAMS: Dict = {
    "kappa": 2.4971,
    "eta":   3.6191,
    "rho":  -0.8796,
    "xi0_breakpoints": [0.0, 0.25, 0.5, 1.0, 2.0, 3.0],
    "xi0_values":      [0.10209961, 0.11963007, 0.09081965,
                        0.08462666, 0.08462666, 0.08383247],
}


# ---------------------------------------------------------------------------
# Forward variance curve
# ---------------------------------------------------------------------------

def get_xi0(t: float, breakpoints: list, values: list) -> float:
    """Evaluate the piecewise-constant instantaneous forward variance ξ₀(t).

    The curve is left-continuous (value for [bₖ, b_{k+1}) is ``values[k]``).
    For t ≥ ``breakpoints[-1]`` the last element of ``values`` is returned.

    Parameters
    ----------
    t : float
        Query time.
    breakpoints : list of float
        Sorted boundary points including 0.0 as the first element.
    values : list of float
        Forward-variance values.  ``values[k]`` applies on
        [breakpoints[k], breakpoints[k+1]).  ``values[-1]`` covers
        [breakpoints[-1], +∞).

    Returns
    -------
    float
        ξ₀(t).
    """
    for k in range(len(breakpoints) - 1):
        if breakpoints[k] <= t < breakpoints[k + 1]:
            return float(values[k])
    return float(values[-1])


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class Bergomi1FModel:
    """Bergomi one-factor stochastic-volatility model, exact-OU Monte Carlo.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed of the OU process Z_t.
    eta : float
        Vol-of-vol amplitude.
    rho : float
        Correlation between asset and OU Brownian motions.
    xi0_breakpoints : list of float
        Breakpoints for the piecewise-constant forward variance ξ₀(t).
    xi0_values : list of float
        ξ₀ values on each interval (see :func:`get_xi0`).
    r : float, optional
        Discount rate for option pricing.  Defaults to 0.02.
    """

    def __init__(
        self,
        kappa: float,
        eta: float,
        rho: float,
        xi0_breakpoints: list,
        xi0_values: list,
        r: float = 0.02,
    ) -> None:
        self.kappa           = float(kappa)
        self.eta             = float(eta)
        self.rho             = float(rho)
        self.xi0_breakpoints = list(xi0_breakpoints)
        self.xi0_values      = list(xi0_values)
        self.r               = float(r)

    # ------------------------------------------------------------------
    # Helper: variance at a given (t, Z) state
    # ------------------------------------------------------------------

    def _variance(self, t: float, Z: np.ndarray) -> np.ndarray:
        """Compute instantaneous variance V_t = ξ₀(t) exp(η Z_t − c(t)).

        Parameters
        ----------
        t : float
            Current time.
        Z : np.ndarray
            OU process values at time t.

        Returns
        -------
        np.ndarray
            Variance values V_t ≥ 0.
        """
        xi0_t = get_xi0(t, self.xi0_breakpoints, self.xi0_values)
        # c(t) = η²/(4κ) · (1 − e^{-2κt})  ensures E[V_t] = ξ₀(t)
        c_t   = (self.eta ** 2 / (4.0 * self.kappa)) * (1.0 - np.exp(-2.0 * self.kappa * t))
        return xi0_t * np.exp(self.eta * Z - c_t)

    # ------------------------------------------------------------------
    # One-step simulation (exact OU + correlated Euler for S)
    # ------------------------------------------------------------------

    def _step(
        self,
        t: float,
        S: np.ndarray,
        Z: np.ndarray,
        dt: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance (S, Z) by one time step dt.

        Parameters
        ----------
        t : float
            Current time.
        S, Z : np.ndarray
            Current asset prices and OU states.
        dt : float
            Step size.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        S_new, Z_new : np.ndarray
            Updated asset prices and OU states at t + dt.
        """
        n       = len(S)
        kappa   = self.kappa
        eta     = self.eta
        rho     = self.rho

        # -- Exact OU step --
        exp_neg  = np.exp(-kappa * dt)
        sigma_Z  = np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
        eps_B    = rng.standard_normal(n)
        Z_new    = Z * exp_neg + sigma_Z * eps_B

        # -- Variance at t and t+dt --
        V_t   = self._variance(t,      Z)
        V_new = self._variance(t + dt, Z_new)

        # Left-endpoint variance integral I_V = dt · V_t ensures the
        # discrete martingale property E[exp(ΔlnS)|S_t,V_t]=1, because
        # I_V is then independent of both ε_B and ε_W.
        I_V   = np.maximum(dt * V_t, 0.0)

        # -- Correlated Euler log-price step --
        # ε_B is the unit-normal driving the OU (≈ dB / sqrt(dt))
        # so we use it as the correlated noise for dW = ρ dB + √(1-ρ²) dB⊥
        eps_W = rng.standard_normal(n)
        dlogS = (-0.5 * I_V
                 + np.sqrt(I_V) * (rho * eps_B + np.sqrt(max(1.0 - rho**2, 0.0)) * eps_W))
        S_new = S * np.exp(dlogS)

        return S_new, Z_new

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
        """Simulate Bergomi 1F paths using the exact OU discretisation.

        The OU process Z_t is advanced with the exact Gaussian transition:

            Z_{t+h} = Z_t e^{−κh} + √((1−e^{−2κh})/(2κ)) · ε_B

        so the marginal distribution of Z_t is exact at every time grid
        point.  The instantaneous variance and log-price are updated with
        correlated Euler steps as described in the module docstring.

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
        Z_paths : np.ndarray, shape (n_paths, n_steps+1)
            Simulated OU-process paths (state variable Z_t).
        """
        rng = np.random.default_rng(seed)
        dt  = T / n_steps

        S = np.full(n_paths, float(S0))
        Z = np.zeros(n_paths)

        S_paths = np.empty((n_paths, n_steps + 1))
        Z_paths = np.empty((n_paths, n_steps + 1))
        S_paths[:, 0] = S
        Z_paths[:, 0] = Z

        for i in range(n_steps):
            t         = i * dt
            S, Z      = self._step(t, S, Z, dt, rng)
            S_paths[:, i + 1] = S
            Z_paths[:, i + 1] = Z

        return S_paths, Z_paths

    def simulate_joint_from_state(
        self,
        T_start: float,
        T_end: float,
        x_states: np.ndarray,
        z_states: np.ndarray,
        n_steps: int = 20,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate Bergomi 1F from given (S, Z) states at T_start to T_end.

        Used in the SKR calibration loop to propagate particles
        κ_i = (S_i, Z_i) forward in time.  The time argument T_start is
        passed to the variance function ξ₀(t) so that the correct
        forward-variance value is used at each step.

        Parameters
        ----------
        T_start : float
            Initial time (passed to the variance curve).
        T_end : float
            Final time.
        x_states : np.ndarray, shape (n_states,)
            Initial forward-price values at T_start.
        z_states : np.ndarray, shape (n_states,)
            Initial OU process values Z at T_start.
        n_steps : int, optional
            Number of inner time steps.  Defaults to 20.
        seed : int, optional
            Random seed.

        Returns
        -------
        X_end : np.ndarray, shape (n_states,)
            Forward prices at T_end.
        Z_end : np.ndarray, shape (n_states,)
            OU process values at T_end.
        """
        x_states = np.asarray(x_states, dtype=float).ravel()
        z_states = np.asarray(z_states, dtype=float).ravel()

        rng = np.random.default_rng(seed)
        T   = T_end - T_start
        dt  = T / n_steps

        S = x_states.copy()
        Z = z_states.copy()

        for i in range(n_steps):
            t    = T_start + i * dt
            S, Z = self._step(t, S, Z, dt, rng)

        return S, Z

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
            Discounted call prices of shape ``(len(K_grid),)``.
        """
        K_grid     = np.asarray(K_grid, dtype=float)
        S_paths, _ = self.simulate_paths(T, n_paths, n_steps, seed=seed, S0=1.0)
        S_T        = S_paths[:, -1]
        discount   = np.exp(-self.r * T)
        prices     = np.array(
            [discount * np.mean(np.maximum(S_T - K, 0.0)) for K in K_grid]
        )
        return prices

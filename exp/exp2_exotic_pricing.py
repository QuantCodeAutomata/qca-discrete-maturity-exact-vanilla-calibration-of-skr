"""
Experiment 2: Exotic pricing under equally calibrated models.

Prices reverse cliquet and memory autocallable under:
    Bass LV, Heston SKR, 3/2 SKR, Bergomi-1F SKR

All four models share identical vanilla marginal calibration (eSSVI
smiles at T ∈ {0.25, 0.5, 1.0, 2.0, 3.0}).  Differences in exotic
prices arise purely from model-specific inter-maturity dynamics.

Usage:
    python exp/exp2_exotic_pricing.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.essvi_smiles import get_target_marginals
from src.models.heston import HestonModel, HESTON_PARAMS
from src.models.model_3_2 import Model3_2, MODEL_3_2_PARAMS
from src.models.bergomi_1f import Bergomi1FModel, BERGOMI_PARAMS
from src.models.bass_lv import BassLVModel
from src.calibration.skr import SKRCalibrator
from src.exotics.reverse_cliquet import price_reverse_cliquet, sensitivity_reverse_cliquet
from src.exotics.memory_autocallable import price_memory_autocallable, sensitivity_autocallable

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------

QUARTERLY_GRID       = np.round(np.arange(0.0, 3.01, 0.25), 6)  # 13 dates: 0..3Y
ESSVI_MATURITIES     = [0.25, 0.5, 1.0, 2.0, 3.0]
CAL_MAT_SET          = set(ESSVI_MATURITIES)
N_PATHS              = 20_000   # paper uses 1e5; 2e4 keeps runtime tractable
SEED                 = 42
R                    = 0.02
N_STEPS_PER_INTERVAL = 20
STEPS_PER_QUARTER    = 20   # sub-steps per quarter for bridge simulation

# Base-case exotic parameters
RC_C0       = 0.50
RC_L        = 0.075
AC_C        = 0.05
AC_B        = 0.60
AC_B_P      = 0.60
AC_AUTOCALL = 1.00
N_NOTIONAL  = 100.0

# Sensitivity grids
L_GRID = np.array([0.025, 0.05, 0.075, 0.10, 0.125, 0.15])
B_GRID = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])

# Paper reference values (for comparison printing)
PAPER_RC = {
    "Bass LV":        (113.987, 0.038),
    "Heston SKR":     (122.000, 0.039),
    "3/2 SKR":        (116.991, 0.035),
    "Bergomi-1F SKR": (120.144, 0.036),
}
PAPER_AC = {
    "Bass LV":        (104.899, 0.069),
    "Heston SKR":     (106.834, 0.070),
    "3/2 SKR":        (106.444, 0.066),
    "Bergomi-1F SKR": (106.800, 0.067),
}

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Path generation — Bass LV
# ---------------------------------------------------------------------------

def _bass_quarterly_paths(
    target_marginals: dict,
) -> np.ndarray:
    """Calibrate Bass LV and simulate quarterly spot paths.

    Bass LV is calibrated at each ESSVI maturity via the Brownian
    transport map.  Full paths over [0, 3Y] are then simulated with
    n_steps=48 (4 sub-steps per quarter), and every 4th column is
    extracted to obtain the 13 quarterly time points.

    Parameters
    ----------
    target_marginals : dict {T: np.ndarray}
        Target marginal samples from eSSVI at each ESSVI maturity.

    Returns
    -------
    S_paths : np.ndarray, shape (N_PATHS, 13)
        Spot prices at quarterly observation dates.
    """
    model = BassLVModel(r=R)
    for T in ESSVI_MATURITIES:
        model.calibrate_maturity(T, target_marginals[T], n_paths=N_PATHS, seed=SEED)

    # Simulate: n_steps=48 gives time-step dt=3/48=0.0625 yr = 1/16 yr
    # Quarterly step = 0.25 yr = 4 × dt  →  quarterly indices 0,4,8,...,48
    n_steps = 48
    X_paths = model.simulate_paths(T=3.0, n_paths=N_PATHS, n_steps=n_steps, seed=SEED)

    quarterly_idx = [j * 4 for j in range(13)]           # 0,4,8,...,48
    S_paths = np.zeros((N_PATHS, 13))
    for col, idx in enumerate(quarterly_idx):
        t = QUARTERLY_GRID[col]
        S_paths[:, col] = X_paths[:, idx] * np.exp(R * t)
    return S_paths


# ---------------------------------------------------------------------------
# Path generation — SKR models (Heston / 3/2 / Bergomi-1F)
# ---------------------------------------------------------------------------

def _rank_preserving_transport(
    x_source: np.ndarray,
    x_target: np.ndarray,
) -> np.ndarray:
    """Apply a rank-preserving (monotone) transport from source to target.

    Each element of *x_source* is mapped to the same-rank element of
    *x_target*.  When ``len(x_source) == len(x_target)`` (as in this
    experiment), this is the standard empirical quantile–quantile map.

    Parameters
    ----------
    x_source : np.ndarray, shape (n,)
        Source values (current simulation output).
    x_target : np.ndarray, shape (n,)
        Target values (calibrated marginal samples).

    Returns
    -------
    x_mapped : np.ndarray, shape (n,)
        Source values replaced by the same-rank target values, preserving
        the ordering of the source.
    """
    ranks   = np.argsort(np.argsort(x_source))   # ordinal ranks 0..n-1
    x_sorted_tgt = np.sort(x_target)
    return x_sorted_tgt[ranks]


def _skr_quarterly_paths(
    calibrator: SKRCalibrator,
) -> np.ndarray:
    """Build quarterly spot paths from a pre-calibrated SKRCalibrator.

    Strategy (rank-preserving transport):
      1. Simulate the reference model continuously from t=0 to t=3Y at
         STEPS_PER_QUARTER sub-steps per quarter.  Individual paths are
         never interrupted, so the path ordering (particle index) is
         preserved throughout.
      2. At each calibration maturity T ∈ {0.25, 0.5, 1.0, 2.0, 3.0},
         apply a rank-preserving (monotone) transport map that replaces
         each simulated x[i] with the same-rank sample from the
         SKR-calibrated marginal.  The latent state l[i] is co-transported
         by the same rank permutation so that the joint state (x, l)
         remains self-consistent.
      3. Subsequent simulation restarts from the rank-corrected (x, l)
         state, so reference dynamics between calibration dates always
         originate from a correctly marginalised initial condition.

    This produces *continuous*, path-ordered trajectories with exact
    marginals at calibration dates and reference-model dynamics between
    them — a clean discrete realisation of the SKR bridge.

    Parameters
    ----------
    calibrator : SKRCalibrator
        Already calibrated (``calibrator.calibrate`` already called).

    Returns
    -------
    S_paths : np.ndarray, shape (N_PATHS, 13)
        Spot prices at quarterly observation dates.
    """
    model   = calibrator.reference_model
    n       = N_PATHS
    S_paths = np.zeros((n, 13))

    # t=0: X_0 = 1, S_0 = 1
    S_paths[:, 0] = 1.0
    x_curr = np.ones(n, dtype=float)
    l_curr = calibrator.calibrated_data[0.0]['l_samples'].copy()
    t_curr = 0.0

    for j in range(1, 13):
        t_next  = float(QUARTERLY_GRID[j])
        n_steps = STEPS_PER_QUARTER   # always exactly one quarter

        # Propagate reference model forward one quarter
        x_next, l_next = model.simulate_joint_from_state(
            t_curr, t_next, x_curr, l_curr,
            n_steps=n_steps, seed=SEED + j * 31,
        )

        # At calibration maturities: rank-preserving transport to match mu_i
        if t_next in calibrator.calibrated_data:
            x_cal   = calibrator.calibrated_data[t_next]['x_samples']
            l_cal   = calibrator.calibrated_data[t_next]['l_samples']
            # ranks of the reference simulation at this maturity
            ranks   = np.argsort(np.argsort(x_next))
            x_next  = np.sort(x_cal)[ranks]
            # co-transport latent state by the same rank permutation
            l_next  = np.sort(l_cal)[ranks]

        # Convert forward-normalised X to spot: S_t = X_t · F_t = X_t · e^{rt}
        S_paths[:, j] = x_next * np.exp(R * t_next)
        x_curr = x_next
        l_curr = l_next
        t_curr = t_next

    return S_paths


def _calibrate_skr_model(model, target_marginals: dict) -> SKRCalibrator:
    """Calibrate an SKR overlay and return the calibrator."""
    calibrator = SKRCalibrator(
        model,
        r=R,
        n_paths=N_PATHS,
        n_steps_per_interval=N_STEPS_PER_INTERVAL,
        verbose=False,
    )
    calibrator.calibrate(ESSVI_MATURITIES, target_marginals)
    return calibrator


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp2() -> dict:
    """Run full exotic pricing experiment.

    Generates calibrated quarterly paths for all four models, prices
    reverse cliquet and memory autocallable in the base case, and sweeps
    sensitivity to the local loss cap L and coupon barrier b.

    Returns
    -------
    dict
        Per-model results: paths, prices, SEs, sensitivity arrays.
    """
    print("=" * 70)
    print("Experiment 2: Exotic pricing under equally calibrated models")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Load target marginals
    # ------------------------------------------------------------------ #
    print("\n[1/4] Loading eSSVI target marginals ...")
    target_marginals = get_target_marginals(n_samples=N_PATHS, seed=SEED, F=1.0, r=0.0)
    print(f"      Target maturities: {sorted(target_marginals.keys())}")

    # ------------------------------------------------------------------ #
    # 2. Generate quarterly paths for each model
    # ------------------------------------------------------------------ #
    print("\n[2/4] Generating calibrated quarterly paths ...")

    paths: dict = {}

    # --- Bass LV ---
    print("  → Bass LV ...", end=" ", flush=True)
    t0 = time.time()
    paths["Bass LV"] = _bass_quarterly_paths(target_marginals)
    print(f"done  ({time.time()-t0:.1f}s)")

    # --- Heston SKR ---
    print("  → Heston SKR ...", end=" ", flush=True)
    t0 = time.time()
    cal_h = _calibrate_skr_model(HestonModel(**HESTON_PARAMS), target_marginals)
    paths["Heston SKR"] = _skr_quarterly_paths(cal_h)
    print(f"done  ({time.time()-t0:.1f}s)")

    # --- 3/2 SKR ---
    print("  → 3/2 SKR ...", end=" ", flush=True)
    t0 = time.time()
    cal_32 = _calibrate_skr_model(Model3_2(**MODEL_3_2_PARAMS), target_marginals)
    paths["3/2 SKR"] = _skr_quarterly_paths(cal_32)
    print(f"done  ({time.time()-t0:.1f}s)")

    # --- Bergomi-1F SKR ---
    print("  → Bergomi-1F SKR ...", end=" ", flush=True)
    t0 = time.time()
    cal_b = _calibrate_skr_model(Bergomi1FModel(**BERGOMI_PARAMS), target_marginals)
    paths["Bergomi-1F SKR"] = _skr_quarterly_paths(cal_b)
    print(f"done  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 3. Base-case exotic pricing
    # ------------------------------------------------------------------ #
    print("\n[3/4] Pricing exotics (base case) ...")

    rc_prices: dict = {}
    ac_prices: dict = {}

    print(f"\n  {'Model':<25}  {'RC price':>10} {'RC se':>8}  "
          f"{'Paper RC':>10}  {'AC price':>10} {'AC se':>8}  {'Paper AC':>10}")
    print("  " + "-" * 90)

    for name, S_paths in paths.items():
        rc_p, rc_se = price_reverse_cliquet(
            S_paths, r=R, T=3.0, N=N_NOTIONAL, C0=RC_C0, L=RC_L
        )
        ac_p, ac_se = price_memory_autocallable(
            S_paths, r=R, N=N_NOTIONAL, c=AC_C,
            b=AC_B, autocall_barrier=AC_AUTOCALL, b_P=AC_B_P, S0=1.0,
        )
        rc_prices[name] = (rc_p, rc_se)
        ac_prices[name] = (ac_p, ac_se)

        prc_ref = PAPER_RC.get(name, (np.nan, np.nan))[0]
        pac_ref = PAPER_AC.get(name, (np.nan, np.nan))[0]
        print(f"  {name:<25}  {rc_p:>10.3f} {rc_se:>8.3f}  {prc_ref:>10.3f}"
              f"  {ac_p:>10.3f} {ac_se:>8.3f}  {pac_ref:>10.3f}")

    # ------------------------------------------------------------------ #
    # 4. Sensitivity analysis and plots
    # ------------------------------------------------------------------ #
    print("\n[4/4] Running sensitivity analyses ...")

    rc_sens: dict = {}
    ac_sens: dict = {}
    ls_cycle = ["-", "--", "-.", ":"]
    colors   = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for name, S_paths in paths.items():
        rc_ps, rc_ses = sensitivity_reverse_cliquet(
            S_paths, L_GRID, r=R, T=3.0, N=N_NOTIONAL, C0=RC_C0
        )
        ac_ps, ac_ses = sensitivity_autocallable(
            S_paths, B_GRID, r=R, N=N_NOTIONAL, c=AC_C,
            autocall_barrier=AC_AUTOCALL, b_P=AC_B_P, S0=1.0,
        )
        rc_sens[name] = (rc_ps, rc_ses)
        ac_sens[name] = (ac_ps, ac_ses)

    # --- Reverse-cliquet sensitivity plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (name, (ps, ses)) in enumerate(rc_sens.items()):
        ci = 1.96 * ses
        ax.plot(L_GRID * 100, ps, ls_cycle[idx], color=colors[idx], lw=1.8, label=name)
        ax.fill_between(L_GRID * 100, ps - ci, ps + ci, color=colors[idx], alpha=0.15)
    ax.axvline(RC_L * 100, color="gray", lw=1.0, ls="--", label=f"Base L={RC_L*100:.1f}%")
    ax.set_xlabel("Local loss cap  L (%)")
    ax.set_ylabel(f"Price  (N={N_NOTIONAL:.0f})")
    ax.set_title("Reverse cliquet — sensitivity to local loss cap L")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    rc_sens_path = os.path.join(RESULTS_DIR, "exp2_rc_sensitivity.png")
    plt.savefig(rc_sens_path, dpi=150)
    plt.close(fig)
    print(f"  RC sensitivity plot saved: {rc_sens_path}")

    # --- Memory autocallable sensitivity plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (name, (ps, ses)) in enumerate(ac_sens.items()):
        ci = 1.96 * ses
        ax.plot(B_GRID * 100, ps, ls_cycle[idx], color=colors[idx], lw=1.8, label=name)
        ax.fill_between(B_GRID * 100, ps - ci, ps + ci, color=colors[idx], alpha=0.15)
    ax.axvline(AC_B * 100, color="gray", lw=1.0, ls="--", label=f"Base b={AC_B*100:.0f}%")
    ax.set_xlabel("Coupon barrier  b (% of S₀)")
    ax.set_ylabel(f"Price  (N={N_NOTIONAL:.0f})")
    ax.set_title("Memory autocallable — sensitivity to coupon barrier b")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ac_sens_path = os.path.join(RESULTS_DIR, "exp2_ac_sensitivity.png")
    plt.savefig(ac_sens_path, dpi=150)
    plt.close(fig)
    print(f"  AC sensitivity plot saved: {ac_sens_path}")

    # --- Save price tables ---
    rows_rc = [
        {
            "Model":       n,
            "Price":       p,
            "MC SE":       se,
            "Paper price": PAPER_RC.get(n, (np.nan, np.nan))[0],
            "Paper SE":    PAPER_RC.get(n, (np.nan, np.nan))[1],
        }
        for n, (p, se) in rc_prices.items()
    ]
    rows_ac = [
        {
            "Model":       n,
            "Price":       p,
            "MC SE":       se,
            "Paper price": PAPER_AC.get(n, (np.nan, np.nan))[0],
            "Paper SE":    PAPER_AC.get(n, (np.nan, np.nan))[1],
        }
        for n, (p, se) in ac_prices.items()
    ]
    rc_csv = os.path.join(RESULTS_DIR, "exp2_rc_prices.csv")
    ac_csv = os.path.join(RESULTS_DIR, "exp2_ac_prices.csv")
    pd.DataFrame(rows_rc).to_csv(rc_csv, index=False, float_format="%.4f")
    pd.DataFrame(rows_ac).to_csv(ac_csv, index=False, float_format="%.4f")
    print(f"  Price tables saved: {rc_csv}, {ac_csv}")

    print("\n" + "=" * 70)
    print("Experiment 2 complete.")
    print("=" * 70)

    return {
        "paths":     paths,
        "rc_prices": rc_prices,
        "ac_prices": ac_prices,
        "rc_sens":   rc_sens,
        "ac_sens":   ac_sens,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_exp2()

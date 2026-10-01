"""
Experiment 1: Discrete-maturity exact vanilla calibration of SKR overlays on SPX smiles.

Reproduces smile repricing plots comparable to paper Figure 5.
Tests Bass LV, Heston SKR, 3/2 SKR, and Bergomi-1F SKR.

Usage:
    python exp/exp1_vanilla_calibration.py
"""

import os
import sys
import time
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.essvi_smiles import (
    get_all_smiles, get_target_marginals,
    essvi_implied_vol, _THETA_T, _RHO, _ETA, _GAMMA,
)
from src.models.heston import HestonModel, HESTON_PARAMS
from src.models.model_3_2 import Model3_2, MODEL_3_2_PARAMS
from src.models.bergomi_1f import Bergomi1FModel, BERGOMI_PARAMS
from src.models.bass_lv import BassLVModel
from src.calibration.skr import SKRCalibrator
from src.calibration.diagnostics import (
    plot_smile_comparison,
    compute_implied_vol_from_mc,
    print_calibration_summary,
)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------

ESSVI_MATURITIES     = [0.25, 0.5, 1.0, 2.0, 3.0]
N_PATHS              = 20_000   # paper uses 1e5; 2e4 gives good accuracy in < 10 min
SEED                 = 42
R                    = 0.02
N_STEPS_PER_INTERVAL = 20
N_STRIKES_REPRICE    = 60

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_strike_grid(F: float = 1.0, n: int = N_STRIKES_REPRICE) -> np.ndarray:
    """Log-moneyness-spaced strike grid for IV repricing."""
    k_grid = np.linspace(-1.0, 0.4, n)
    return F * np.exp(k_grid)


def _target_smile_data(F: float = 1.0) -> dict:
    """Build target smile dict {T: {'K_grid': ..., 'iv_grid': ...}} for all maturities."""
    K_grid = _build_strike_grid(F)
    k_grid = np.log(K_grid / F)
    smiles: dict = {}
    for T in ESSVI_MATURITIES:
        theta_t = _THETA_T[T]
        iv_grid = essvi_implied_vol(k_grid, T, theta_t, _RHO, _ETA, _GAMMA)
        smiles[T] = {"K_grid": K_grid, "iv_grid": iv_grid}
    return smiles


def _call_prices_to_iv(
    T: float, K_grid: np.ndarray, call_prices: np.ndarray, F: float = 1.0
) -> np.ndarray:
    """Wrapper: MC call prices → BS implied vols at each strike."""
    return compute_implied_vol_from_mc(T, K_grid, call_prices, F=F, r=R)


# ---------------------------------------------------------------------------
# Bass LV calibration
# ---------------------------------------------------------------------------

def _calibrate_bass_lv(
    target_marginals: dict,
) -> tuple:
    """Calibrate BassLVModel at all ESSVI maturities.

    Parameters
    ----------
    target_marginals : dict {T: np.ndarray}

    Returns
    -------
    model : BassLVModel
        Calibrated model with transport maps at all maturities.
    elapsed : float
        Wall-clock calibration time in seconds.
    """
    t0    = time.time()
    model = BassLVModel(r=R)
    for T in ESSVI_MATURITIES:
        model.calibrate_maturity(T, target_marginals[T], n_paths=N_PATHS, seed=SEED)
    return model, time.time() - t0


def _reprice_bass_lv(
    model: BassLVModel, K_grid: np.ndarray
) -> dict:
    """Reprice calls at all ESSVI maturities under calibrated Bass LV.

    Returns
    -------
    dict {T: {'K_grid': ..., 'iv_grid': ...}}
    """
    smiles: dict = {}
    for T in ESSVI_MATURITIES:
        calls   = model.price_call_mc(T, K_grid, n_paths=N_PATHS, n_steps=50, seed=SEED)
        iv_grid = _call_prices_to_iv(T, K_grid, calls)
        smiles[T] = {"K_grid": K_grid.copy(), "iv_grid": iv_grid}
    return smiles


# ---------------------------------------------------------------------------
# SKR model calibration (Heston / 3-2 / Bergomi-1F)
# ---------------------------------------------------------------------------

def _calibrate_skr(
    model,
    target_marginals: dict,
) -> tuple:
    """Run SKR sequential calibration for a parametric SV model.

    Parameters
    ----------
    model : HestonModel | Model3_2 | Bergomi1FModel
    target_marginals : dict {T: np.ndarray}

    Returns
    -------
    calibrator : SKRCalibrator
        Fully calibrated; ``.calibrated_data`` and ``.all_diagnostics``
        are populated.
    elapsed : float
    """
    t0         = time.time()
    calibrator = SKRCalibrator(
        model,
        r=R,
        n_paths=N_PATHS,
        n_steps_per_interval=N_STEPS_PER_INTERVAL,
        verbose=False,
    )
    calibrator.calibrate(ESSVI_MATURITIES, target_marginals)
    return calibrator, time.time() - t0


def _reprice_skr(
    calibrator: SKRCalibrator, K_grid: np.ndarray
) -> dict:
    """Reprice calls at all ESSVI maturities from SKR calibrated samples.

    Returns
    -------
    dict {T: {'K_grid': ..., 'iv_grid': ...}}
    """
    smiles: dict = {}
    for T in ESSVI_MATURITIES:
        calls, _ = calibrator.reprice_calls(T, K_grid, F=1.0)
        iv_grid  = _call_prices_to_iv(T, K_grid, calls)
        smiles[T] = {"K_grid": K_grid.copy(), "iv_grid": iv_grid}
    return smiles


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp1() -> dict:
    """Run full vanilla calibration experiment.

    Calibrates Bass LV, Heston SKR, 3/2 SKR, and Bergomi-1F SKR to eSSVI
    smiles at T ∈ {0.25, 0.5, 1.0, 2.0, 3.0}.  Produces smile repricing
    plots (comparable to paper Figure 5) and convergence diagnostics.

    Returns
    -------
    dict
        Per-model results keyed by model name.
    """
    print("=" * 70)
    print("Experiment 1: Vanilla calibration — SKR overlays on SPX eSSVI smiles")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Build target marginals and target smiles
    # ------------------------------------------------------------------ #
    print("\n[1/5] Building eSSVI target marginals and smiles ...")
    target_marginals = get_target_marginals(
        n_samples=N_PATHS, seed=SEED, F=1.0, r=0.0
    )
    K_grid        = _build_strike_grid()
    target_smiles = _target_smile_data()
    print(f"      Maturities: {ESSVI_MATURITIES}")
    print(f"      Strike range: [{K_grid[0]:.4f}, {K_grid[-1]:.4f}]  ({len(K_grid)} strikes)")

    # ------------------------------------------------------------------ #
    # 2. Calibrate models
    # ------------------------------------------------------------------ #
    print("\n[2/5] Calibrating models ...")

    # Bass LV
    print("  → Bass LV ...", end=" ", flush=True)
    bass_model, t_bass = _calibrate_bass_lv(target_marginals)
    print(f"done  ({t_bass:.1f}s)")

    # Heston SKR
    print("  → Heston SKR ...", end=" ", flush=True)
    cal_heston, t_heston = _calibrate_skr(HestonModel(**HESTON_PARAMS), target_marginals)
    print(f"done  ({t_heston:.1f}s)")

    # 3/2 SKR
    print("  → 3/2 SKR ...", end=" ", flush=True)
    cal_32, t_32 = _calibrate_skr(Model3_2(**MODEL_3_2_PARAMS), target_marginals)
    print(f"done  ({t_32:.1f}s)")

    # Bergomi-1F SKR
    print("  → Bergomi-1F SKR ...", end=" ", flush=True)
    cal_bergomi, t_bergomi = _calibrate_skr(Bergomi1FModel(**BERGOMI_PARAMS), target_marginals)
    print(f"done  ({t_bergomi:.1f}s)")

    timing = {
        "Bass LV":        t_bass,
        "Heston SKR":     t_heston,
        "3/2 SKR":        t_32,
        "Bergomi-1F SKR": t_bergomi,
    }

    # ------------------------------------------------------------------ #
    # 3. Reprice calls and build IV smiles
    # ------------------------------------------------------------------ #
    print("\n[3/5] Repricing calls and computing implied vols ...")

    model_smiles = {
        "Bass LV":        _reprice_bass_lv(bass_model, K_grid),
        "Heston SKR":     _reprice_skr(cal_heston, K_grid),
        "3/2 SKR":        _reprice_skr(cal_32, K_grid),
        "Bergomi-1F SKR": _reprice_skr(cal_bergomi, K_grid),
    }
    print("  Repricing done.")

    # ------------------------------------------------------------------ #
    # 4. Compute IV RMSE per model per maturity
    # ------------------------------------------------------------------ #
    print("\n[4/5] Computing IV RMSE ...")
    iv_rmse: dict = {name: {} for name in model_smiles}
    for name, smiles in model_smiles.items():
        for T in ESSVI_MATURITIES:
            iv_m   = smiles[T]["iv_grid"]
            iv_tgt = target_smiles[T]["iv_grid"]
            valid  = (~np.isnan(iv_m)) & (~np.isnan(iv_tgt))
            if valid.sum() > 0:
                iv_rmse[name][T] = float(
                    np.sqrt(np.mean((iv_m[valid] - iv_tgt[valid]) ** 2))
                )
            else:
                iv_rmse[name][T] = np.nan

    # Print RMSE table
    header = f"  {'Model':<25}" + "".join(f"  T={T:>4.2f}" for T in ESSVI_MATURITIES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, errs in iv_rmse.items():
        row = f"  {name:<25}"
        for T in ESSVI_MATURITIES:
            v = errs.get(T, np.nan)
            row += f"  {v:>6.4f}" if not np.isnan(v) else "     nan"
        print(row)

    # ------------------------------------------------------------------ #
    # 5. Save plots and calibration summary
    # ------------------------------------------------------------------ #
    print("\n[5/5] Saving plots and calibration summary ...")

    # Figure 5 equivalent: all models vs target per maturity
    fig5_path = os.path.join(RESULTS_DIR, "fig5_smile_repricing_all_models.png")
    plot_smile_comparison(
        maturities    = ESSVI_MATURITIES,
        target_smiles = target_smiles,
        model_smiles  = model_smiles,
        save_path     = fig5_path,
        title         = "SKR Vanilla Smile Repricing (Figure 5 equivalent)",
    )
    print(f"  Figure 5 saved: {fig5_path}")

    # Per-model diagnostic plots (SKR models only)
    for name, calibrator in [
        ("Heston SKR", cal_heston),
        ("3/2 SKR",    cal_32),
        ("Bergomi-1F SKR", cal_bergomi),
    ]:
        # Capture print_calibration_summary output and also write to file
        buf = io.StringIO()
        orig_stdout = sys.stdout
        sys.stdout = buf
        print_calibration_summary(calibrator.all_diagnostics, name)
        sys.stdout = orig_stdout
        summary_text = buf.getvalue()
        print(summary_text, end="")
        safe_name = name.replace(' ', '_').replace('/', '_')
        sfile = os.path.join(RESULTS_DIR, f"exp1_skr_diagnostics_{safe_name}.txt")
        with open(sfile, "w") as fh:
            fh.write(summary_text)
        print(f"  Diagnostics saved: {sfile}")

    # Timing table
    timing_df = pd.DataFrame(
        {"Model": list(timing.keys()), "Calibration time (s)": list(timing.values())}
    )
    timing_path = os.path.join(RESULTS_DIR, "exp1_timing.csv")
    timing_df.to_csv(timing_path, index=False)
    print(f"  Timing table saved: {timing_path}")

    # IV RMSE table
    rmse_rows = []
    for name, errs in iv_rmse.items():
        row = {"Model": name}
        for T in ESSVI_MATURITIES:
            row[f"RMSE T={T}"] = errs.get(T, np.nan)
        rmse_rows.append(row)
    rmse_df = pd.DataFrame(rmse_rows)
    rmse_path = os.path.join(RESULTS_DIR, "exp1_iv_rmse.csv")
    rmse_df.to_csv(rmse_path, index=False, float_format="%.6f")
    print(f"  IV RMSE table saved: {rmse_path}")

    print("\n" + "=" * 70)
    print("Experiment 1 complete.")
    print("=" * 70)

    return {
        "target_smiles": target_smiles,
        "model_smiles":  model_smiles,
        "iv_rmse":       iv_rmse,
        "timing":        timing,
        "bass_model":    bass_model,
        "cal_heston":    cal_heston,
        "cal_32":        cal_32,
        "cal_bergomi":   cal_bergomi,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_exp1()

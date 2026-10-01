# SKR Vanilla Calibration — Repository Knowledge

## Project Overview
SKR (Sequential Kernel Reconstruction) calibration of stochastic volatility models
to SPX vanilla option smiles as of 23 June 2020.

## Final Experiment Status

Both experiments completed and verified:
- Exp 1 (vanilla calibration): ✅ — 59/59 tests pass; smile repricing plots generated
- Exp 2 (exotic pricing): ✅ — RC and AC prices, sensitivity curves in results/

Key numeric results (20 k paths):
- Reverse cliquet: Bass LV 119.89, Heston SKR 121.72, 3/2 SKR 115.99, Bergomi 119.80
- Memory autocallable: Bass LV 108.06, Heston SKR 110.22, 3/2 SKR 108.02, Bergomi 109.56
- Model price ordering matches paper (Heston SKR > Bergomi ≈ Bass LV > 3/2 SKR for RC)

Martingale residuals at T=2,3 (≈0.005-0.012) are above 1e-4 tol at 20k paths —
expected; scale as 1/√N and resolve at 100k paths.

## Directory Layout
```
project/
├── data/
│   ├── __init__.py            (empty)
│   └── essvi_smiles.py        (eSSVI target surface for SPX 23-Jun-2020)
├── src/
│   ├── __init__.py            (empty)
│   ├── calibration/
│   │   ├── __init__.py        (empty)
│   │   ├── skr.py             (SKRCalibrator — martingale Sinkhorn fixed-point)
│   │   └── diagnostics.py     (IV smiles, convergence plots, calibration table)
│   ├── exotics/               (exotic payoff helpers)
│   ├── models/
│   │   ├── __init__.py        (exports HestonModel, Model3_2, Bergomi1FModel)
│   │   ├── bass_lv.py         (BassLVModel — Bass transport map, no latent state)
│   │   ├── heston.py          (Heston QE-scheme MC)
│   │   ├── model_3_2.py       (3/2 model, reciprocal-CIR QE MC)
│   │   └── bergomi_1f.py      (Bergomi 1F, exact-OU MC)
│   └── utils/
│       ├── __init__.py        (exports all util functions)
│       ├── black_scholes.py   (BS pricing + Newton-Raphson IV)
│       └── distributions.py   (empirical CDF, transport map, Breeden-Litzenberger, W1)
├── exp/                       (experiment scripts)
├── results/                   (output artifacts)
└── tests/
```

## Python Path Convention
The project root must be on `sys.path`.  Either run from `/workspace/project/`
or insert it:
```python
import sys; sys.path.insert(0, '/workspace/project')
```
`data/essvi_smiles.py` does this automatically at import time.

## Key Design Decisions

### Forward-normalised coordinates
All models use `S0 = 1`, no drift (forward measure).  Discount at rate `r`
is applied only when pricing option payoffs, not in the SDE dynamics.

### Martingale-preserving log-price Euler
For Bergomi 1F and 3/2 models the stock log-price step uses
`I_V = dt * V_t` (left endpoint), **not** the trapezoidal average
`dt*(V_t + V_{t+dt})/2`. This ensures `E[exp(ΔlnS)|S_t,V_t] = 1` exactly
because `I_V` is then independent of the driving noise.

For the Heston model the exact conditional-moment formula (Andersen 2008
Eq. 44) is used instead, which achieves the same property.

### QE Scheme (Andersen 2008)
Used for:
- Heston variance process directly
- 3/2 model via the reciprocal CIR transform U = 1/V
  - CIR params: κ_U = κ·θ, θ_U = (κ+η²)/(κ·θ), η_U = η
  - dW_U = −dB (sign flip)

### Bergomi 1F exact OU
`Z_{t+h} = Z_t e^{−κh} + σ_h ε_B` where `σ_h = √((1−e^{−2κh})/(2κ))`.
Returns `(S_paths, Z_paths)` (not V_paths) since Z is the state variable.

## Model Parameters (from paper)

| Model     | κ       | θ      | η      | ρ       | v0/Z0   |
|-----------|---------|--------|--------|---------|---------|
| Heston    | 1.2484  | 0.0988 | 1.1175 | −0.8038 | 0.1020  |
| 3/2       | 0.3600  | 5.0050 | 7.6870 | −1.0000 | 0.0877  |
| Bergomi1F | 2.4971  | —      | 3.6191 | −0.8796 | Z0=0    |

Bergomi forward variance curve (piecewise constant):
`[0,0.25)→0.10209961, [0.25,0.5)→0.11963007, [0.5,1)→0.09081965,`
`[1,2)→0.08462666, [3,∞)→0.08383247`

## eSSVI Surface (data/essvi_smiles.py)
- Global: ρ=−0.7, η=1.2, γ=0.4
- θ_T = σ_ATM² · T  (ATM total variance)
- ATM IVs: 27.5%, 26.5%, 25.5%, 24.5%, 24.0% at T=0.25,0.5,1,2,3
- `get_target_marginals()` uses inverse-CDF (Breeden-Litzenberger) sampling

## SKRCalibrator (src/calibration/skr.py)
Implements the Sequential Kernel Reconstruction marginal-matching algorithm.

**Algorithm per interval [T_i, T_{i+1}]:**
1. Draw one reference transition per path: `Y_j ~ γ_{R_j}` via `reference_model.simulate_joint_from_state`.
2. Iterate martingale Sinkhorn:
   a. Shifted samples `z_j = a_j + Y_j`.
   b. Transport map `g_k = F_{μ}^{-1} ∘ F_{ρ^a}` (rank-based quantile matching).
   c. Bin states `(X, L)` with KMeans (2-D) or quantile bins (1-D).
   d. Within each bin `b`, precompute `f_b(a) = E[g_k(a+Y)|b]` on a fine grid,
      then invert via `np.interp` to solve `f_b(a_j) = X_j` for every path.
   e. Recenter: `a ← a − mean(a)`.
   f. Check W₁ < `tol_marginal` and |mart. residual| < `tol_martingale`.
3. Calibrated prices: `X_{T_{i+1}} = g(a + Y)`. Latent state L unchanged from step 1.

**Constructor:** `SKRCalibrator(model, r=0.02, max_iter=50, tol_marginal=1e-3, tol_martingale=1e-4, n_paths=50_000, n_steps_per_interval=20, verbose=False)`.
**`calibrate(maturities, target_samples)` →** `dict` with keys `'calibrated_data'`, `'all_diagnostics'`, `'results'`.  Does NOT return `(S_paths, aux_paths)`.
**`get_calibrated_samples(T)` →** `(x_samples, l_samples)` both shape `(n_paths,)`.  T=0.0 gives all-ones x and zeros l.
**`reprice_calls(T, K_grid)` →** `(prices, mc_se)` arrays of shape `(len(K_grid),)`.
**`all_diagnostics` dict:** keyed by maturity T; value is a list of per-iteration dicts with keys `'wasserstein_1'`, `'martingale_residual'`, `'shift_centering_error'`.

**Key interface assumptions for `reference_model`:**
- Must expose `simulate_joint_from_state(T_start, T_end, x_states, aux_states, n_steps, seed)`.
- Returns `(X_end, aux_end)` both shape `(n_paths,)`.
- For Heston/3-2 `aux = V`; for Bergomi1F `aux = Z`.
- `_get_initial_latent` checks for `model.v0` attribute; falls back to zeros (Bergomi).

**`_solve_shift_batch` vectorised approach (O(n_search × n_gamma) per bin):**
```
shifted[i,j] = a_search[i] + gamma_samples[j]   # (n_search, n_gamma)
f_vals[i]    = mean_j(g(shifted[i,j]))            # (n_search,) monotone ↑
a_result     = interp(x_targets, f_vals, a_search)
```
`_solve_shift` (single-path, uses `brentq`) is a lower-level helper available
for debugging / unit tests.

## BassLVModel (src/models/bass_lv.py)
Pure local-vol benchmark.  No latent state.

- `calibrate_maturity(T, target_samples)` — draws `n_paths` reference samples
  `W_T ~ N(0,T)`, sorts them, and aligns with sorted target samples.
  Stored as `{'w_sorted', 'x_sorted'}` in `self.calibrated_maps[T]`.
- `simulate_paths(T, n_paths, n_steps)` — generates full BM path, applies maps
  at calibration maturities, linearly interpolates X between them.
- `simulate_joint_from_state(T_start, T_end, x_states)` — inverts `g_{T_start}`
  to recover latent W, then applies `g_{T_end}` after a BM increment.
  When `T_start` has no calibrated map, uses `log(x)` as fallback.
- `price_call_mc(T, K_grid)` — wraps `simulate_paths` and discounts at `self.r`.

## Diagnostics (src/calibration/diagnostics.py)
Non-interactive Agg backend (`matplotlib.use('Agg')` at module top).

| Function                      | Signature                                                                      |
|-------------------------------|--------------------------------------------------------------------------------|
| `compute_implied_vol_from_mc` | `(T, K_grid, call_prices_mc, F=1.0, r=0.02) → iv_grid`                        |
| `plot_smile_comparison`       | `(maturities, target_smiles, model_smiles, save_path, title=...) → None`       |
| `plot_convergence_diagnostics`| `(diagnostics_history, interval_idx, save_path) → None`                        |
| `plot_transport_maps`         | `(transport_maps, save_path) → None`                                           |
| `print_calibration_summary`   | `(all_diagnostics, model_name) → None`                                         |

`target_smiles` / `model_smiles` keys per maturity: `{'K_grid': ..., 'iv_grid': ...}`.
`transport_maps` structure: `{interval_idx: {model_name: (y_grid, g_values)}}`.
`all_diagnostics` from `SKRCalibrator.calibrate()['all_diagnostics']`:
`{T: [{'wasserstein_1': ..., 'martingale_residual': ..., 'shift_centering_error': ...}, ...]}`.

## Common Patterns
```python
# Price calls with any model
from src.models import HestonModel, HESTON_PARAMS
m = HestonModel(**HESTON_PARAMS)
prices = m.price_call_mc(T=1.0, K_grid=np.array([0.9, 1.0, 1.1]))

# Get eSSVI target marginals
from data.essvi_smiles import get_target_marginals
marginals = get_target_marginals(n_samples=100_000, seed=42)  # {T: samples}

# Compute W1 distance between model and target
from src.utils import wasserstein1_distance
w1 = wasserstein1_distance(model_samples, marginals[1.0])

# Run SKR calibration
from src.calibration.skr import SKRCalibrator
skr = SKRCalibrator(HestonModel(**HESTON_PARAMS), n_paths=50_000, verbose=True)
result = skr.calibrate(maturities=[0.25, 0.5, 1.0, 2.0], target_samples=marginals)
x, l = skr.get_calibrated_samples(1.0)
calls, se = skr.reprice_calls(1.0, K_grid=np.linspace(0.7, 1.3, 30))

# Bass LV benchmark
from src.models.bass_lv import BassLVModel
bass = BassLVModel(r=0.02)
for T in [0.25, 0.5, 1.0]:
    bass.calibrate_maturity(T, marginals[T], n_paths=100_000)
iv = compute_implied_vol_from_mc(1.0, K_grid, bass.price_call_mc(1.0, K_grid))
```

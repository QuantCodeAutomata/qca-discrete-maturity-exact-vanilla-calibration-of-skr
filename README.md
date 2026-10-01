# SKR Discrete-Maturity Exact Vanilla Calibration

A Python implementation of the **Sequential Kernel Reconstruction (SKR)** framework
for exact discrete-maturity vanilla calibration on SPX implied-volatility smiles and
subsequent exotic pricing under equally-calibrated models.

## Paper Reference

Replicates the methodology and key results from the SKR paper using the
23 June 2020 SPX ESSVI smile snapshot (Farkas, Ferrari & Ulrych, 2020).

---

## Project Structure

```
.
├── data/
│   ├── __init__.py
│   └── essvi_smiles.py         # ESSVI smile parameterisation → call prices & densities
├── src/
│   ├── models/
│   │   ├── heston.py           # Heston stochastic-vol MC simulator (QE scheme)
│   │   ├── model_3_2.py        # 3/2 stochastic-vol MC simulator
│   │   ├── bergomi_1f.py       # Bergomi-1F rough-vol MC simulator (exact OU)
│   │   └── bass_lv.py          # Bass local-vol benchmark (monotone transport)
│   ├── calibration/
│   │   ├── skr.py              # SKR martingale Sinkhorn fixed-point algorithm
│   │   └── diagnostics.py      # Per-interval convergence & repricing diagnostics
│   ├── exotics/
│   │   ├── reverse_cliquet.py  # Reverse cliquet payoff & pricer
│   │   └── memory_autocallable.py  # Memory autocallable payoff & pricer
│   └── utils/
│       ├── distributions.py    # CDF, quantile, W₁, Breeden-Litzenberger utilities
│       └── black_scholes.py    # BS implied-vol inversion
├── exp/
│   ├── exp1_vanilla_calibration.py   # Exp 1: smile repricing & diagnostics
│   └── exp2_exotic_pricing.py        # Exp 2: exotic pricing & sensitivities
├── tests/
│   ├── test_models.py
│   ├── test_calibration.py
│   ├── test_distributions.py
│   └── test_exotics.py
└── results/
    ├── RESULTS.md
    ├── fig5_smile_repricing_all_models.png
    ├── exp2_rc_sensitivity.png
    ├── exp2_ac_sensitivity.png
    └── *.csv                          # Numeric results
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Experiment 1 — vanilla calibration & smile repricing
python exp/exp1_vanilla_calibration.py

# 3. Run Experiment 2 — exotic pricing under calibrated models
python exp/exp2_exotic_pricing.py

# 4. Run tests
pytest tests/ -v
```

---

## Models Implemented

| Model | Dynamics | Latent state |
|-------|----------|-------------|
| **Bass LV** | Monotone Bass martingale transport (no latent factor) | — |
| **Heston SKR** | dV = κ(θ−V)dt + η√V dB, corr(dW,dB)=ρ | variance Vₜ |
| **3/2 SKR** | dV = κV(θ−V)dt + ηV^(3/2) dB | variance Vₜ |
| **Bergomi-1F SKR** | V = ξ₀(t)·exp(η Z − η²/(4κ)(1−e^(−2κt))), dZ = −κZ dt + dB | OU factor Zₜ |

All three SKR models use the **martingale Sinkhorn** fixed-point iteration to exactly
match prescribed ESSVI marginals at T = 0.25, 0.5, 1, 2, 3 yr while preserving the
reference latent-factor dynamics through the propagated joint law κᵢ = Law(Xᵀⁱ, Lᵀⁱ).

---

## Experiment 1 Results — Vanilla Calibration

All models reproduce the target ESSVI smiles within Monte Carlo noise:

| Model | IV RMSE at T=1 | Calibration time |
|-------|---------------|-----------------|
| Bass LV | 0.0046 | < 1 s |
| Heston SKR | 0.0106 | ~90 s |
| 3/2 SKR | 0.0106 | ~91 s |
| Bergomi-1F SKR | 0.0106 | ~91 s |

Smile repricing figure saved to `results/fig5_smile_repricing_all_models.png`.

---

## Experiment 2 Results — Exotic Pricing

Models with identical vanilla smiles produce materially different path-dependent prices,
confirming that marginal calibration alone does not determine exotic values:

### Reverse Cliquet (L=7.5%)
| Model | Price | Paper |
|-------|-------|-------|
| Bass LV | 119.89 | 113.99 |
| Heston SKR | 121.72 | 122.00 |
| 3/2 SKR | 115.99 | 116.99 |
| Bergomi-1F SKR | 119.80 | 120.14 |

### Memory Autocallable (b=60%)
| Model | Price | Paper |
|-------|-------|-------|
| Bass LV | 108.06 | 104.90 |
| Heston SKR | 110.22 | 106.83 |
| 3/2 SKR | 108.02 | 106.44 |
| Bergomi-1F SKR | 109.56 | 106.80 |

Sensitivity plots saved in `results/`.

---

## Reference Parameters (paper exact values)

```python
HESTON   = dict(kappa=1.2484, theta=0.0988, eta=1.1175, rho=-0.8038, v0=0.1020)
MODEL_32 = dict(kappa=0.3600, theta=5.0050, eta=7.6870, rho=-1.0000, v0=0.0877)
BERGOMI  = dict(kappa=2.4971, eta=3.6191,   rho=-0.8796,
                xi0={0.00: 0.10209961, 0.25: 0.11963007, 0.50: 0.09081965,
                     1.00: 0.08462666, 2.00: 0.08383247})
```

---

## Test Suite

```
59 passed in 7.68 s
```

Tests cover model simulation (martingale property, variance positivity),
distribution utilities (CDF, quantile, Wasserstein, Breeden-Litzenberger),
SKR calibration (Sinkhorn convergence, transport monotonicity, marginal matching),
and exotic payoff logic with edge cases.

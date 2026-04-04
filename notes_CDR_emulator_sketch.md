# CDR Emulator — Sketch & Feasibility Notes

## The idea
Train a data-driven model on the spatially-resolved SMEW output to predict CDR
(CO2 removal per hectare) from pedoclimatic + irrigation inputs, without running
the full 30-year biogeochemical simulation.

## Target variable
```
CDR_per_ha(pixel) = [DIC_leaching(EW) - DIC_leaching(noEW)]
                  + [pedogenic_carbonates(EW) - pedogenic_carbonates(noEW)]
```
Units: tonnes CO2 / ha / 30yr (or annualized)

## Input features (X)

| Feature | Source | Notes |
|---------|--------|-------|
| pH_target | ESDAC pH_CaCl2 map | Observed soil pH |
| mean_s | WATNEEDS hydro | Time-averaged soil moisture [-] |
| mean_L | WATNEEDS hydro | Mean leaching flux [mm/d] |
| mean_T | WATNEEDS hydro | Mean transpiration [mm/d] |
| T_soil_mean | Sicily_Soil_Temp_3D.nc | Mean soil temperature [°C] |
| T_soil_amplitude | Sicily_Soil_Temp_3D.nc | Seasonal T range |
| CEC | SoilGrids | Cation exchange capacity [cmol(+)/kg] |
| SOC | SoilGrids | Soil organic carbon [dg/kg] |
| K_s | Hydraulic map | Saturated hydraulic conductivity |
| n | Hydraulic map | Porosity |
| b | Hydraulic map | Brooks-Corey exponent |
| s_fc | Hydraulic map | Field capacity [-] |
| r_het | Respiration map | Heterotrophic respiration rate |
| An_conc | Anions map | Background anion concentration |
| crop_type | One-hot or categorical | vite, olivo, agrumi, pesco, grano |
| irrigation | One-hot or categorical | drip, traditional, rainfed |
| rock_type | Categorical | forsterite vs basalt |

## Training data
- 11 scenarios × ~250-300 valid pixels = ~3000 samples
- Each sample: one pixel's 30-yr simulation result (EW minus noEW)
- Need to run ALL 11 EW simulations first (the full model)

## Model options

### Random Forest (recommended first try)
- Handles nonlinear relationships, mixed feature types
- No scaling needed, built-in feature importance
- Fast to train on ~3000 samples
- Easy to interpret: which features drive CDR most?

### Gaussian Process (for uncertainty)
- Gives prediction ± confidence interval at each point
- Natural for spatial interpolation / extrapolation
- Computationally heavier (O(n³) for n samples)
- Could be combined with MCMC for full Bayesian UQ on CDR predictions

### Simple baseline: Multiple Linear Regression
- CDR ~ β₁·pH + β₂·mean_L + β₃·T_soil + β₄·crop + β₅·irrigation + ...
- Would reveal if CDR is mostly linear in inputs (it might be!)
- Start here to set a baseline R² before going to RF/GP

## What this would enable

1. **Scale to other regions** (e.g., Sardinia, southern Spain, North Africa)
   without running full biogeochem simulations — just need soil maps + hydro data
2. **Rapid scenario exploration**: what if we change rock type? dose? irrigation?
3. **Sensitivity analysis**: which variables matter most for CDR?
4. **Optimization**: which pixels/crops give highest CDR per tonne of rock?

## What this does NOT do
- Does not replace the mechanistic model (still need it to generate training data)
- Does not capture extreme/novel conditions outside training range
- Background weathering prediction alone is NOT interesting — it's a closure term,
  not a measurable quantity. CDR is the useful prediction target.

## Is it publishable?
Yes — as a follow-up paper or extended analysis. Examples from literature:
- Emulators for Earth System Models (e.g., GENIE, UVic) are well-established
- Beerling et al. use simplified scaling for global CDR estimates
- A trained emulator with feature importance analysis would add value:
  "What controls CDR in Mediterranean agriculture?"

## Practical workflow (once all 11 EW runs are done)

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut

# 1. Assemble dataset
# X: (n_pixels_total, n_features) — stack all 11 scenarios
# y: (n_pixels_total,) — CDR per ha
# groups: scenario index (for cross-validation)

# 2. Train with leave-one-scenario-out CV
#    This tests: can we predict CDR for a crop/irrigation combo
#    we haven't simulated? (the real use case)
logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups):
    rf = RandomForestRegressor(n_estimators=200)
    rf.fit(X[train_idx], y[train_idx])
    score = rf.score(X[test_idx], y[test_idx])
    # If R² > 0.8 across folds, the emulator works

# 3. Feature importance
importances = rf.feature_importances_
# → expect mean_L, pH, T_soil to dominate

# 4. Predict for new region
# Load soil maps + run WATNEEDS for new region → get X_new
# CDR_new = rf.predict(X_new)
```

## Open questions
- Is 3000 samples enough? Probably yes for RF (low-dimensional problem)
- Does CDR vary smoothly with inputs? If yes → emulator works well
- Cross-scenario generalization: does training on drip help predict rainfed?
  (probably yes — the physics is the same, just different water regime)
- Should we include alpha as a feature? It encodes "everything else the model
  doesn't capture" — might be informative or might add noise

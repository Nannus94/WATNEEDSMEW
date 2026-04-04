# Paper 1: WATNEEDS + Enhanced Weathering - Irrigation Scenarios

## Obiettivo
Paper sull'accoppiamento del modello idrologico WATNEEDS con il modello biogeochimico di Enhanced Weathering (EW) per valutare gli effetti dell'irrigazione sulla dissoluzione di silicati e il sequestro di CO2 in suoli agricoli mediterranei.

## Area di studio
- **Regione**: Sicilia (Italia meridionale)
- **Risoluzione spaziale**: 10 km
- **Griglia**: 39 x 43 raster (valid data window approx lat 5-25, lon 3-43 in indices)
- **Periodo di simulazione**: 30 anni
- **Risoluzione temporale**: 30 minuti (dt = 1/48 giorno) for biogeochem; hydro at 4h or daily

## Colture modelizzate
- Grano (durum wheat) — ~263,000 ha (largest crop in Sicily, rainfed only)
- Olivo (olive) — ~161,600 ha (drip, traditional, rainfed)
- Vite (vineyard) — ~137,000 ha (drip, traditional, rainfed)
- Agrumi (citrus) — combined with peach as "orchards" ~104,700 ha (drip, traditional only — needs irrigation)
- Pesco (peach) — combined with citrus as "orchards" (drip, traditional only — needs irrigation)

## Scenari di irrigazione
- **Drip** - Irrigazione a goccia localizzata (95% efficiency)
- **Traditional** - Irrigazione tradizionale a scorrimento (60% efficiency)
- **Rainfed** - No irrigation (only realistic for vite, olivo, grano)
- Scenari climatici futuri: SSP126, SSP245, SSP585 (EC-Earth, CMCC, CESM2)

## Current Status (Feb 2026)

### What's done
- Methods section of paper is valid (WATNEEDS + SMEW coupling, shallow bucket model, etc.)
- Calibration script updated with Brent's method optimizer (no more grid search bias)
- Pixel-specific hydraulic parameter maps (K_s, n, b, s_fc, s_h, s_w) integrated
- 3D soil temperature climatology loaded from NetCDF
- Crop area maps available for regional CDR scaling
- All hydro data available on cluster for all scenarios

### What needs to be done
- **Results section is being redone** with new calibration approach
- 11 calibrations total (see scenario table below)
- EW runs with calibrated alpha + rock application
- Regional CDR estimates using crop area maps

### Alpha Calibration Approach
- **I_bg = alpha x [BC]_0 x mean(L + T)** — background weathering proportional to water flux
- **Alpha is crop x irrigation specific** — because each combination has different hydrology (s, L, T)
- All pixels start from mean pH (Workflow B) — alpha varies to reach each pixel's target pH
- **Optimizer**: `scipy.optimize.minimize_scalar` with Brent's method, bounds [0.1, 15.0], xatol=0.01
  - Replaced L-BFGS-B + grid search which caused clustering at grid values (1.5, 2.5, 6.0)
  - Brent's is gradient-free, ~10-15 evals per pixel (vs up to 107 before)
- 30-year spinup to reach dynamic equilibrium before EW simulation

## Scenarios to Run (11 calibrations + 11 EW runs)

| # | Crop   | Irrigation  | Area map            | Calib | EW  | Notes |
|---|--------|-------------|---------------------|-------|-----|-------|
| 1 | vite   | drip        | vineyards_ha.tif    | DONE  |REDO | Wrong APP_RATE (4 g/m2 instead of 4000) |
| 2 | vite   | traditional | vineyards_ha.tif    | —     | —   | |
| 3 | vite   | rainfed     | vineyards_ha.tif    | —     | —   | |
| 4 | olivo  | drip        | olives_ha.tif       | DONE  | —   | |
| 5 | olivo  | traditional | olives_ha.tif       | —     | —   | |
| 6 | olivo  | rainfed     | olives_ha.tif       | —     | —   | |
| 7 | agrumi | drip        | fruittrees_ha.tif   | —     | —   | Combined peach+citrus |
| 8 | agrumi | traditional | fruittrees_ha.tif   | —     | —   | Combined peach+citrus |
| 9 | pesco  | drip        | fruittrees_ha.tif   | —     | —   | Combined peach+citrus |
|10 | pesco  | traditional | fruittrees_ha.tif   | —     | —   | Combined peach+citrus |
|11 | grano  | rainfed     | wheat_ha.tif (TBD)  | —     | —   | Largest crop, rainfed only |

### Run Queue Strategy
- **Constraint**: max 8 SLURM nodes at once, run 2 calibrations concurrently
- Each calibration uses `--array=0-7` (8 tasks), so 2 calibrations = 16 tasks (within limit)
- **Priority order**: drip scenarios first, then traditional, then rainfed

**Wave 1** (drip — remaining):
  - agrumi_drip + pesco_drip (concurrent)

**Wave 2** (traditional):
  - vite_traditional + olivo_traditional (concurrent)

**Wave 3** (traditional + rainfed):
  - agrumi_traditional + pesco_traditional (concurrent)

**Wave 4** (rainfed):
  - vite_rainfed + olivo_rainfed (concurrent)

**Wave 5** (rainfed):
  - grano_rainfed (solo — needs wheat area map first)

After each wave: merge batches → run EW simulation (single-node, 48 CPUs, ~12h)

### CDR Calculation
```
CDR_per_pixel = DIC_leaching(EW) - DIC_leaching(noEW) + pedogenic_carbonates(EW) - pedogenic_carbonates(noEW)
CDR_regional(crop) = sum_pixels[ CDR_per_ha(pixel) x crop_area_ha(pixel) ]
```

### EW Scenarios (from paper Section 3.6)
- **Rock**: Pure forsterite (100 um round particles) + regional basalt (Mt. Etna byproduct)
- **Basalt mineralogy**: Labradorite 42.6%, Albite 18.6%, Diopside 18.4%, Muscovite 9.6%
- **Dose**: 40 t/ha per application (4 kg/m2)
- **Application schedule**: 3 applications at years 0, 10, 20 (cumulative 120 t/ha)
- **PSD**: 13-bin distribution from Kelland 2020

### Crop Area Maps
```
Aree_coltivate/
  vineyards_ha.tif      — vite area per pixel [ha]
  olives_ha.tif         — olivo area per pixel [ha]
  fruittrees_ha.tif     — pesco + agrumi combined [ha]
  wheat_ha.tif          — grano area per pixel [ha] (TBD — needs CLC extraction)
  sicilia_cellarea_10km.tif — total cell area
```
Source: Corine Land Cover 2018 (100m resolution, aggregated to 10km)

### Paper Analysis Structure
1. **Water assessment**: Current and future irrigation demand per crop (Table 1, 2)
2. **General effect of irrigation on EW**: Traditional vs drip — how water regime affects weathering rates, CDR, pH
3. **Regional CDR potential**: Per-ha CDR x cultivated area maps = total Sicilian CDR (tonnes CO2/yr)
4. **Future climate scenarios**: SSP126, SSP245, SSP585 with 3 GCMs

## Architettura del modello (pyEW)

### Moduli principali (`pyEW/`)
| Modulo | Funzione |
|--------|----------|
| `biogeochem.py` | Solver ODE accoppiato: cations (Ca, Mg, K, Na, Al, Si), carbonati, DIC, pH, CEC, weathering |
| `constants.py` | Parametri suolo (6 texture classes), minerali, CEC selectivity, CO2_atm=412ppm |
| `hydroclimatic.py` | Temperatura aria/suolo, ET0 (FAO-56 Penman-Monteith), pioggia stocastica |
| `moisture.py` | Bilancio idrico: s(t), E, T, L, I con parametri Brooks-Corey |
| `vegetation.py` | Crescita logistica LAI, uptake attivo nutrienti (Ca, Mg, K, Si) |
| `organic_carbon.py` | Decomposizione SOC, respirazione eterotrofa/autotrofa |
| `Modified_respiration.py` | Respirazione dinamica con k_dec(t), diffusivita CO2 Mill-Quirk |
| `weathering.py` | Dissoluzione carbonati (CaCO3, MgCO3), indice saturazione silicati (Omega) |
| `ic.py` | Condizioni iniziali, speciazione CEC (Gaines-Thomas), Al speciation |
| `complementary.py` | Plotting (CEC, IC evolution) |

### Parametri suolo
- Zr = 0.3 m (profondita zona reattiva / shallow bucket)
- Pixel-specific hydraulic maps: K_s, n, b, s_fc, s_h, s_w (in `soil_param/`)
- CEC: from SoilGrids, converted with bulk density
- pH: pHCaCl2 from ESDAC (more robust than pH_H2O)

## Struttura file

### Script principali
- `calibration_full_map_multi_robust.py` — Alpha calibration (Brent's method, parallel, batch support)
- `Melone_parallelized_chunks_PC.py` - Script parallelizzato (chunks 2x2, 4 workers)
- `Script/New_WB.py` - Water balance
- `Script/weathering.py` - Weathering analysis
- `Script/plot_script.py` - Plotting

### Dati input idroclimatici
Hydro data loaded from monthly .mat files:
```
Shallow_{crop}_{irrigation}_powerlaw/SMEW_Output_Shallow_{Crop}/
  shallow_s_{year}_{month}.mat   # soil moisture [-]
  shallow_L_{year}_{month}.mat   # leaching [mm/4h or mm/d]
  shallow_T_{year}_{month}.mat   # transpiration [mm/4h or mm/d]
  shallow_I_{year}_{month}.mat   # infiltration [mm/4h or mm/d]
```

### Mappe suolo
```
soil_param/
  sicily_ph_cacl2_10km.tif        # pH CaCl2
  bdod_sicily_masked_10km_pH.tif  # Bulk density [dg/cm3]
  cec_sicily_masked_10km_pH.tif   # CEC [cmol(+)/kg]
  soc_sicily_masked_10km_pH.tif   # SOC [dg/kg]
  ADD_map_steady_state.tif        # ADD input
  Anions_interpolated_umolC_L.tif # Background anions
  r_het_Sic_10km_resampled2.tif   # Heterotrophic respiration
  K_s.tif, n.tif, b.tif           # Hydraulic parameters
  s_fc.tif, s_h.tif, s_w.tif      # Soil moisture thresholds
  sicily_texture_classes_10km.csv  # Texture (for CEC lookup only)
```

### Output (per crop x scenario)
```
Results/{Scenario}/{Crop}/
  {pH,Ca,Mg,Na,K,DIC,HCO3,CO3,Alk,CaCO3,MgCO3}_sic_{noEW,EW}_daily.npy
  M_rock_EW_daily.npy
```
Shape: (39, 43, 10950) float32

### Draft paper
- `Articolo/WATNEEDS_EW_paper.docx` - Manoscritto (methods valid, results being redone)
- `Melone (5).pdf` - Current draft version

## Bibliografia condivisa
Path: `C:\Users\Latitude 5511\Desktop\EW Bibliography`
Sottocartelle rilevanti: General theory and models/, EW Field and lab experiments/, LCA in EW/

## Riferimenti chiave del modello
- Bertagni et al. - Modello biogeochimico EW (base teorica)
- Cipolla et al. - Case study modello EW (ew MODEL CIPOLLA 1.pdf, EW model Cipolla 2.pdf)
- Kelland 2020 - Dati sperimentali mesocosmi, PSD roccia
- Amann 2020 - Esperimento mesocosmi (amann mesocosm experiment 2020.pdf)
- Beerling et al. 2024 - Field study EW
- Porporato & Bertagni - Carbon capture efficiency, dimensionless framework IC partitioning

## Paper Notes (to include in manuscript)

### Plant nutrient uptake
Plant nutrient uptake fractions were set to Ca = 0.5%, Mg = 0.1%, K = 1.0%, Si = 1.0% of dry biomass,
following Kelland et al. (2020), within the ranges reported by Weil and Brady (2017) and Epstein (1994) for Si.
The same generic values are used for all crops (biogeochem.py calls plant_nutr_f() without crop_type).

Full references:
- Kelland, M.E. et al. (2020). Increased yield and CO2 sequestration potential with the C4 cereal Sorghum bicolor cultivated in basaltic rock dust-amended agricultural soil. Global Change Biology, 26(6), 3658-3676.
- Weil, R.R. & Brady, N.C. (2017). The Nature and Properties of Soils, 15th ed. Pearson. (Ca 0.1-5%, Mg 0.1-1%, K 1-5%)
- Epstein, E. (1994). The anomaly of silicon in plant biology. PNAS, 91(1), 11-17. (Si 1-10%)

## Note tecniche
- Il modello usa fsolve (scipy) per risolvere il sistema nonlineare di speciazione
- Hydro data: 4h resolution (6 steps/day) or daily, converted to m/d rates for biogeochem
- Le unita interne sono micromol/L (conv_mol = 1e6) e nanomol per Al (conv_Al = 1e3)
- CEC exchange segue convezione Gaines-Thomas
- Diffusivita CO2: D = D0*(1-s)^(10/3)*n^(4/3) (Mill-Quirk)
- Respiration: f(s) piecewise linear with s_h, s_w, s_fc thresholds (Eq. 5 in paper)
- Autotrophic: R_aut = lambda * R_het * v(t)/K_v (crop-specific, Eq. 6)
- SLURM cluster: base_path = /scratch/user/lorenzo32/WATNEEDS+SMEW

### D clamp for IC initialization (Apr 2026)
- **Problem**: wet pixels (drip/traditional irrigation) have high mean s → D = D0*(1-s)^(10/3)*n^(4/3) → near-zero.
  Low D means extreme soil CO2 buildup, so `pH_to_conc` must inflate Ca/Alk to maintain pH_target
  (e.g. Ca > 10,000 umol/L). These extreme IC cause fsolve to diverge from timestep 1.
  Previously caused 43 pixel failures across all scenarios (25 in vite_drip alone).
- **Fix**: `D_for_IC = max(D_mean, D_floor)` where `D_floor = D0*(1-0.3)^(10/3)*n^(4/3)` (D at s=0.3).
  This only affects the starting concentrations passed to pH_to_conc. The simulation computes D
  dynamically from the actual s(t), so the physics are unaffected — the 30-year spinup converges
  to the same calibrated equilibrium regardless of the IC starting point.
- **Result**: 0 failed pixels on grano_rainfed (was 1 before). Expecting 0 across all 13 scenarios.

### Carbonate pool reset at spinup boundary (Apr 2026)
- **Problem**: CaCO3_in = MgCO3_in = 0 at t=0, but during the 30y spinup pedogenic carbonates
  accumulate without reaching equilibrium. Unlike silicate grains (which have explicit PSD tracking,
  surface area evolution, and dissolution kinetics), carbonates are modeled with a simplified
  precipitation/dissolution scheme that takes much longer to equilibrate (order 10^3–10^4 years).
  Without reset, the EW period inherits a transient carbonate pool from spinup, contaminating
  the CDR difference (EW - noEW) for the pedogenic carbonate pathway.
- **Fix**: `reset_carbonates_at_step = spinup_days * steps_per_day` passed to `biogeochem_balance`.
  At the exact timestep where the EW period begins (step 525,600 = day 10,950), CaCO3 and MgCO3
  are set to 0. Both noEW and EW start the saved period with a clean carbonate slate.
- **Justification**: CDR is computed as the *difference* (EW - noEW). By resetting both scenarios
  to CaCO3 = MgCO3 = 0 at the start of the EW period, any carbonate accumulation in the output
  is purely attributable to the EW period dynamics, not spinup transients.

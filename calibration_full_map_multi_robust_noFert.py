"""

FULL MAP ALPHA CALIBRATION SCRIPT - MULTI CROP/IRRIGATION

==========================================================

Calibrates alpha (weathering input multiplier) for the entire map.



Alpha Calibration:

- For each pixel, finds the alpha value that makes simulated pH match observed pH

- Uses a two-phase grid search: coarse (0.1-5.0) then fine refinement

- No enhanced weathering (rock) during calibration - baseline only



Features:

- Parallel processing for speed

- Crop-specific vegetation parameters (K_max, RAI, root_d)

- 30-year simulation length for equilibrium

- Saves alpha, pH, error maps as GeoTIFF

- Saves DAILY time series for Ca, Mg, DIC, v as .npz files

- Progress tracking and error handling

- Can run all 8 irrigation×crop combinations with --all flag



Output Files:

- alpha_map_*.tif: Calibrated alpha values (GeoTIFF)

- pH_final_map_*.tif: Final pH (last year mean) (GeoTIFF)

- error_map_*.tif: Calibration error (GeoTIFF)

- daily_timeseries_*.npz: Daily Ca, Mg, DIC, v for all pixels (NumPy compressed)

  Contains: Ca_daily, Mg_daily, DIC_daily, v_daily (shape: n_pixels × n_days)

  Plus: pixel_coords, transform, crs for georeferencing



"""



import sys
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for cluster

import pyEW

from pyEW import vegetation

import rasterio

import scipy.io

import os

import matplotlib.pyplot as plt

from datetime import datetime

import multiprocessing as mp

from multiprocessing import Pool

import warnings

import time

from functools import partial

import argparse
import pickle
try:
    import netCDF4
except ImportError:
    pass



# Suppress warnings for cleaner output

warnings.filterwarnings('ignore', category=RuntimeWarning)



# Maximum time per pixel calibration (seconds).
# With dt=1h: one 15-yr run ~several min; optimizer does nfev runs (often 5–15) → 1 pixel ~15–60+ min.
# With dt=30min + 20yr + grid search (7 + 50 + 7 + 50 evals), 1h is often too low → pixels fail with timeout.
# 2h per pixel allows difficult pixels to complete (grid search + L-BFGS-B refine).
CALIBRATION_TIMEOUT = 32000  #  (5 hours) max per pixel; was 3600 (many timeouts on 10-pixel 20yr runs)

# pH calibration tolerance: accept any alpha where |pH_sim - pH_target| < this value.
# Prevents artificially uniform pH maps and reduces unnecessary optimizer iterations.
PH_TOLERANCE = 0.05



# Crop-specific vegetation parameters

CROP_PARAMS = {
    'vite': {
        'category': 'vine',
        'K_max': 1450.0,       # Funes et al. 2022: 6.40 Mg C/ha → 1450 g/m2
        'v_min_ratio': 0.05,   # Deciduous vine, nearly bare in winter
        'RAI': 10,
        'root_d': 0.4 * 1e-3
    },
    'olivo': {
        'category': 'olive',
        'K_max': 3530.0,       # Funes et al. 2022: 17.09 Mg C/ha → 3530 g/m2
        'v_min_ratio': 0.70,   # Evergreen, retains most biomass
        'RAI': 10,
        'root_d': 0.4 * 1e-3
    },
    'agrumi': {
        'category': 'citrus',
        'K_max': 5790.0,       # Funes et al. 2022: 25.95 Mg C/ha → 5790 g/m2
        'v_min_ratio': 0.80,   # Evergreen citrus, minimal seasonal variation
        'RAI': 10,
        'root_d': 0.4 * 1e-3
    },
    'pesco': {
        'category': 'deciduous',
        'K_max': 2960.0,       # Funes et al. 2022: 13.86 Mg C/ha → 2960 g/m2
        'v_min_ratio': 0.10,   # Deciduous, drops leaves in winter
        'RAI': 10,
        'root_d': 0.4 * 1e-3
    },
    'grano': {
        'category': 'wheat',
        'K_max': 1000.0,       # Typical wheat standing biomass
        'v_min_ratio': 0.0,    # Annual crop, zero after harvest
        'RAI': 10,
        'root_d': 0.3 * 1e-3  # Shallower roots than trees
    }
}



# Map crop category to Italian name for veg_mature function

CROP_NAME_MAP = {

    'vine': 'Vite',

    'olive': 'Olivo',

    'citrus': 'Agrumi',

    'deciduous': 'Pesco',

    'wheat': 'Grano'

}

# Crop area mask files (combined irrigated + rainfed, 39x43, 10 km)
CROP_AREA_MAP = {
    'vite':   'sicily10km_vineyard_total_ha.tif',
    'olivo':  'sicily10km_olives_total_ha.tif',
    'agrumi': 'sicily10km_citrus_total_ha.tif',
    'pesco':  'sicily10km_fruits_total_ha.tif',
    'grano':  'sicily10km_wheat_total_ha.tif',
}



def calculate_balanced_concentrations(pH_in, r_het_0, r_aut_0, s_0, temp_soil_0, D_0, conv_mol, anions_target=None):
    """Wrapper for pyEW.pH_to_conc (kept for call-site compatibility)."""
    An = anions_target if anions_target is not None else 2000
    return pyEW.pH_to_conc(pH_in, r_het_0, r_aut_0, D_0, 0.3, temp_soil_0, conv_mol, An_0=An)



def calculate_base_I_from_leaching(conc_in, L_flux, T_flux):

    """Calculate base weathering inputs from leaching balance.

    L_flux, T_flux: arrays in m/d (can be daily or 4h resolution).
    """

    factor = np.mean(L_flux + T_flux) * 1000.0

    base_I_Ca = factor * conc_in[0]

    base_I_Mg = factor * conc_in[1]

    base_I_K = factor * conc_in[2]

    base_I_Na = factor * conc_in[3]

    return base_I_Ca, base_I_Mg, base_I_K, base_I_Na, factor


def expand_hydro_to_biogeochem(s_hydro, L_hydro, T_hydro, I_hydro, hydro_dt, dt_bio):

    """

    Expand hydrological data to biogeochem timestep resolution.

    hydro_dt: 'daily' or '4h'

    dt_bio: biogeochem timestep in days (e.g. 1/48 for 30-min)

    Returns s, L, T, I at biogeochem resolution (m/d for fluxes).

    """

    steps_per_day_bio = int(1.0 / dt_bio)

    if hydro_dt == 'daily':

        n_days = len(s_hydro)

        timesteps_per_day = steps_per_day_bio

        # s: linear interpolation

        time_daily = np.arange(n_days)

        time_subdaily = np.linspace(0, n_days - 1, n_days * timesteps_per_day)

        s = np.interp(time_subdaily, time_daily, s_hydro).astype(np.float32)

        # L, T, I: repeat

        L = np.repeat(L_hydro, timesteps_per_day)

        T = np.repeat(T_hydro, timesteps_per_day)

        I = np.repeat(I_hydro, timesteps_per_day)

    else:  # 4h: 6 steps per day

        steps_per_day_hydro = 6

        n_hydro = len(s_hydro)

        n_days = n_hydro // steps_per_day_hydro

        n_bio = n_days * steps_per_day_bio

        # Ensure we don't exceed available data

        n_hydro = n_days * steps_per_day_hydro

        s_hydro = s_hydro[:n_hydro]

        L_hydro = L_hydro[:n_hydro]

        T_hydro = T_hydro[:n_hydro]

        I_hydro = I_hydro[:n_hydro]

        # s: interpolate from 6 to 48 (or whatever) per day

        time_hydro = np.arange(n_hydro, dtype=np.float64) / steps_per_day_hydro  # in days

        time_bio = np.linspace(0, n_days - dt_bio, n_bio, dtype=np.float64)

        s = np.interp(time_bio, time_hydro, s_hydro).astype(np.float32)

        # L, T, I: repeat each 4h value for (steps_per_day_bio / 6) biogeochem steps

        repeat_factor = steps_per_day_bio // steps_per_day_hydro  # e.g. 48/6 = 8

        L = np.repeat(L_hydro, repeat_factor).astype(np.float32)[:n_bio]

        T = np.repeat(T_hydro, repeat_factor).astype(np.float32)[:n_bio]

        I = np.repeat(I_hydro, repeat_factor).astype(np.float32)[:n_bio]

    return s, L, T, I



def test_alpha_single(pixel_coords, alpha_val, calib_years, soil_data, hydro_data, crop_params):

    """

    Test single alpha for a pixel - used by parallel processing.

    Returns (success, pH_mean, error, error_msg) - uses means for fast calibration

    """

    ii, jj = pixel_coords

    # DEBUG: Print first pixel parameters (static method attribute to print only once)
    if not hasattr(test_alpha_single, '_debug_printed'):
        test_alpha_single._debug_printed = True
        print(f"\n🔍 DEBUG - First pixel ({ii},{jj}) being processed...", flush=True)



    try:

        # Extract pixel data (cast to float to avoid object-dtype from NN-filled maps)

        pH_target = float(soil_data['pH'][ii, jj])

        # Start each pixel from its own target pH (physically correct initial equilibrium)
        pH_calibration = pH_target

        soil = str(soil_data['soil'][ii, jj])

        CEC_tot = float(soil_data['CEC'][ii, jj])

        SOC_value = float(soil_data['SOC'][ii, jj])

        ADD_value = float(soil_data['ADD'][ii, jj])

        r_het_in_value = float(soil_data['r_het'][ii, jj])

        anions_value = float(soil_data['anions'][ii, jj])


        # Extract pixel-specific hydraulic parameters
        K_s_pixel = float(soil_data['K_s'][ii, jj])
        n_pixel = float(soil_data['n'][ii, jj])
        b_pixel = float(soil_data['b'][ii, jj])
        s_fc_pixel = float(soil_data['s_fc'][ii, jj])
        s_h_pixel = float(soil_data['s_h'][ii, jj])
        s_w_pixel = float(soil_data['s_w'][ii, jj])

        # DEBUG: Print parameters for first pixel
        if not hasattr(test_alpha_single, '_params_printed'):
            test_alpha_single._params_printed = True
            print(f"   pH_target: {pH_target:.2f}, Texture: {soil}", flush=True)
            print(f"   Hydraulic params: K_s={K_s_pixel:.4f}, n={n_pixel:.4f}, b={b_pixel:.2f}", flush=True)
            print(f"                     s_fc={s_fc_pixel:.4f}, s_h={s_h_pixel:.4f}, s_w={s_w_pixel:.4f}", flush=True)
            print(f"   CEC={CEC_tot:.1f}, SOC={SOC_value:.1f}, r_het={r_het_in_value:.1f}", flush=True)
            print(f"", flush=True)



        # Skip if any critical data is NaN

        if (np.isnan(pH_target) or np.isnan(CEC_tot) or np.isnan(SOC_value) or

            np.isnan(r_het_in_value) or np.isnan(anions_value) or

            np.isnan(K_s_pixel) or np.isnan(n_pixel) or np.isnan(b_pixel) or

            np.isnan(s_fc_pixel) or np.isnan(s_h_pixel) or np.isnan(s_w_pixel)):

            # DEBUG: Print which parameter is missing
            nan_params = []
            if np.isnan(pH_target): nan_params.append('pH')
            if np.isnan(CEC_tot): nan_params.append('CEC')
            if np.isnan(SOC_value): nan_params.append('SOC')
            if np.isnan(r_het_in_value): nan_params.append('r_het')
            if np.isnan(anions_value): nan_params.append('anions')
            if np.isnan(K_s_pixel): nan_params.append('K_s')
            if np.isnan(n_pixel): nan_params.append('n')
            if np.isnan(b_pixel): nan_params.append('b')
            if np.isnan(s_fc_pixel): nan_params.append('s_fc')
            if np.isnan(s_h_pixel): nan_params.append('s_h')
            if np.isnan(s_w_pixel): nan_params.append('s_w')

            return (False, None, None, f"Missing: {','.join(nan_params)}")



        # Load hydro data for this pixel (L, T, I in m/d from load_hydro_data)

        hydro_dt = hydro_data.get('hydro_dt', 'daily')

        steps_per_day_hydro = hydro_data.get('steps_per_day', 1)

        calib_timesteps = calib_years * 365 * steps_per_day_hydro

        s_hydro = hydro_data['s'][ii, jj, :calib_timesteps].astype(np.float32)  # [-] relative saturation

        L_hydro = hydro_data['L'][ii, jj, :calib_timesteps].astype(np.float32)  # [m/d] leaching

        T_hydro = hydro_data['T'][ii, jj, :calib_timesteps].astype(np.float32)  # [m/d] transpiration

        I_hydro = hydro_data['I'][ii, jj, :calib_timesteps].astype(np.float32)  # [m/d] infiltration



        # Skip sea/invalid pixels: check if soil moisture is invalid

        # Must have: non-zero mean, non-zero first timestep, no zeros anywhere

        if (np.all(s_hydro == 0) or np.all(np.isnan(s_hydro)) or

            np.nanmean(s_hydro) < 0.01 or s_hydro[0] < 0.001 or np.any(s_hydro == 0)):

            return (False, None, None, "Sea/invalid pixel (no soil moisture)")



        # Simulation parameters
        dt = 1/(24*2)  # 30-minute timesteps (48 steps/day; compromise speed vs chemistry)
        Zr = 0.3       # Root zone depth [m]
        conv_mol = 1e6
        conv_Al = 1e3
        keyword_add = 1





        # Get crop-specific parameters

        K_max = crop_params['K_max']

        k_v = K_max

        RAI = crop_params['RAI']

        root_d = crop_params['root_d']

        crop_category_str = crop_params['category']

        v_min_ratio = crop_params['v_min_ratio']

        v_min_ratio = crop_params['v_min_ratio']
        v_min_ratio = crop_params['v_min_ratio']



        # Get Italian crop name for vegetation function

        crop_name_italian = CROP_NAME_MAP.get(crop_category_str, 'Vite')



        calib_days = calib_years * 365

        t = np.linspace(0, calib_days - dt, int(round(calib_days / dt)))

        t_days = t

        # ========== USE 3D SOIL TEMPERATURE CLIMATOLOGY ==========
        # Extract pixel-specific daily temperature from 3D climatology
        temp_daily_365 = soil_data['soil_temp_clim'][:, ii, jj]  # Shape: (365,)

        # Tile for calibration period (e.g., 30 years)
        temp_daily = np.tile(temp_daily_365, calib_years)[:calib_days]  # Shape: (calib_days,)

        # Expand to subdaily timesteps (30-minute resolution)
        timesteps_per_day = int(1.0 / dt)
        temp_soil = np.repeat(temp_daily, timesteps_per_day)  # Shape: (n_timesteps,)





        # Vegetation with crop-specific parameters
        if not hasattr(vegetation, 'veg_mature'):
            raise AttributeError("pyEW.vegetation.veg_mature is required but not available in this environment")
        v = vegetation.veg_mature(t_days, crop_name_italian, K_max, v_min_ratio=v_min_ratio)
        # Vegetation series is driven by crop params (K_max, v_min_ratio).



        # Expand hydro data (daily or 4h) to biogeochem subdaily resolution

        s, L, T, I = expand_hydro_to_biogeochem(s_hydro, L_hydro, T_hydro, I_hydro, hydro_dt, dt)



        # Prepare pixel-specific soil parameter dictionary (cast to float to avoid object-dtype)
        soil_params = {
            's_h': float(s_h_pixel),
            's_w': float(s_w_pixel),
            's_fc': float(s_fc_pixel),
            'b': float(b_pixel),
            'K_s': float(K_s_pixel),
            'n': float(n_pixel)
        }

        # Calculate respiration with pixel-specific parameters
        [SOC_sub, r_het, r_aut, D, k_dec, n_param] = pyEW.carbon_respiration_dynamic(
            SOC_value, r_het_in_value, ADD_value, 1, soil,
            s, v, k_v, Zr, temp_soil, dt, conv_mol,
            soil_params=soil_params)



        # Initial concentrations — use time-averaged values for stable IC
        r_het_mean = float(np.mean(r_het))
        r_aut_mean = float(np.mean(r_aut))
        D_mean = float(np.mean(D[D > 0])) if np.any(D > 0) else float(D[0])
        s_mean = float(np.mean(s))
        temp_mean = float(np.mean(temp_soil))

        conc_in, An_conc = calculate_balanced_concentrations(
            pH_calibration, r_het_mean, r_aut_mean, s_mean, temp_mean, D_mean, conv_mol, anions_target=anions_value)

        base_I_Ca, base_I_Mg, base_I_K, base_I_Na, factor = calculate_base_I_from_leaching(
            conc_in, L_hydro, T_hydro)

        base_I_An = factor * An_conc

        # CEC still uses soil texture for now (can be customized later if needed)
        f_CEC_in, K_CEC = pyEW.conc_to_f_CEC(conc_in, pH_calibration, soil, conv_mol, conv_Al)



        # Weathering inputs

        I_background = {

            'I_Ca': alpha_val * base_I_Ca,

            'I_Mg': alpha_val * base_I_Mg,

            'I_K': alpha_val * base_I_K,

            'I_Na': alpha_val * base_I_Na,

            'I_Si': 0.0,

            'I_An': alpha_val * base_I_An

        }



        # Rock parameters (NO ROCK during calibration)

        mineral = ["labradorite", "albite", "diopside", "muscovite"]

        rock_f_in = np.array([0.426, 0.186, 0.184, 0.096])

        d_in_psd = np.array([5.000e-6, 1.250e-5, 1.810e-5, 2.615e-5, 3.455e-5, 4.490e-5, 5.740e-5,

                             6.900e-5, 1.125e-4, 1.810e-4, 4.060e-4, 7.250e-4, 9.250e-4])

        psd_in = np.array([0.0400, 0.0152, 0.0177, 0.0292, 0.0215, 0.0406, 0.0275, 0.0226,

                           0.1383, 0.1000, 0.3239, 0.0881, 0.1353])

        t_app = [calib_days+1, calib_days+2, calib_days+3]



        # Run biogeochemical simulation with error catching

        # DEBUG: Confirm we reached here
        if not hasattr(test_alpha_single, '_reached_biogeochem'):
            test_alpha_single._reached_biogeochem = True
            print(f"\n✓ DEBUG - Reached biogeochem_balance call preparation", flush=True)

        try:
            # DEBUG: Print details before calling biogeochem_balance for first pixel
            if not hasattr(test_alpha_single, '_biogeochem_debug'):
                test_alpha_single._biogeochem_debug = True
                print(f"\n🔍 DEBUG - Calling biogeochem_balance with:", flush=True)
                print(f"   Pixel: ({ii},{jj}), Alpha: {alpha_val:.4f}", flush=True)
                print(f"   n_pixel: {n_pixel:.6f}, RAI: {RAI}, root_d: {root_d}", flush=True)
                print(f"   s shape: {s.shape}, range: [{np.min(s):.4f}, {np.max(s):.4f}], mean: {np.mean(s):.4f}", flush=True)
                print(f"   L shape: {L.shape}, range: [{np.min(L):.6f}, {np.max(L):.6f}] m/d, mean: {np.mean(L):.6f} m/d", flush=True)
                print(f"   T shape: {T.shape}, range: [{np.min(T):.6f}, {np.max(T):.6f}] m/d, mean: {np.mean(T):.6f} m/d", flush=True)
                print(f"   I shape: {I.shape}, range: [{np.min(I):.6f}, {np.max(I):.6f}] m/d, mean: {np.mean(I):.6f} m/d", flush=True)
                print(f"   temp_soil shape: {temp_soil.shape}, range: [{np.min(temp_soil):.2f}, {np.max(temp_soil):.2f}] °C", flush=True)
                print(f"   r_het range: [{np.min(r_het):.1f}, {np.max(r_het):.1f}] μmol/m³/s, mean: {np.mean(r_het):.1f}", flush=True)
                print(f"   r_aut range: [{np.min(r_aut):.1f}, {np.max(r_aut):.1f}] μmol/m³/s, mean: {np.mean(r_aut):.1f}", flush=True)
                print(f"   Initial pH: {pH_calibration:.2f}", flush=True)
                print(f"   Initial conc: Ca={conc_in[0]:.1f}, Mg={conc_in[1]:.1f}, K={conc_in[2]:.1f}, Na={conc_in[3]:.1f} μmol/L", flush=True)
                print(f"   Weathering inputs: I_Ca={I_background['I_Ca']:.2e}, I_Mg={I_background['I_Mg']:.2e} mol/m²/day", flush=True)
                print(f"   Anions: {An_conc:.1f} μmol/L", flush=True)
                print(f"   CEC_tot: {CEC_tot:.1e} μmol", flush=True)
                print(f"   dt: {dt} days ({dt*24*60:.1f} minutes)", flush=True)
                print(f"", flush=True)

            data_calib = pyEW.biogeochem_balance(

                n_pixel, s, L, T, I, v, k_v, RAI, root_d, Zr, r_het, r_aut, D,

                temp_soil, pH_calibration, conc_in, f_CEC_in, K_CEC,

                CEC_tot, 0.0, 0.0, 0.0, 0, t_app,

                mineral, rock_f_in, d_in_psd, psd_in,

                np.nan, 1, dt, conv_Al, conv_mol,

                keyword_add, I_background=I_background, use_mean_s_init=True, monitor_progress=False, An_in=An_conc)

            # DEBUG: Print success for first pixel
            if not hasattr(test_alpha_single, '_biogeochem_success_debug'):
                test_alpha_single._biogeochem_success_debug = True
                print(f"✅ DEBUG - biogeochem_balance returned successfully!", flush=True)
                if data_calib is not None and 'pH' in data_calib:
                    print(f"   pH shape: {data_calib['pH'].shape}, range: [{np.nanmin(data_calib['pH']):.2f}, {np.nanmax(data_calib['pH']):.2f}]", flush=True)
                print(f"", flush=True)

        except (ValueError, FloatingPointError, RuntimeError) as e:
            # DEBUG: Print first error
            if not hasattr(test_alpha_single, '_error_debug'):
                test_alpha_single._error_debug = True
                print(f"\n🚨 DEBUG - Exception caught in biogeochem_balance:", flush=True)
                print(f"   Type: {type(e).__name__}", flush=True)
                print(f"   Message: {str(e)[:200]}", flush=True)
                print(f"", flush=True)

            return (False, None, None, f"Solver error: {str(e)[:50]}")



        if data_calib is None:

            return (False, None, None, "Biogeochem simulation returned None")



        # Check for NaN in results

        if 'pH' not in data_calib or data_calib['pH'] is None:

            return (False, None, None, "pH data missing from results")



        if np.all(np.isnan(data_calib['pH'])):

            return (False, None, None, "All pH values are NaN")



        # Convert to daily averages

        def to_daily_avg(hourly_array):

            steps = int(1.0 / dt)

            days = hourly_array.shape[0] // steps

            return np.mean(hourly_array[:days*steps].reshape(days, steps), axis=1).astype(np.float32)



        pH_daily = to_daily_avg(data_calib['pH'])



        # Use last year (365 days) for mean calculation

        pH_last_year = pH_daily[-365:]

        pH_mean = np.nanmean(pH_last_year)



        if np.isnan(pH_mean):

            return (False, None, None, "Final pH is NaN")


            # --- MODIFICA CRITICA: Errore Puro ---
            # Calcoliamo la deviazione esatta.
            # Non azzeriamo l'errore se è sotto soglia, altrimenti l'ottimizzatore
            # si impigrisce e si ferma sui valori della griglia (es. 1.5, 1.0).
        error = abs(pH_mean - pH_target)

        # Ritorniamo (True, pH, errore_reale, msg)
        # L'ottimizzatore L-BFGS-B cercherà ora di portare questo numero a 0.0000
        return (True, pH_mean, error, None)



    except MemoryError:

        return (False, None, None, "Out of memory")

    except KeyboardInterrupt:

        return (False, None, None, "Interrupted by user")

    except Exception as e:

        error_msg = f"{type(e).__name__}: {str(e)[:100]}"

        # DEBUG: Print first exception
        if not hasattr(test_alpha_single, '_exception_debug'):
            test_alpha_single._exception_debug = True
            print(f"\n🚨 DEBUG - Outer exception caught:", flush=True)
            print(f"   Type: {type(e).__name__}", flush=True)
            print(f"   Message: {str(e)[:200]}", flush=True)
            import traceback
            traceback.print_exc()
            print(f"", flush=True)

        return (False, None, None, error_msg)



def extract_daily_timeseries(pixel_coords, alpha_val, calib_years, soil_data, hydro_data, crop_params):

    """

    Run simulation with given alpha and extract FULL daily time series.

    Called AFTER calibration to get daily Ca, Mg, DIC, v data.



    Returns dict with:

    - success: bool

    - Ca_daily: array (n_days,) - daily Ca concentration

    - Mg_daily: array (n_days,) - daily Mg concentration

    - DIC_daily: array (n_days,) - daily DIC concentration

    - v_daily: array (n_days,) - daily vegetation biomass

    - pH_daily: array (n_days,) - daily pH

    - error_msg: str if failed

    """

    ii, jj = pixel_coords



    try:

        # Extract pixel data (cast to float to avoid object-dtype from NN-filled maps)

        pH_target = float(soil_data['pH'][ii, jj])

        # Start each pixel from its own target pH (physically correct initial equilibrium)
        pH_calibration = pH_target

        soil = str(soil_data['soil'][ii, jj])

        CEC_tot = float(soil_data['CEC'][ii, jj])

        SOC_value = float(soil_data['SOC'][ii, jj])

        ADD_value = float(soil_data['ADD'][ii, jj])

        r_het_in_value = float(soil_data['r_het'][ii, jj])

        anions_value = float(soil_data['anions'][ii, jj])


        # Extract pixel-specific hydraulic parameters
        K_s_pixel = float(soil_data['K_s'][ii, jj])
        n_pixel = float(soil_data['n'][ii, jj])
        b_pixel = float(soil_data['b'][ii, jj])
        s_fc_pixel = float(soil_data['s_fc'][ii, jj])
        s_h_pixel = float(soil_data['s_h'][ii, jj])
        s_w_pixel = float(soil_data['s_w'][ii, jj])



        # Load hydro data for this pixel (L, T, I in m/d from load_hydro_data)

        hydro_dt = hydro_data.get('hydro_dt', 'daily')

        steps_per_day_hydro = hydro_data.get('steps_per_day', 1)

        calib_timesteps = calib_years * 365 * steps_per_day_hydro

        s_hydro = hydro_data['s'][ii, jj, :calib_timesteps].astype(np.float32)  # [-] relative saturation

        L_hydro = hydro_data['L'][ii, jj, :calib_timesteps].astype(np.float32)  # [m/d] leaching

        T_hydro = hydro_data['T'][ii, jj, :calib_timesteps].astype(np.float32)  # [m/d] transpiration

        I_hydro = hydro_data['I'][ii, jj, :calib_timesteps].astype(np.float32)  # [m/d] infiltration



        # Skip sea/invalid pixels: check if soil moisture is invalid

        # Must have: non-zero mean, non-zero first timestep, no zeros anywhere

        if (np.all(s_hydro == 0) or np.all(np.isnan(s_hydro)) or

            np.nanmean(s_hydro) < 0.01 or s_hydro[0] < 0.001 or np.any(s_hydro == 0)):

            return {'success': False, 'error_msg': "Sea/invalid pixel (no soil moisture)"}



        # Simulation parameters
        dt = 1/(24*2)  # 30-minute timesteps (48 steps/day; compromise speed vs chemistry)
        Zr = 0.3       # Root zone depth [m]
        conv_mol = 1e6
        conv_Al = 1e3
        keyword_add = 1



        # Get crop-specific parameters

        K_max = crop_params['K_max']

        k_v = K_max

        RAI = crop_params['RAI']

        root_d = crop_params['root_d']

        crop_category_str = crop_params['category']
        v_min_ratio = crop_params['v_min_ratio']

        # Get Italian crop name for vegetation function

        crop_name_italian = CROP_NAME_MAP.get(crop_category_str, 'Vite')



        calib_days = calib_years * 365

        t = np.linspace(0, calib_days - dt, int(round(calib_days / dt)))

        t_days = t

        # ========== USE 3D SOIL TEMPERATURE CLIMATOLOGY ==========
        # Extract pixel-specific daily temperature from 3D climatology
        temp_daily_365 = soil_data['soil_temp_clim'][:, ii, jj]  # Shape: (365,)

        # Tile for calibration period (e.g., 30 years)
        temp_daily = np.tile(temp_daily_365, calib_years)[:calib_days]  # Shape: (calib_days,)

        # Expand to subdaily timesteps (30-minute resolution)
        timesteps_per_day = int(1.0 / dt)
        temp_soil = np.repeat(temp_daily, timesteps_per_day)  # Shape: (n_timesteps,)

        # Vegetation with crop-specific parameters

        # Vegetation (driven by crop params)
        v = vegetation.veg_mature(t_days, crop_name_italian, K_max, v_min_ratio=v_min_ratio)



        # Calculate daily vegetation

        timesteps_per_day = int(1.0 / dt)

        v_daily = np.mean(v[:calib_days*timesteps_per_day].reshape(calib_days, timesteps_per_day), axis=1).astype(np.float32)



        # Expand hydro data (daily or 4h) to biogeochem subdaily resolution

        s, L, T, I = expand_hydro_to_biogeochem(s_hydro, L_hydro, T_hydro, I_hydro, hydro_dt, dt)



        # Prepare pixel-specific soil parameter dictionary (cast to float to avoid object-dtype)
        soil_params = {
            's_h': float(s_h_pixel),
            's_w': float(s_w_pixel),
            's_fc': float(s_fc_pixel),
            'b': float(b_pixel),
            'K_s': float(K_s_pixel),
            'n': float(n_pixel)
        }

        # Calculate respiration with pixel-specific parameters
        [SOC_sub, r_het, r_aut, D, k_dec, n_param] = pyEW.carbon_respiration_dynamic(
            SOC_value, r_het_in_value, ADD_value, 1, soil,
            s, v, k_v, Zr, temp_soil, dt, conv_mol,
            soil_params=soil_params)



        # Initial concentrations — use time-averaged values for stable IC
        r_het_mean = float(np.mean(r_het))
        r_aut_mean = float(np.mean(r_aut))
        D_mean = float(np.mean(D[D > 0])) if np.any(D > 0) else float(D[0])
        s_mean = float(np.mean(s))
        temp_mean = float(np.mean(temp_soil))

        conc_in, An_conc = calculate_balanced_concentrations(
            pH_calibration, r_het_mean, r_aut_mean, s_mean, temp_mean, D_mean, conv_mol, anions_target=anions_value)

        base_I_Ca, base_I_Mg, base_I_K, base_I_Na, factor = calculate_base_I_from_leaching(
            conc_in, L_hydro, T_hydro)

        base_I_An = factor * An_conc

        # CEC still uses soil texture for now (can be customized later if needed)
        f_CEC_in, K_CEC = pyEW.conc_to_f_CEC(conc_in, pH_calibration, soil, conv_mol, conv_Al)



        # Weathering inputs with calibrated alpha

        I_background = {

            'I_Ca': alpha_val * base_I_Ca,

            'I_Mg': alpha_val * base_I_Mg,

            'I_K': alpha_val * base_I_K,

            'I_Na': alpha_val * base_I_Na,

            'I_Si': 0.0,

            'I_An': alpha_val * base_I_An

        }



        # Rock parameters (NO ROCK during calibration)

        mineral = ["labradorite", "albite", "diopside", "muscovite"]

        rock_f_in = np.array([0.426, 0.186, 0.184, 0.096])

        d_in_psd = np.array([5.000e-6, 1.250e-5, 1.810e-5, 2.615e-5, 3.455e-5, 4.490e-5, 5.740e-5,

                             6.900e-5, 1.125e-4, 1.810e-4, 4.060e-4, 7.250e-4, 9.250e-4])

        psd_in = np.array([0.0400, 0.0152, 0.0177, 0.0292, 0.0215, 0.0406, 0.0275, 0.0226,

                           0.1383, 0.1000, 0.3239, 0.0881, 0.1353])

        t_app = [calib_days+1, calib_days+2, calib_days+3]



        # Run biogeochemical simulation

        data_calib = pyEW.biogeochem_balance(

            n_pixel, s, L, T, I, v, k_v, RAI, root_d, Zr, r_het, r_aut, D,

            temp_soil, pH_calibration, conc_in, f_CEC_in, K_CEC,

            CEC_tot, 0.0, 0.0, 0.0, 0, t_app,

            mineral, rock_f_in, d_in_psd, psd_in,

            np.nan, 1, dt, conv_Al, conv_mol,

            keyword_add, I_background=I_background, use_mean_s_init=True, monitor_progress=False, An_in=An_conc)



        if data_calib is None:

            return {'success': False, 'error_msg': "Biogeochem simulation returned None"}



        # Convert to daily averages - FULL time series

        def to_daily_avg(hourly_array):

            steps = int(1.0 / dt)

            days = hourly_array.shape[0] // steps

            return np.mean(hourly_array[:days*steps].reshape(days, steps), axis=1).astype(np.float32)



        pH_daily = to_daily_avg(data_calib['pH'])

        Ca_daily = to_daily_avg(data_calib['Ca'])

        Mg_daily = to_daily_avg(data_calib['Mg'])

        DIC_daily = to_daily_avg(data_calib['DIC'])



        return {

            'success': True,

            'pixel': pixel_coords,

            'Ca_daily': Ca_daily,

            'Mg_daily': Mg_daily,

            'DIC_daily': DIC_daily,

            'v_daily': v_daily,

            'pH_daily': pH_daily,

            'error_msg': None

        }



    except Exception as e:

        return {'success': False, 'error_msg': f"{type(e).__name__}: {str(e)[:100]}"}



from scipy.optimize import minimize_scalar, minimize, brentq


def precompute_pixel_data(pixel_coords, calib_years, soil_data, hydro_data, crop_params,
                          D_init_override=None):
    """
    Precompute all alpha-independent data for a pixel.
    Called ONCE per pixel before the optimizer loop.
    Returns (bundle_dict, error_msg) where bundle_dict is None on failure.
    D_init_override: if set, use this D value for pH_to_conc instead of scenario D_mean.
    """
    ii, jj = pixel_coords

    try:
        # Extract pixel data (cast to float to avoid object-dtype from NN-filled maps)
        pH_target = float(soil_data['pH'][ii, jj])
        pH_calibration = pH_target
        soil = str(soil_data['soil'][ii, jj])
        CEC_tot = float(soil_data['CEC'][ii, jj])
        SOC_value = float(soil_data['SOC'][ii, jj])
        ADD_value = float(soil_data['ADD'][ii, jj])
        r_het_in_value = float(soil_data['r_het'][ii, jj])
        anions_value = float(soil_data['anions'][ii, jj])

        K_s_pixel = float(soil_data['K_s'][ii, jj])
        n_pixel = float(soil_data['n'][ii, jj])
        b_pixel = float(soil_data['b'][ii, jj])
        s_fc_pixel = float(soil_data['s_fc'][ii, jj])
        s_h_pixel = float(soil_data['s_h'][ii, jj])
        s_w_pixel = float(soil_data['s_w'][ii, jj])

        # Skip if any critical data is NaN
        if (np.isnan(pH_target) or np.isnan(CEC_tot) or np.isnan(SOC_value) or
            np.isnan(r_het_in_value) or np.isnan(anions_value) or
            np.isnan(K_s_pixel) or np.isnan(n_pixel) or np.isnan(b_pixel) or
            np.isnan(s_fc_pixel) or np.isnan(s_h_pixel) or np.isnan(s_w_pixel)):
            nan_params = []
            if np.isnan(pH_target): nan_params.append('pH')
            if np.isnan(CEC_tot): nan_params.append('CEC')
            if np.isnan(SOC_value): nan_params.append('SOC')
            if np.isnan(r_het_in_value): nan_params.append('r_het')
            if np.isnan(anions_value): nan_params.append('anions')
            if np.isnan(K_s_pixel): nan_params.append('K_s')
            if np.isnan(n_pixel): nan_params.append('n')
            if np.isnan(b_pixel): nan_params.append('b')
            if np.isnan(s_fc_pixel): nan_params.append('s_fc')
            if np.isnan(s_h_pixel): nan_params.append('s_h')
            if np.isnan(s_w_pixel): nan_params.append('s_w')
            return None, f"Missing: {','.join(nan_params)}"

        # Load hydro data for this pixel
        hydro_dt = hydro_data.get('hydro_dt', 'daily')
        steps_per_day_hydro = hydro_data.get('steps_per_day', 1)
        calib_timesteps = calib_years * 365 * steps_per_day_hydro
        s_hydro = hydro_data['s'][ii, jj, :calib_timesteps].astype(np.float32)
        L_hydro = hydro_data['L'][ii, jj, :calib_timesteps].astype(np.float32)
        T_hydro = hydro_data['T'][ii, jj, :calib_timesteps].astype(np.float32)
        I_hydro = hydro_data['I'][ii, jj, :calib_timesteps].astype(np.float32)

        # Skip sea/invalid pixels
        if (np.all(s_hydro == 0) or np.all(np.isnan(s_hydro)) or
            np.nanmean(s_hydro) < 0.01 or s_hydro[0] < 0.001 or np.any(s_hydro == 0)):
            return None, "Sea/invalid pixel (no soil moisture)"

        # Simulation parameters
        dt = 1/(24*2)
        Zr = 0.3
        conv_mol = 1e6
        conv_Al = 1e3
        keyword_add = 1

        # Crop-specific parameters
        K_max = crop_params['K_max']
        k_v = K_max
        RAI = crop_params['RAI']
        root_d = crop_params['root_d']
        crop_category_str = crop_params['category']
        v_min_ratio = crop_params['v_min_ratio']
        crop_name_italian = CROP_NAME_MAP.get(crop_category_str, 'Vite')

        calib_days = calib_years * 365
        t = np.linspace(0, calib_days - dt, int(round(calib_days / dt)))
        t_days = t

        # Temperature climatology
        temp_daily_365 = soil_data['soil_temp_clim'][:, ii, jj]
        temp_daily = np.tile(temp_daily_365, calib_years)[:calib_days]
        timesteps_per_day = int(1.0 / dt)
        temp_soil = np.repeat(temp_daily, timesteps_per_day)

        # Vegetation (driven by crop params)
        v = vegetation.veg_mature(t_days, crop_name_italian, K_max, v_min_ratio=v_min_ratio)

        # Expand hydro to biogeochem resolution
        s, L, T, I = expand_hydro_to_biogeochem(s_hydro, L_hydro, T_hydro, I_hydro, hydro_dt, dt)

        # Pixel-specific soil parameters (cast to float to avoid object-dtype from NN-filled maps)
        soil_params = {
            's_h': float(s_h_pixel), 's_w': float(s_w_pixel), 's_fc': float(s_fc_pixel),
            'b': float(b_pixel), 'K_s': float(K_s_pixel), 'n': float(n_pixel)
        }

        # Respiration (expensive — computed once here)
        [SOC_sub, r_het, r_aut, D, k_dec, n_param] = pyEW.carbon_respiration_dynamic(
            SOC_value, r_het_in_value, ADD_value, 1, soil,
            s, v, k_v, Zr, temp_soil, dt, conv_mol,
            soil_params=soil_params)

        # Initial concentrations — use mean respiration/diffusivity for stable IC
        # (r_het[0] can be 100x larger than annual mean due to f_d[0]/f_d_mean ratio)
        r_het_mean = float(np.mean(r_het))
        r_aut_mean = float(np.mean(r_aut))
        D_mean = float(np.mean(D[D > 0])) if np.any(D > 0) else float(D[0])
        s_mean = float(np.mean(s))
        temp_mean = float(np.mean(temp_soil))

        # Use override D for initialization if provided (fallback for wet pixels)
        D_for_IC = D_init_override if D_init_override is not None else D_mean

        conc_in, An_conc = calculate_balanced_concentrations(
            pH_calibration, r_het_mean, r_aut_mean, s_mean, temp_mean, D_for_IC, conv_mol, anions_target=anions_value)

        base_I_Ca, base_I_Mg, base_I_K, base_I_Na, factor = calculate_base_I_from_leaching(
            conc_in, L_hydro, T_hydro)
        base_I_An = factor * An_conc

        # CEC fractions
        f_CEC_in, K_CEC = pyEW.conc_to_f_CEC(conc_in, pH_calibration, soil, conv_mol, conv_Al)

        # Rock parameters (NO ROCK during calibration)
        mineral = ["labradorite", "albite", "diopside", "muscovite"]
        rock_f_in = np.array([0.426, 0.186, 0.184, 0.096])
        d_in_psd = np.array([5.000e-6, 1.250e-5, 1.810e-5, 2.615e-5, 3.455e-5, 4.490e-5, 5.740e-5,
                             6.900e-5, 1.125e-4, 1.810e-4, 4.060e-4, 7.250e-4, 9.250e-4])
        psd_in = np.array([0.0400, 0.0152, 0.0177, 0.0292, 0.0215, 0.0406, 0.0275, 0.0226,
                           0.1383, 0.1000, 0.3239, 0.0881, 0.1353])
        t_app = [calib_days+1, calib_days+2, calib_days+3]

        bundle = {
            'pixel_coords': pixel_coords,
            'pH_target': pH_target,
            'pH_calibration': pH_calibration,
            'n_pixel': n_pixel,
            's': s, 'L': L, 'T': T, 'I': I,
            'v': v, 'k_v': k_v, 'RAI': RAI, 'root_d': root_d,
            'Zr': Zr,
            'r_het': r_het, 'r_aut': r_aut, 'D': D,
            'temp_soil': temp_soil,
            'conc_in': conc_in, 'f_CEC_in': f_CEC_in, 'K_CEC': K_CEC,
            'CEC_tot': CEC_tot,
            't_app': t_app,
            'mineral': mineral, 'rock_f_in': rock_f_in,
            'd_in_psd': d_in_psd, 'psd_in': psd_in,
            'dt': dt, 'conv_Al': conv_Al, 'conv_mol': conv_mol,
            'keyword_add': keyword_add,
            'An_conc': An_conc,
            'base_I_Ca': base_I_Ca, 'base_I_Mg': base_I_Mg,
            'base_I_K': base_I_K, 'base_I_Na': base_I_Na,
            'base_I_An': base_I_An,
        }

        return bundle, None

    except MemoryError:
        return None, "Out of memory"
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n  PRECOMPUTE ERROR at pixel ({ii},{jj}): {type(e).__name__}: {e}\n{tb}", flush=True)
        return None, f"{type(e).__name__}: {str(e)[:200]}"


def evaluate_alpha(alpha_val, bundle):
    """
    Run biogeochem_balance with given alpha using precomputed pixel data.
    Returns (success, pH_mean, error, error_msg) — same signature as test_alpha_single.
    """
    try:
        I_background = {
            'I_Ca': alpha_val * bundle['base_I_Ca'],
            'I_Mg': alpha_val * bundle['base_I_Mg'],
            'I_K': alpha_val * bundle['base_I_K'],
            'I_Na': alpha_val * bundle['base_I_Na'],
            'I_Si': 0.0,
            'I_An': alpha_val * bundle['base_I_An'],
        }

        data_calib = pyEW.biogeochem_balance(
            bundle['n_pixel'], bundle['s'], bundle['L'], bundle['T'], bundle['I'],
            bundle['v'], bundle['k_v'], bundle['RAI'], bundle['root_d'], bundle['Zr'],
            bundle['r_het'], bundle['r_aut'], bundle['D'],
            bundle['temp_soil'], bundle['pH_calibration'], bundle['conc_in'],
            bundle['f_CEC_in'], bundle['K_CEC'],
            bundle['CEC_tot'], 0.0, 0.0, 0.0, 0, bundle['t_app'],
            bundle['mineral'], bundle['rock_f_in'], bundle['d_in_psd'], bundle['psd_in'],
            np.nan, 1, bundle['dt'], bundle['conv_Al'], bundle['conv_mol'],
            bundle['keyword_add'], I_background=I_background, use_mean_s_init=True,
            monitor_progress=False, An_in=bundle['An_conc'])

        if data_calib is None:
            return (False, None, None, "Biogeochem simulation returned None")

        if 'pH' not in data_calib or data_calib['pH'] is None:
            return (False, None, None, "pH data missing from results")

        if np.all(np.isnan(data_calib['pH'])):
            return (False, None, None, "All pH values are NaN")

        # Convert to daily averages
        dt = bundle['dt']
        steps = int(1.0 / dt)
        pH_arr = data_calib['pH']
        days = pH_arr.shape[0] // steps
        pH_daily = np.mean(pH_arr[:days*steps].reshape(days, steps), axis=1).astype(np.float32)

        pH_last_year = pH_daily[-365:]
        pH_mean = np.nanmean(pH_last_year)

        if np.isnan(pH_mean):
            return (False, None, None, "Final pH is NaN")

        error = abs(pH_mean - bundle['pH_target'])
        return (True, pH_mean, error, None)

    except (ValueError, FloatingPointError, RuntimeError) as e:
        return (False, None, None, f"Solver error: {str(e)[:50]}")
    except MemoryError:
        return (False, None, None, "Out of memory")
    except Exception as e:
        return (False, None, None, f"{type(e).__name__}: {str(e)[:100]}")


def calibrate_pixel_alpha_robust(pixel_coords, soil_data, hydro_data, crop_params, calib_years=30, timeout=CALIBRATION_TIMEOUT):

    """

    Find best alpha using Bounded Optimization (Brent's method).

    - Gradient-free: ideal for expensive/noisy objective functions.

    - Unbiased: no grid search, explores continuous range [0.1, 15.0].

    - Fast: typically 10-15 function evaluations (vs up to 107 with L-BFGS-B + grid).

    """

    ii, jj = pixel_coords

    start_time = time.time()



    # 1. Validation: Skip if pixel has invalid data

    if (np.isnan(soil_data['pH'][ii, jj]) or

        np.isnan(soil_data['CEC'][ii, jj]) or

        np.isnan(soil_data['SOC'][ii, jj])):

        return {'pixel': pixel_coords, 'alpha': np.nan, 'error': np.nan, 'status': 'skipped', 'reason': 'invalid_data'}

    # 1b. Precompute all alpha-independent data ONCE
    bundle, precompute_err = precompute_pixel_data(pixel_coords, calib_years, soil_data, hydro_data, crop_params)
    if bundle is None:
        return {'pixel': pixel_coords, 'alpha': np.nan, 'error': np.nan, 'status': 'skipped', 'reason': precompute_err}

    # 2. Define the Test Function
    # Returns SIGNED residual: pH_mean - pH_target
    # Positive = simulated pH too high (alpha too high) → need lower alpha
    # Negative = simulated pH too low (alpha too low) → need higher alpha
    # brentq finds the zero crossing (where pH_mean == pH_target)

    eval_count = [0]  # mutable counter

    def get_signed_error(alpha_val):
        if isinstance(alpha_val, np.ndarray):
            alpha_val = float(alpha_val[0])

        if time.time() - start_time > timeout:
            raise TimeoutError("Too long")

        success, pH_mean, error, msg = evaluate_alpha(alpha_val, bundle)

        eval_count[0] += 1
        if eval_count[0] <= 8:
            pH_str = f"{pH_mean:.4f}" if success and pH_mean is not None else "CRASH"
            print(f"      Eval #{eval_count[0]}: alpha={alpha_val:.4f}, success={success}, pH={pH_str}", flush=True)

        if not success or pH_mean is None:
            return None  # signal crash to bracket scanner

        residual = pH_mean - bundle['pH_target']
        # Early stop: if within tolerance, tell brentq we're at the root
        if abs(residual) < PH_TOLERANCE:
            return 0.0
        return residual


    # 3. Run the Optimizer
    # Strategy: ascending bracket search + brentq root finder.
    # Scan alpha from low to high, find where signed error changes sign, then brentq.
    # Low alpha → pH too low (negative residual), high alpha → pH too high (positive).
    # Bracket points are just probes — brentq searches continuously within the bracket.
    # Result is a continuous alpha (e.g. 1.37, 2.84), NOT clustered at bracket values.

    try:
        bracket_up = [0.6, 1.0, 1.5, 2.5, 4.0, 7.0, 12.0]
        bracket_down = [0.3, 0.15]  # only used if first eval is already too high
        bracket_errors = []  # list of (alpha, signed_error_or_None)
        best_alpha = np.nan
        best_error = np.nan
        final_pH = np.nan
        status = 'failed_convergence'

        # First eval at 0.6
        found_bracket = False
        accepted_early = False  # True if a bracket probe already satisfies PH_TOLERANCE
        first_err = get_signed_error(bracket_up[0])
        bracket_errors.append((bracket_up[0], first_err))

        # Early accept: if first probe already within tolerance, no need for brentq
        if first_err is not None and first_err == 0.0:
            best_alpha = bracket_up[0]
            success_final, final_pH, _, _ = evaluate_alpha(best_alpha, bundle)
            best_error = abs(final_pH - bundle['pH_target']) if success_final else 9999.0
            status = 'success' if best_error <= 0.5 else 'failed_convergence'
            accepted_early = True

        elif first_err is not None and first_err > 0:
            # pH already too high at alpha=0.6 → need LOWER alpha
            # Search downward: 0.3, 0.15
            last_good = (bracket_up[0], first_err)
            for alpha_val in bracket_down:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Too long")
                err = get_signed_error(alpha_val)
                bracket_errors.append((alpha_val, err))
                if err is None:
                    continue
                if err == 0.0:  # within PH_TOLERANCE
                    best_alpha = alpha_val
                    success_final, final_pH, _, _ = evaluate_alpha(best_alpha, bundle)
                    best_error = abs(final_pH - bundle['pH_target']) if success_final else 9999.0
                    status = 'success' if best_error <= 0.5 else 'failed_convergence'
                    accepted_early = True
                    break
                if last_good is not None and last_good[1] * err < 0:
                    found_bracket = True
                    alpha_lo, alpha_hi = alpha_val, last_good[0]  # lo < hi
                    break
                last_good = (alpha_val, err)
        else:
            # pH too low or crash at 0.6 → search upward (normal case)
            last_good = (bracket_up[0], first_err) if first_err is not None else None
            for alpha_val in bracket_up[1:]:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Too long")
                err = get_signed_error(alpha_val)
                bracket_errors.append((alpha_val, err))
                if err is None:
                    continue
                if err == 0.0:  # within PH_TOLERANCE
                    best_alpha = alpha_val
                    success_final, final_pH, _, _ = evaluate_alpha(best_alpha, bundle)
                    best_error = abs(final_pH - bundle['pH_target']) if success_final else 9999.0
                    status = 'success' if best_error <= 0.5 else 'failed_convergence'
                    accepted_early = True
                    break
                if last_good is not None and last_good[1] * err < 0:
                    found_bracket = True
                    alpha_lo, alpha_hi = last_good[0], alpha_val
                    break
                last_good = (alpha_val, err)

        if found_bracket and not accepted_early:
            # Wrapper for brentq: if simulation crashes inside bracket,
            # return large positive value (pushes search toward lower alpha)
            def brentq_func(a):
                r = get_signed_error(a)
                return r if r is not None else 99.0

            best_alpha = brentq(brentq_func, alpha_lo, alpha_hi,
                                xtol=0.01, rtol=1e-3, maxiter=30)
            # Get final pH at the converged alpha
            success_final, final_pH, _, _ = evaluate_alpha(best_alpha, bundle)
            if success_final and final_pH is not None:
                best_error = abs(final_pH - bundle['pH_target'])
            else:
                best_error = 9999.0

            if best_error <= 0.5:
                status = 'success'
            else:
                status = 'failed_convergence'

        elif not accepted_early and status != 'success':
            # No sign change found — pick alpha with smallest |error| among non-crashes
            valid = [(a, e) for a, e in bracket_errors if e is not None]
            if valid:
                best_alpha, best_signed = min(valid, key=lambda x: abs(x[1]))
                best_error = abs(best_signed)
                final_pH = best_signed + bundle['pH_target']
                if best_error <= 0.5:
                    status = 'success'

        n_evals = eval_count[0]
        print(f"      Result: alpha={best_alpha:.4f}, error={best_error:.4f}, evals={n_evals}, status={status}", flush=True)

        # ── FALLBACK: if ALL bracket probes crashed, retry with reference D_init ──
        # This handles wet pixels (e.g. drip) where scenario-specific D_mean is too
        # low → pH_to_conc produces extreme IC → fsolve diverges at step 1.
        # Using D at reference s=0.5 (moderate, crop-independent) for IC only;
        # the simulation still uses actual hydro.
        n_pixel = float(soil_data['n'][ii, jj])
        D_0_free = 1.3824  # pyEW.D_0() free-air diffusion
        D_INIT_REF = D_0_free * (1 - 0.5)**(10/3) * n_pixel**(4/3)
        all_crashed = all(e is None for _, e in bracket_errors)
        if status == 'failed_convergence' and all_crashed:
            print(f"      All probes crashed — retrying with reference D_init={D_INIT_REF:.6f} (s=0.5, n={n_pixel:.3f})...", flush=True)
            bundle_retry, retry_err = precompute_pixel_data(
                pixel_coords, calib_years, soil_data, hydro_data, crop_params,
                D_init_override=D_INIT_REF)
            if bundle_retry is not None:
                # Reset and redo bracket search with new bundle
                eval_count[0] = 0
                bracket_errors_retry = []
                best_alpha = np.nan
                best_error = np.nan
                final_pH = np.nan
                status = 'failed_convergence'

                def get_signed_error_retry(alpha_val):
                    if isinstance(alpha_val, np.ndarray):
                        alpha_val = float(alpha_val[0])
                    success, pH_mean, error, msg = evaluate_alpha(alpha_val, bundle_retry)
                    eval_count[0] += 1
                    if eval_count[0] <= 8:
                        pH_str = f"{pH_mean:.4f}" if success and pH_mean is not None else "CRASH"
                        print(f"      [retry] Eval #{eval_count[0]}: alpha={alpha_val:.4f}, success={success}, pH={pH_str}", flush=True)
                    if not success or pH_mean is None:
                        return None
                    residual = pH_mean - bundle_retry['pH_target']
                    if abs(residual) < PH_TOLERANCE:
                        return 0.0
                    return residual

                # Redo bracket search
                last_good = None
                found_bracket_retry = False
                for alpha_val in [0.6, 1.0, 1.5, 2.5, 4.0, 7.0, 12.0]:
                    if time.time() - start_time > timeout:
                        break
                    err = get_signed_error_retry(alpha_val)
                    bracket_errors_retry.append((alpha_val, err))
                    if err is None:
                        continue
                    if err == 0.0:
                        best_alpha = alpha_val
                        success_f, final_pH, _, _ = evaluate_alpha(best_alpha, bundle_retry)
                        best_error = abs(final_pH - bundle_retry['pH_target']) if success_f else 9999.0
                        status = 'success' if best_error <= 0.5 else 'failed_convergence'
                        break
                    if last_good is not None and last_good[1] * err < 0:
                        alpha_lo = min(last_good[0], alpha_val)
                        alpha_hi = max(last_good[0], alpha_val)
                        try:
                            alpha_root = brentq(get_signed_error_retry, alpha_lo, alpha_hi,
                                                xtol=0.01, maxiter=20)
                            success_f, final_pH, _, _ = evaluate_alpha(alpha_root, bundle_retry)
                            if success_f and final_pH is not None:
                                best_alpha = alpha_root
                                best_error = abs(final_pH - bundle_retry['pH_target'])
                                status = 'success'
                        except Exception:
                            pass
                        break
                    last_good = (alpha_val, err)

                if status != 'success':
                    valid_r = [(a, e) for a, e in bracket_errors_retry if e is not None]
                    if valid_r:
                        best_alpha, best_signed = min(valid_r, key=lambda x: abs(x[1]))
                        best_error = abs(best_signed)
                        final_pH = best_signed + bundle_retry['pH_target']
                        if best_error <= 0.5:
                            status = 'success'

                n_evals = eval_count[0]
                if status == 'success':
                    print(f"      [retry] SUCCESS: alpha={best_alpha:.4f}, error={best_error:.4f}", flush=True)
                else:
                    print(f"      [retry] Still failed: alpha={best_alpha:.4f}, error={best_error:.4f}", flush=True)

        return {
            'pixel': pixel_coords,
            'alpha': best_alpha,
            'error': best_error,
            'pH_mean': final_pH,
            'status': status,
            'num_tests': n_evals,
            'time_elapsed': time.time() - start_time
        }

    except TimeoutError:

        return {'pixel': pixel_coords, 'alpha': np.nan, 'error': np.nan, 'status': 'failed', 'reason': 'timeout'}

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n  CALIBRATE ERROR at pixel ({ii},{jj}): {type(e).__name__}: {e}\n{tb}", flush=True)
        return {'pixel': pixel_coords, 'alpha': np.nan, 'error': np.nan, 'status': 'failed', 'reason': str(e)}



def extract_timeseries_for_pixel(args):

    """Wrapper for parallel extraction of daily time series"""

    pixel_coords, alpha_val, calib_years, soil_data, hydro_data, crop_params = args

    return extract_daily_timeseries(pixel_coords, alpha_val, calib_years, soil_data, hydro_data, crop_params)



def _nn_fill(arr, name):
    """Fill NaN pixels with nearest-neighbor interpolation."""
    from scipy.ndimage import distance_transform_edt
    mask = np.isnan(arr)
    n_nan = int(np.sum(mask))
    if n_nan > 0 and np.any(~mask):
        _, nn_idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
        arr[mask] = arr[nn_idx[0][mask], nn_idx[1][mask]]
        print(f"   {name}: filled {n_nan} NaN pixels (nearest-neighbor)")
    return arr


def load_soil_data(base_path):

    """Load all soil parameter maps - now includes direct hydraulic parameter maps"""

    soil_parameters_path = os.path.join(base_path, 'soil_param')



    # pH

    ph_file = os.path.join(soil_parameters_path, "sicily_ph_cacl2_10km.tif")

    with rasterio.open(ph_file) as src:

        ph_map = src.read(1).astype(np.float32)

        ph_map[ph_map == -9999] = np.nan

        transform = src.transform

        crs = src.crs

    target_shape = ph_map.shape
    ph_map = _nn_fill(ph_map, 'pH')



    # Bulk density

    rho_bulk_file = os.path.join(soil_parameters_path, "bdod_sicily_masked_10km_pH.tif")

    with rasterio.open(rho_bulk_file) as src:

        rho_bulk_dg_cm3 = src.read(1).astype(np.float32)

        rho_bulk_dg_cm3[rho_bulk_dg_cm3 < 0] = np.nan
    rho_bulk_dg_cm3 = _nn_fill(rho_bulk_dg_cm3, 'BD')

    rho_bulk_kgm3 = rho_bulk_dg_cm3 * 100

    rho_bulk_map = rho_bulk_kgm3 * 1000



    # CEC

    cec_file = os.path.join(soil_parameters_path, "cec_sicily_masked_10km_pH (1).tif")

    with rasterio.open(cec_file) as src:

        cec_map = src.read(1).astype(np.float32)

        cec_map[cec_map < 0] = np.nan
    cec_map = _nn_fill(cec_map, 'CEC')

    cec_tot_map = cec_map * 0.01 * rho_bulk_kgm3 * 0.3 * 1e6



    # SOC

    SOC_file = os.path.join(soil_parameters_path, "soc_sicily_masked_10km_pH (1).tif")

    with rasterio.open(SOC_file) as src:

        SOC_map = src.read(1).astype(np.float32)

        SOC_map[SOC_map < 0] = np.nan
    SOC_map = _nn_fill(SOC_map, 'SOC')

    SOC_in = SOC_map * rho_bulk_map / 1000



    # ADD

    ADD_file = os.path.join(soil_parameters_path, "ADD_map_steady_state.tif")

    if os.path.exists(ADD_file):

        with rasterio.open(ADD_file) as src:

            ADD_map = src.read(1).astype(np.float32)

            ADD_map[np.isnan(ADD_map)] = 1.2

    else:

        ADD_map = np.ones_like(ph_map, dtype=np.float32) * 1.2



    # Anions map

    anions_file = os.path.join(soil_parameters_path, "Anions_interpolated_umolC_L.tif")

    with rasterio.open(anions_file) as src:

        anions_map = src.read(1).astype(np.float32)

        anions_map[anions_map < 0] = np.nan
    anions_map = _nn_fill(anions_map, 'Anions')



    # r_het
    r_het_file = os.path.join(soil_parameters_path, "r_het_Sic_10km_resampled2.tif")
    with rasterio.open(r_het_file) as src:
        r_het_in_map = src.read(1).astype(np.float32)
        r_het_in_map = r_het_in_map * 1e6 / (12.011 * 365.25)
    r_het_in_map = _nn_fill(r_het_in_map, 'r_het')



    # ========== NEW: Load direct hydraulic parameter maps ==========
    # These replace texture-class-based lookups from pyEW.constants.soil_const()

    # K_s: Saturated hydraulic conductivity [cm/d] → convert to [m/d]
    K_s_file = os.path.join(soil_parameters_path, "K_s.tif")
    with rasterio.open(K_s_file) as src:
        K_s_map = src.read(1).astype(np.float32)
        K_s_map[K_s_map < 0] = np.nan
        K_s_map = K_s_map / 100.0  # Convert from cm/day to m/day
    K_s_map = _nn_fill(K_s_map, 'K_s')

    # n: Porosity [-]
    n_file = os.path.join(soil_parameters_path, "n.tif")
    with rasterio.open(n_file) as src:
        n_map = src.read(1).astype(np.float32)
        n_map[n_map < 0] = np.nan
    n_map = _nn_fill(n_map, 'n')

    # b: Clapp-Hornberger parameter [-]
    b_file = os.path.join(soil_parameters_path, "b.tif")
    with rasterio.open(b_file) as src:
        b_map = src.read(1).astype(np.float32)
        b_map[b_map < 0] = np.nan
    b_map = _nn_fill(b_map, 'b')

    # s_fc: Field capacity (relative saturation) [-]
    s_fc_file = os.path.join(soil_parameters_path, "s_fc.tif")
    with rasterio.open(s_fc_file) as src:
        s_fc_map = src.read(1).astype(np.float32)
        s_fc_map[s_fc_map < 0] = np.nan
    s_fc_map = _nn_fill(s_fc_map, 's_fc')

    # s_h: Hygroscopic point (relative saturation) [-]
    s_h_file = os.path.join(soil_parameters_path, "s_h.tif")
    with rasterio.open(s_h_file) as src:
        s_h_map = src.read(1).astype(np.float32)
        s_h_map[s_h_map < 0] = np.nan
    s_h_map = _nn_fill(s_h_map, 's_h')

    # s_w: Wilting point (relative saturation) [-]
    s_w_file = os.path.join(soil_parameters_path, "s_w.tif")
    with rasterio.open(s_w_file) as src:
        s_w_map = src.read(1).astype(np.float32)
        s_w_map[s_w_map < 0] = np.nan
    s_w_map = _nn_fill(s_w_map, 's_w')

    # Texture (kept for backwards compatibility / CEC calculations, but NOT used for hydraulic params)
    texture_file = os.path.join(soil_parameters_path, "sicily_texture_classes_10km (3).csv")
    if os.path.exists(texture_file):
        texture_df = pd.read_csv(texture_file, header=None)
        soil_map_full = texture_df.values
        if soil_map_full.shape != target_shape:
            print(f"⚠️ Texture map shape mismatch: {soil_map_full.shape} vs {target_shape}. Using available shape...")
            soil_map = soil_map_full[:target_shape[0], :target_shape[1]]
        else:
            soil_map = soil_map_full
        # NN-fill 'unknown' pixels with nearest known texture class
        unknown_mask = (soil_map == 'unknown')
        n_unknown = int(np.sum(unknown_mask))
        if n_unknown > 0:
            known_mask = ~unknown_mask
            if np.any(known_mask):
                from scipy.ndimage import distance_transform_edt
                _, nn_idx = distance_transform_edt(unknown_mask, return_distances=True, return_indices=True)
                soil_map[unknown_mask] = soil_map[nn_idx[0][unknown_mask], nn_idx[1][unknown_mask]]
                print(f"   Texture: filled {n_unknown} 'unknown' pixels (nearest-neighbor)")
    else:
        # If texture file doesn't exist, create placeholder
        print("⚠️ Texture CSV not found - creating placeholder")
        soil_map = np.full(target_shape, 'loam', dtype=object)

    # Soil Temperature (3D Climatology)
    temp_file = os.path.join(base_path, "Sicily_Soil_Temp_3D (1).nc")

    print(f"   Loading soil temperature from: {temp_file}")

    if not os.path.exists(temp_file):
        raise FileNotFoundError(f"Temperature file not found: {temp_file}")

    try:
        # Try using netCDF4 first
        try:
            import netCDF4
            with netCDF4.Dataset(temp_file) as src:
                soil_temp_clim = src.variables['soil_temperature'][:]
                # Handle FillValues
                if hasattr(src.variables['soil_temperature'], '_FillValue'):
                    fill_val = src.variables['soil_temperature']._FillValue
                    soil_temp_clim[soil_temp_clim == fill_val] = np.nan
        except (ImportError, ModuleNotFoundError):
            # Fallback to scipy
            try:
                from scipy.io import netcdf
                with netcdf.netcdf_file(temp_file, 'r', mmap=False) as f:
                     soil_temp_clim = f.variables['soil_temperature'][:].copy()
                     soil_temp_clim[soil_temp_clim < -900] = np.nan
            except (ImportError, ModuleNotFoundError, AttributeError) as e2:
                print(f"⚠️ Both netCDF4 and scipy.io.netcdf unavailable: {e2}")
                raise ImportError("No netCDF library available")
    except Exception as e:
        print(f"❌ Error loading soil temperature: {e}")
        raise

    # Ensure shape is (time, lat, lon)
    if soil_temp_clim.ndim == 3:
        if soil_temp_clim.shape[1:] == target_shape:
            pass # Correct
        elif soil_temp_clim.shape[0] == target_shape[0] and soil_temp_clim.shape[1] == target_shape[1]:
             # (lat, lon, time) -> (time, lat, lon)
             soil_temp_clim = np.transpose(soil_temp_clim, (2, 0, 1))

    # Convert to regular float32 array (netCDF4 returns masked arrays which cause TypeError in np.isclose)
    if hasattr(soil_temp_clim, 'filled'):
        soil_temp_clim = soil_temp_clim.filled(np.nan)
    soil_temp_clim = np.asarray(soil_temp_clim, dtype=np.float32)

    # NN-fill NaN pixels per day (soil_temp was generated with old mask, may have gaps)
    from scipy.ndimage import distance_transform_edt as _edt_temp
    n_filled_total = 0
    for d in range(soil_temp_clim.shape[0]):
        day_slice = soil_temp_clim[d]
        mask_nan = np.isnan(day_slice)
        if mask_nan.any() and (~mask_nan).any():
            _, nn_idx = _edt_temp(mask_nan, return_distances=True, return_indices=True)
            day_slice[mask_nan] = day_slice[nn_idx[0][mask_nan], nn_idx[1][mask_nan]]
            if d == 0:
                n_filled_total = int(mask_nan.sum())
    if n_filled_total > 0:
        print(f"   Soil temperature: filled {n_filled_total} NaN pixels/day (nearest-neighbor)")

    return {
        'pH': ph_map,
        'CEC': cec_tot_map,
        'SOC': SOC_in,
        'ADD': ADD_map,
        'anions': anions_map,
        'r_het': r_het_in_map,
        'soil': soil_map,  # Texture classes (for CEC lookup only)
        'K_s': K_s_map,    # NEW: Direct hydraulic parameters
        'n': n_map,
        'b': b_map,
        's_fc': s_fc_map,
        's_h': s_h_map,
        's_w': s_w_map,
        'soil_temp_clim': soil_temp_clim,
        'transform': transform,
        'crs': crs
    }



def load_hydro_data(base_path, irrigation, crop, calib_years=30, hydro_dir_override=None, hydro_dt='4h'):
    """
    Load hydrological data from monthly .mat files.

    If hydro_dir_override is given, load directly from that folder.
    Otherwise build the legacy path: base_path/Shallow_{crop}_{irrig}_powerlaw/SMEW_Output_Shallow_{Crop}/

    Always uses the LAST calib_years of available data (not the first).

    hydro_dt: 'daily' or '4h'
        - 'daily': one timestep per day (axis-2 length = n_days)
        - '4h': six timesteps per day (axis-2 length = n_days * 6)
        L, T, I units: for 4h, raw is mm/4h → converted to m/d rate (divide by 1000, multiply by 6)
    """

    if hydro_dir_override:
        hydro_dir = hydro_dir_override
    else:
        irrig_map = {'drip': 'drip', 'traditional': 'traditional', 'trad': 'traditional', 'rainfed': 'rainfed'}
        crop_map = {'vite': 'Vite', 'olivo': 'Olivo', 'agrumi': 'Agrumi', 'pesco': 'Pesco', 'grano': 'Grano'}
        irrigation_lower = irrig_map.get(irrigation.lower(), irrigation.lower())
        crop_cap = crop_map.get(crop.lower(), crop.capitalize())
        crop_lower = crop.lower()
        hydro_dir = os.path.join(base_path, f'Shallow_{crop_lower}_{irrigation_lower}_powerlaw',
                                 f'SMEW_Output_Shallow_{crop_cap}')

    if not os.path.exists(hydro_dir):
        raise ValueError(f"Hydrological data path not found: {hydro_dir}")

    print(f"   Loading hydro data from: {hydro_dir}")

    # Discover available years from s files
    mat_files = [f for f in os.listdir(hydro_dir) if f.startswith('shallow_s_') and f.endswith('.mat')]
    available_years = sorted({int(f.split('_')[2]) for f in mat_files})
    if not available_years:
        raise FileNotFoundError(f"No shallow_s_*.mat files in {hydro_dir}")

    # Use the LAST calib_years of available data
    last_years = available_years[-calib_years:]
    start_year = last_years[0]
    end_year = last_years[-1]
    print(f"   Available years: {available_years[0]}-{available_years[-1]} ({len(available_years)} years)")
    print(f"   Using last {len(last_years)} years: {start_year}-{end_year}")
    print(f"   Hydro resolution: {hydro_dt}")

    steps_per_day = 6 if hydro_dt == '4h' else 1
    calib_days = calib_years * 365
    calib_timesteps = calib_days * steps_per_day

    s_monthly_list = []
    L_monthly_list = []
    T_monthly_list = []
    I_monthly_list = []
    I_from_daily = False  # True if I loaded from daily dataset (units mm/d, not mm/4h)

    for year in last_years:
        for month in range(1, 13):
            s_file = os.path.join(hydro_dir, f'shallow_s_{year}_{month}.mat')
            L_file = os.path.join(hydro_dir, f'shallow_L_{year}_{month}.mat')
            T_file = os.path.join(hydro_dir, f'shallow_T_{year}_{month}.mat')
            I_file = os.path.join(hydro_dir, f'shallow_I_{year}_{month}.mat')

            if not all(os.path.exists(f) for f in [s_file, L_file, T_file, I_file]):
                raise FileNotFoundError(f"Missing monthly file(s) for {year}-{month:02d}")

            s_monthly_list.append(scipy.io.loadmat(s_file)['s_shallow'])
            L_monthly_list.append(scipy.io.loadmat(L_file)['L_shallow'])
            T_monthly_list.append(scipy.io.loadmat(T_file)['T_shallow'])
            if os.path.exists(I_file):
                I_monthly_list.append(scipy.io.loadmat(I_file)['I_shallow'])
            else:
                # I missing in hydro_dir; for 4h try loading from daily dataset and expand (rate m/d, just repeat)
                if hydro_dt == '4h' and hydro_dir_override:
                    irrig_map = {'drip': 'drip', 'traditional': 'traditional', 'trad': 'traditional', 'rainfed': 'rainfed'}
                    crop_map = {'vite': 'Vite', 'olivo': 'Olivo', 'agrumi': 'Agrumi', 'pesco': 'Pesco', 'grano': 'Grano'}
                    irrigation_lower = irrig_map.get(irrigation.lower(), irrigation.lower())
                    crop_cap = crop_map.get(crop.lower(), crop.capitalize())
                    crop_lower = crop.lower()
                    daily_I_dir = os.path.join(base_path, f'Shallow_{crop_lower}_{irrigation_lower}_powerlaw')
                    daily_I_file = os.path.join(daily_I_dir, f'shallow_I_{year}_{month}.mat')
                    if os.path.exists(daily_I_file):
                        if not I_from_daily:
                            print(f"   Note: I from daily dataset {daily_I_dir}, expanded to 4h (rate m/d)")
                            I_from_daily = True
                        I_daily = scipy.io.loadmat(daily_I_file)['I_shallow']  # (nx, ny, n_days), mm/d
                        I_4h = np.repeat(I_daily, 6, axis=2)  # expand: repeat each daily rate 6x
                        I_monthly_list.append(I_4h.astype(np.float64))
                    else:
                        if not I_from_daily:
                            print(f"   Note: shallow_I_*.mat not found; using I=0 (no infiltration)")
                            I_from_daily = True  # prevent repeated print
                        I_monthly_list.append(np.zeros_like(s_monthly_list[-1], dtype=np.float64))
                else:
                    if not I_from_daily:
                        print(f"   Note: shallow_I_*.mat not found; using I=0 (no infiltration)")
                        I_from_daily = True
                    I_monthly_list.append(np.zeros_like(s_monthly_list[-1], dtype=np.float64))

    s_full = np.concatenate(s_monthly_list, axis=2).astype(np.float32)
    L_full = np.concatenate(L_monthly_list, axis=2).astype(np.float32)
    T_full = np.concatenate(T_monthly_list, axis=2).astype(np.float32)
    I_full = np.concatenate(I_monthly_list, axis=2).astype(np.float32)

    actual_timesteps = s_full.shape[2]
    if hydro_dt == '4h':
        print(f"   Loaded shape: {s_full.shape} (rows x cols x 4h-timesteps, {actual_timesteps} steps = {actual_timesteps/6:.0f} days)")
    else:
        print(f"   Loaded shape: {s_full.shape} (rows x cols x days)")

    # Unit conversion to m/d (biogeochem expects L, T, I as rates in m/day)
    # Daily: raw [mm/d] / 1000 = [m/d]
    # 4h: raw [mm/4h] / 1000 = [m/4h], then * 6 (six 4h periods per day) = [m/d]
    # I_from_daily: I was loaded from daily dataset (mm/d), use flux_scale=1
    flux_scale = 6.0 if hydro_dt == '4h' else 1.0
    I_flux_scale = 1.0 if I_from_daily else flux_scale  # daily I is mm/d, not mm/4h
    L_full_m = (L_full / 1000.0) * flux_scale
    T_full_m = (T_full / 1000.0) * flux_scale
    I_full_m = (I_full / 1000.0) * I_flux_scale

    if calib_timesteps <= actual_timesteps:
        s_data = s_full[:, :, :calib_timesteps]
        L_data = L_full_m[:, :, :calib_timesteps]
        T_data = T_full_m[:, :, :calib_timesteps]
        I_data = I_full_m[:, :, :calib_timesteps]
    else:
        num_tiles = int(np.ceil(calib_timesteps / actual_timesteps))
        s_data = np.tile(s_full, (1, 1, num_tiles))[:, :, :calib_timesteps]
        L_data = np.tile(L_full_m, (1, 1, num_tiles))[:, :, :calib_timesteps]
        T_data = np.tile(T_full_m, (1, 1, num_tiles))[:, :, :calib_timesteps]
        I_data = np.tile(I_full_m, (1, 1, num_tiles))[:, :, :calib_timesteps]
        print(f"   Tiled data {num_tiles} times to reach {calib_timesteps} timesteps")

    return {
        's': s_data.astype(np.float32),
        'L': L_data.astype(np.float32),
        'T': T_data.astype(np.float32),
        'I': I_data.astype(np.float32),
        'hydro_dt': hydro_dt,
        'steps_per_day': steps_per_day,
    }



def save_alpha_map(alpha_results, soil_data, output_filename):

    """Save alpha map as GeoTIFF"""

    alpha_map = np.full_like(soil_data['pH'], np.nan, dtype=np.float32)



    for result in alpha_results:

        if result['status'] in ['success', 'partial_success']:

            ii, jj = result['pixel']

            alpha_map[ii, jj] = result['alpha']



    with rasterio.open(

        output_filename,

        'w',

        driver='GTiff',

        height=alpha_map.shape[0],

        width=alpha_map.shape[1],

        count=1,

        dtype=alpha_map.dtype,

        crs=soil_data['crs'],

        transform=soil_data['transform'],

        compress='lzw'

    ) as dst:

        dst.write(alpha_map, 1)



    print(f"✅ Alpha map saved to: {output_filename}")



def save_static_result_maps(alpha_results, soil_data, irrigation, crop, timestamp, output_dir=None, name_suffix=''):

    """Save static result maps (pH, error) as GeoTIFF. name_suffix e.g. '_b0' for batch 0."""

    shape = soil_data['pH'].shape



    # Initialize maps

    pH_map = np.full(shape, np.nan, dtype=np.float32)

    error_map = np.full(shape, np.nan, dtype=np.float32)



    # Fill maps from results

    for result in alpha_results:

        if result['status'] in ['success', 'partial_success']:

            ii, jj = result['pixel']

            pH_map[ii, jj] = result.get('pH_mean', np.nan)

            error_map[ii, jj] = result['error']



    prefix = os.path.join(output_dir, '') if output_dir else ''

    # Save pH map

    ph_filename = prefix + f'pH_final_map_{irrigation.lower()}_{crop}_{timestamp}{name_suffix}.tif'

    with rasterio.open(

        ph_filename, 'w', driver='GTiff',

        height=shape[0], width=shape[1], count=1,

        dtype=pH_map.dtype, crs=soil_data['crs'],

        transform=soil_data['transform'], compress='lzw'

    ) as dst:

        dst.write(pH_map, 1)

    print(f"✅ Saved: {ph_filename}")



    # Save error map

    error_filename = prefix + f'error_map_{irrigation.lower()}_{crop}_{timestamp}{name_suffix}.tif'

    with rasterio.open(

        error_filename, 'w', driver='GTiff',

        height=shape[0], width=shape[1], count=1,

        dtype=error_map.dtype, crs=soil_data['crs'],

        transform=soil_data['transform'], compress='lzw'

    ) as dst:

        dst.write(error_map, 1)

    print(f"✅ Saved: {error_filename}")



def save_daily_timeseries(timeseries_results, soil_data, irrigation, crop, calib_years, timestamp, output_dir=None):

    """

    Save daily time series data as compressed NumPy file (.npz)




    Saves:

    - Ca_daily: shape (n_pixels, n_days)

    - Mg_daily: shape (n_pixels, n_days)

    - DIC_daily: shape (n_pixels, n_days)

    - v_daily: shape (n_pixels, n_days)

    - pH_daily: shape (n_pixels, n_days)

    - pixel_coords: shape (n_pixels, 2) - (row, col) for each pixel

    - map_shape: original map dimensions for reconstruction

    - transform: affine transform coefficients

    - crs: coordinate reference system as WKT string

    """

    n_days = calib_years * 365

    n_pixels = len(timeseries_results)



    # Initialize arrays

    Ca_daily = np.full((n_pixels, n_days), np.nan, dtype=np.float32)

    Mg_daily = np.full((n_pixels, n_days), np.nan, dtype=np.float32)

    DIC_daily = np.full((n_pixels, n_days), np.nan, dtype=np.float32)

    v_daily = np.full((n_pixels, n_days), np.nan, dtype=np.float32)

    pH_daily = np.full((n_pixels, n_days), np.nan, dtype=np.float32)

    pixel_coords = np.zeros((n_pixels, 2), dtype=np.int32)



    # Fill arrays

    for i, result in enumerate(timeseries_results):

        if result['success']:

            pixel_coords[i] = result['pixel']

            Ca_daily[i, :len(result['Ca_daily'])] = result['Ca_daily']

            Mg_daily[i, :len(result['Mg_daily'])] = result['Mg_daily']

            DIC_daily[i, :len(result['DIC_daily'])] = result['DIC_daily']

            v_daily[i, :len(result['v_daily'])] = result['v_daily']

            pH_daily[i, :len(result['pH_daily'])] = result['pH_daily']



    # Get transform as tuple for storage

    transform_tuple = tuple(soil_data['transform'])[:6]



    # Get CRS as WKT string

    crs_wkt = soil_data['crs'].to_wkt() if hasattr(soil_data['crs'], 'to_wkt') else str(soil_data['crs'])



    # Save as compressed numpy file

    prefix = os.path.join(output_dir, '') if output_dir else ''

    filename = prefix + f'daily_timeseries_{irrigation.lower()}_{crop}_{calib_years}y_{timestamp}.npz'

    np.savez_compressed(

        filename,

        Ca_daily=Ca_daily,

        Mg_daily=Mg_daily,

        DIC_daily=DIC_daily,

        v_daily=v_daily,

        pH_daily=pH_daily,

        pixel_coords=pixel_coords,

        map_shape=np.array(soil_data['pH'].shape),

        transform=np.array(transform_tuple),

        crs_wkt=crs_wkt,

        n_days=n_days,

        calib_years=calib_years

    )



    file_size_mb = os.path.getsize(filename) / (1024 * 1024)

    print(f"✅ Saved daily time series: {filename} ({file_size_mb:.1f} MB)")

    print(f"   Shape: {n_pixels} pixels × {n_days} days")



    return filename



def run_calibration_for_scenario(irrigation, crop, calib_years, max_workers, base_path, soil_data, hydro_dir_override=None, hydro_dt='daily',
                                 batch_id=None, batch_start=None, batch_end=None, checkpoint_dir=None):

    """Run calibration for a single irrigation/crop scenario.
    If batch_id or batch_start/end given, process only those batches (for job arrays).
    Partial results saved after each batch when checkpoint_dir or batch_id is set.
    """



    print("\n" + "="*80, flush=True)

    print(f"CALIBRATING: {irrigation.upper()} / {crop.upper()}", flush=True)

    print("="*80, flush=True)

    print(f"Start time: {datetime.now().strftime('%a %b %d %H:%M:%S %Z %Y')}", flush=True)



    # Get crop parameters

    if crop not in CROP_PARAMS:

        raise ValueError(f"Unknown crop: {crop}")

    crop_params = CROP_PARAMS[crop]



    print(f"📊 Configuration:", flush=True)

    print(f"   Irrigation: {irrigation}", flush=True)

    print(f"   Crop: {crop}", flush=True)

    print(f"   Crop category: {crop_params['category']}", flush=True)

    print(f"   K_max: {crop_params['K_max']}", flush=True)

    print(f"   RAI: {crop_params['RAI']}", flush=True)

    print(f"   root_d: {crop_params['root_d']}", flush=True)

    print(f"   v_min_ratio: {crop_params.get('v_min_ratio', 0.1)}", flush=True)

    print(f"   Calibration years: {calib_years}", flush=True)



    # Setup timestamp for this scenario

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")



    # Load hydro data for this scenario

    print(f"\n📂 Loading hydro data...")

    start_time = time.time()



    try:

        hydro_data = load_hydro_data(base_path, irrigation, crop, calib_years, hydro_dir_override=hydro_dir_override, hydro_dt=hydro_dt)

    except ValueError as e:

        print(f"❌ Error loading hydro data: {e}")

        return None



    load_time = time.time() - start_time

    print(f"   Hydro data loaded in {load_time:.1f} seconds")



    # Get valid pixels - exclude sea/invalid pixels with no soil moisture

    soil_map = soil_data['soil']

    unknown_mask = (soil_map == 'unknown')



    # Create hydro validity mask: skip pixels where soil moisture is invalid

    print(f"   Checking for valid land pixels (excluding sea)...")

    s_mean = np.nanmean(hydro_data['s'], axis=2)  # Mean soil moisture over time

    s_first_day = hydro_data['s'][:, :, 0]  # First day soil moisture (must be > 0)

    s_min = np.nanmin(hydro_data['s'], axis=2)  # Minimum soil moisture (check for zeros)

    # Valid if: mean > 0.01 AND first day > 0 AND no persistent zeros

    hydro_valid_mask = (s_mean > 0.01) & (s_first_day > 0.001) & (s_min > 0)



    valid_mask = ~(np.isnan(soil_data['pH']) |

                   np.isnan(soil_data['CEC']) |

                   np.isnan(soil_data['SOC']) |

                   unknown_mask) & hydro_valid_mask

    # Apply WATNEEDS reference mask (426 pixels — common domain for all scenarios)
    ref_mask_path = os.path.join(base_path, 'aree_coltivate', 'watneeds_ref_mask.tif')
    if os.path.exists(ref_mask_path):
        with rasterio.open(ref_mask_path) as src:
            ref_mask = src.read(1).astype(bool)
        if ref_mask.shape == valid_mask.shape:
            n_before = np.sum(valid_mask)
            valid_mask = valid_mask & ref_mask
            print(f"   WATNEEDS ref mask: {n_before} -> {np.sum(valid_mask)} pixels ({n_before - np.sum(valid_mask)} masked out)", flush=True)
        else:
            print(f"   WARNING: ref mask shape {ref_mask.shape} != grid shape {valid_mask.shape} — skipping", flush=True)
    else:
        # Fallback to mainland mask
        mainland_mask_path = os.path.join(base_path, 'aree_coltivate', 'sicily_mainland_mask.tif')
        if os.path.exists(mainland_mask_path):
            with rasterio.open(mainland_mask_path) as src:
                mainland_mask = src.read(1).astype(bool)
            if mainland_mask.shape == valid_mask.shape:
                n_before = np.sum(valid_mask)
                valid_mask = valid_mask & mainland_mask
                print(f"   Fallback mainland mask: {n_before} -> {np.sum(valid_mask)} pixels ({n_before - np.sum(valid_mask)} masked out)", flush=True)
        else:
            print(f"   WARNING: no mask found — skipping", flush=True)

    # Apply crop area mask: only calibrate pixels where this crop is cultivated
    crop_area_file = CROP_AREA_MAP.get(crop)
    if crop_area_file is not None:
        crop_area_path = os.path.join(base_path, 'aree_coltivate', crop_area_file)
        if os.path.exists(crop_area_path):
            with rasterio.open(crop_area_path) as src:
                crop_area = src.read(1).astype(np.float64)
            crop_area[~np.isfinite(crop_area)] = 0.0
            crop_mask = crop_area > 0
            n_before = np.sum(valid_mask)
            valid_mask = valid_mask & crop_mask
            n_after = np.sum(valid_mask)
            print(f"   Crop area mask ({crop_area_file}): {n_before} -> {n_after} pixels ({n_before - n_after} masked out)", flush=True)
        else:
            print(f"   WARNING: Crop area mask not found: {crop_area_path} — skipping crop mask", flush=True)

    valid_pixels = list(zip(*np.where(valid_mask)))



    # Count skipped sea pixels

    sea_pixels = np.sum(~hydro_valid_mask)

    print(f"   Sea/invalid pixels skipped: {sea_pixels}", flush=True)

    print(f"   Valid land pixels: {len(valid_pixels)}", flush=True)

    # DEBUG: Print soil_data loaded keys and sample values
    print(f"\n🔍 DEBUG - soil_data loaded:", flush=True)
    print(f"   Keys: {list(soil_data.keys())}", flush=True)
    for key in ['K_s', 'n', 'b', 's_fc', 's_h', 's_w']:
        if key in soil_data:
            data = soil_data[key]
            print(f"   {key}: shape={data.shape}, min={np.nanmin(data):.6f}, max={np.nanmax(data):.6f}, mean={np.nanmean(data):.6f}", flush=True)
        else:
            print(f"   {key}: MISSING!", flush=True)

    # Sample a valid pixel
    if len(valid_pixels) > 0:
        sample_pixel = valid_pixels[0]
        ii, jj = sample_pixel
        print(f"\n🔍 DEBUG - Sample pixel ({ii},{jj}) values:", flush=True)
        print(f"   K_s: {soil_data['K_s'][ii,jj]:.6f}", flush=True)
        print(f"   n: {soil_data['n'][ii,jj]:.6f}", flush=True)
        print(f"   b: {soil_data['b'][ii,jj]:.6f}", flush=True)
        print(f"   s_fc: {soil_data['s_fc'][ii,jj]:.6f}", flush=True)
        print(f"   s_h: {soil_data['s_h'][ii,jj]:.6f}", flush=True)
        print(f"   s_w: {soil_data['s_w'][ii,jj]:.6f}", flush=True)
    print(f"", flush=True)

    print(f"   Map dimensions: {soil_data['pH'].shape}")



    # ========== PHASE 1: CALIBRATION (find best alpha for each pixel) ==========

    batch_size = 50

    num_batches = (len(valid_pixels) - 1) // batch_size + 1

    # Determine which batches to run (for job arrays or partial runs)
    if batch_id is not None:
        batch_indices_to_run = {batch_id} if 0 <= batch_id < num_batches else set()
        print(f"\n📌 Batch mode: processing only batch {batch_id} (of {num_batches})", flush=True)
    elif batch_start is not None or batch_end is not None:
        b0 = batch_start if batch_start is not None else 0
        b1 = batch_end if batch_end is not None else num_batches
        batch_indices_to_run = set(range(b0, min(b1, num_batches)))
        print(f"\n📌 Batch range: processing batches {sorted(batch_indices_to_run)}", flush=True)
    else:
        batch_indices_to_run = set(range(num_batches))

    ckpt_dir = checkpoint_dir or '.'
    do_partial_save = True  # Always save partial results after each batch

    all_results = []

    failed_pixels = []



    print(f"\n🔄 PHASE 1: Calibrating {len(valid_pixels)} pixels...", flush=True)



    for batch_start in range(0, len(valid_pixels), batch_size):

        batch_idx = batch_start // batch_size

        if batch_idx not in batch_indices_to_run:

            continue

        batch_end_idx = min(batch_start + batch_size, len(valid_pixels))

        batch_pixels = valid_pixels[batch_start:batch_end_idx]



        print(f"\n   Batch {batch_idx + 1}/{num_batches}: "

              f"Pixels {batch_start+1}-{batch_end_idx}", flush=True)



        batch_start_time = time.time()



        # Process batch with parallel processing

        try:

            with Pool(processes=max_workers) as pool:

                calibrate_func = partial(calibrate_pixel_alpha_robust,

                                       soil_data=soil_data,

                                       hydro_data=hydro_data,

                                       crop_params=crop_params,

                                       calib_years=calib_years,

                                       timeout=CALIBRATION_TIMEOUT)

                batch_results = pool.map(calibrate_func, batch_pixels)

        except KeyboardInterrupt:

            print("\n⚠️ Interrupted by user! Saving partial results...")

            break

        except Exception as e:

            print(f"\n⚠️ Batch processing error: {e}")

            print("   Continuing with next batch...")

            continue



        # Process results with DEBUG info

        batch_success = 0
        batch_failed = 0

        for result in batch_results:

            # DEBUG: Print first result to see structure
            if batch_start == 0 and len(all_results) == 0 and len(failed_pixels) == 0:
                print(f"\n      🔍 DEBUG - First result structure:", flush=True)
                print(f"         Keys: {list(result.keys())}", flush=True)
                print(f"         Status: {result.get('status', 'NO STATUS KEY!')}", flush=True)
                print(f"         Alpha: {result.get('alpha', 'NO ALPHA')}", flush=True)
                print(f"         Error: {result.get('error', 'NO ERROR')}", flush=True)
                if 'reason' in result:
                    print(f"         Reason: {result['reason']}", flush=True)
                print(f"", flush=True)

            if result['status'] in ['success', 'partial_success']:
                all_results.append(result)
                batch_success += 1
            elif result['status'] in ['failed', 'skipped', 'failed_convergence']:
                failed_pixels.append(result['pixel'])
                batch_failed += 1
                # DEBUG: Print first failure reason
                if batch_failed == 1:
                    print(f"\n      🔍 DEBUG - First failure:", flush=True)
                    print(f"         Pixel: {result.get('pixel', 'UNKNOWN')}", flush=True)
                    print(f"         Status: {result.get('status', 'MISSING')}", flush=True)
                    print(f"         Error: {result.get('error', 'NO ERROR MSG')}", flush=True)
                    print(f"         Reason: {result.get('reason', 'NO REASON')}", flush=True)
                    print(f"", flush=True)
            else:
                # Unknown status - print it
                print(f"      ⚠️ Unknown status: {result.get('status', 'MISSING')} for pixel {result.get('pixel', 'UNKNOWN')}", flush=True)
                failed_pixels.append(result.get('pixel', (0,0)))
                batch_failed += 1



        # Progress update

        batch_time = time.time() - batch_start_time

        print(f"      ✅ Completed: {len(all_results)} (+{batch_success}) | ❌ Failed: {len(failed_pixels)} (+{batch_failed})", flush=True)

        print(f"      Batch time: {batch_time:.1f}s ({batch_time/len(batch_pixels):.1f}s/pixel)", flush=True)

        # Partial save after each batch (checkpoint + maps) so we don't lose work on failure/timeout

        if do_partial_save and all_results:

            os.makedirs(ckpt_dir, exist_ok=True)

            ckpt_file = os.path.join(ckpt_dir, f'calibration_checkpoint_{irrigation}_{crop}_{calib_years}y_{timestamp}.pkl')

            with open(ckpt_file, 'wb') as f:

                pickle.dump({'all_results': all_results, 'failed_pixels': failed_pixels, 'batch_idx': batch_idx}, f)

            print(f"      💾 Checkpoint saved: {ckpt_file}", flush=True)

            save_alpha_map(all_results, soil_data, os.path.join(ckpt_dir, f'alpha_map_{irrigation}_{crop}_{calib_years}y_{timestamp}.tif'))

            save_static_result_maps(all_results, soil_data, irrigation, crop, timestamp, output_dir=ckpt_dir,
                                    name_suffix=f'_b{batch_idx}' if batch_id is not None else '')



    # ========== PHASE 2: EXTRACT DAILY TIME SERIES (for successful pixels) ==========

    print(f"\n🔄 PHASE 2: Extracting daily time series for {len(all_results)} successful pixels...")



    # Prepare arguments for parallel extraction

    extraction_args = [

        (result['pixel'], result['alpha'], calib_years, soil_data, hydro_data, crop_params)

        for result in all_results

    ]



    timeseries_results = []

    extraction_batch_size = 25  # Smaller batch for memory efficiency



    for batch_start in range(0, len(extraction_args), extraction_batch_size):

        batch_end = min(batch_start + extraction_batch_size, len(extraction_args))

        batch_args = extraction_args[batch_start:batch_end]



        print(f"   Extracting batch {batch_start//extraction_batch_size + 1}/"

              f"{(len(extraction_args)-1)//extraction_batch_size + 1}...")



        try:

            with Pool(processes=max_workers) as pool:

                batch_timeseries = pool.map(extract_timeseries_for_pixel, batch_args)

            timeseries_results.extend(batch_timeseries)

        except Exception as e:

            print(f"   ⚠️ Extraction error: {e}")

            # Add empty results for failed extractions

            for _ in batch_args:

                timeseries_results.append({'success': False, 'error_msg': str(e)})



    successful_extractions = sum(1 for r in timeseries_results if r['success'])

    print(f"   ✅ Successfully extracted: {successful_extractions}/{len(timeseries_results)}")



    # ========== SAVE RESULTS ==========

    print(f"\n💾 Saving results...")



    # Batch mode: save batch-specific npz for merge script; also partial maps

    if batch_id is not None:

        batch_npz = os.path.join(ckpt_dir, f'calibration_batch_{irrigation}_{crop}_{calib_years}y_b{batch_id}_{timestamp}.npz')

        transform_tuple = tuple(soil_data['transform'])[:6]

        crs_wkt = soil_data['crs'].to_wkt() if hasattr(soil_data['crs'], 'to_wkt') else str(soil_data['crs'])

        np.savez_compressed(batch_npz, all_results=all_results, timeseries_results=timeseries_results,

                            failed_pixels=failed_pixels, batch_id=batch_id,

                            map_shape=np.array(soil_data['pH'].shape), transform=np.array(transform_tuple),

                            crs_wkt=crs_wkt, irrigation=irrigation, crop=crop,

                            calib_years=calib_years, timestamp=timestamp)

        print(f"✅ Batch output saved: {batch_npz}")

        save_alpha_map(all_results, soil_data, os.path.join(ckpt_dir, f'alpha_map_{irrigation}_{crop}_{calib_years}y_b{batch_id}_{timestamp}.tif'))

        save_static_result_maps(all_results, soil_data, irrigation, crop, timestamp, output_dir=ckpt_dir)

        return os.path.join(ckpt_dir, f'alpha_map_{irrigation}_{crop}_{calib_years}y_b{batch_id}_{timestamp}.tif')



    # Full run: save alpha map (GeoTIFF)

    alpha_map_filename = os.path.join(ckpt_dir, f'alpha_map_{irrigation.lower()}_{crop}_{calib_years}y_{timestamp}.tif')

    save_alpha_map(all_results, soil_data, alpha_map_filename)



    # Save static maps - pH, error (GeoTIFF)

    save_static_result_maps(all_results, soil_data, irrigation, crop, timestamp, output_dir=ckpt_dir)



    # Save daily time series (compressed NumPy)

    timeseries_filename = save_daily_timeseries(

        timeseries_results, soil_data, irrigation, crop, calib_years, timestamp, output_dir=ckpt_dir

    )



    # Save summary

    summary_filename = os.path.join(ckpt_dir, f'calibration_summary_{irrigation.lower()}_{crop}_{timestamp}.txt')

    with open(summary_filename, 'w') as f:

        f.write(f"ALPHA CALIBRATION SUMMARY\n")

        f.write(f"=========================\n")

        f.write(f"Irrigation: {irrigation}\n")

        f.write(f"Crop: {crop}\n")

        f.write(f"Crop params: K_max={crop_params['K_max']}, RAI={crop_params['RAI']}, root_d={crop_params['root_d']}\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        f.write(f"Calibration years: {calib_years}\n")

        f.write(f"\nResults:\n")

        f.write(f"Total pixels processed: {len(valid_pixels)}\n")

        f.write(f"Successful calibrations: {len(all_results)}\n")

        f.write(f"Failed calibrations: {len(failed_pixels)}\n")

        f.write(f"Success rate: {len(all_results)/len(valid_pixels)*100:.1f}%\n")

        f.write(f"\nDaily time series extraction:\n")

        f.write(f"Successful extractions: {successful_extractions}\n")

        f.write(f"Time series file: {timeseries_filename}\n")

        f.write(f"\nAlpha statistics:\n")



        if all_results:

            alphas = [r['alpha'] for r in all_results]

            errors = [r['error'] for r in all_results]

            num_tests = [r.get('num_tests', 0) for r in all_results]

            times = [r.get('time_elapsed', 0) for r in all_results]



            f.write(f"  Mean alpha: {np.mean(alphas):.3f}\n")

            f.write(f"  Std alpha: {np.std(alphas):.3f}\n")

            f.write(f"  Min alpha: {np.min(alphas):.3f}\n")

            f.write(f"  Max alpha: {np.max(alphas):.3f}\n")

            f.write(f"  Mean error: {np.mean(errors):.3f}\n")

            f.write(f"  Max error: {np.max(errors):.3f}\n")

            f.write(f"\nEfficiency statistics:\n")

            f.write(f"  Mean alpha tests per pixel: {np.mean(num_tests):.1f}\n")

            f.write(f"  Mean time per pixel (calibration): {np.mean(times):.1f}s\n")



    print(f"   Summary saved to: {summary_filename}")



    # Save failed pixels

    if failed_pixels:

        failed_filename = os.path.join(ckpt_dir, f'failed_pixels_{irrigation.lower()}_{crop}_{timestamp}.txt')

        with open(failed_filename, 'w') as f:

            f.write("# Failed pixels (row, col)\n")

            for pixel in failed_pixels:

                f.write(f"{pixel[0]},{pixel[1]}\n")

        print(f"   Failed pixels saved to: {failed_filename}")



    print(f"\n✅ Scenario {irrigation}/{crop} COMPLETE!")

    print(f"   Alpha map: {alpha_map_filename}")

    print(f"   Daily time series: {timeseries_filename}")

    print(f"   Success rate: {len(all_results)/len(valid_pixels)*100:.1f}%")



    return alpha_map_filename



def main():

    """Main execution function"""

    parser = argparse.ArgumentParser(description='Full map alpha calibration')

    parser.add_argument('--irrigation', type=str,

                       choices=['drip', 'traditional', 'trad', 'rainfed'],

                       help='Irrigation type (required unless --all is used)')

    parser.add_argument('--crop', type=str,

                       choices=['vite', 'olivo', 'agrumi', 'pesco', 'grano'],

                       help='Crop type (required unless --all is used)')

    parser.add_argument('--years', type=int, default=30,

                       help='Calibration years (default: 30)')

    parser.add_argument('--workers', type=int, default=None,

                       help='Number of parallel workers (default: all CPUs)')

    parser.add_argument('--hydro_dir', type=str, default=None,

                       help='Direct path to folder with shallow_*.mat files (overrides default path construction)')

    parser.add_argument('--hydro_dt', type=str, default='4h', choices=['daily', '4h'],

                       help='Hydrological data timestep: daily (1 step/day) or 4h (6 steps/day). Default: 4h')

    parser.add_argument('--batch-id', type=int, default=None,

                       help='Process only this batch (0-indexed). For SLURM job arrays. Use with SLURM_ARRAY_TASK_ID.')

    parser.add_argument('--batch-start', type=int, default=None,

                       help='Process batches from this index (inclusive). Alternative to --batch-id.')

    parser.add_argument('--batch-end', type=int, default=None,

                       help='Process batches up to this index (exclusive). Use with --batch-start.')

    parser.add_argument('--checkpoint-dir', type=str, default=None,

                       help='Directory for checkpoint files. Default: current directory. Partial results saved after each batch.')

    parser.add_argument('--all', action='store_true',

                       help='Run all 8 irrigation×crop combinations')



    args = parser.parse_args()



    # Validate arguments

    if not args.all and (args.irrigation is None or args.crop is None):

        parser.error("Either --all or both --irrigation and --crop are required")



    # Configuration

    base_path = '/scratch/user/lorenzo32/WATNEEDS+SMEW'

    # Determine number of workers:
    # 1. Use explicit --workers argument if provided
    # 2. Otherwise, check SLURM_CPUS_PER_TASK (respects SLURM allocation)
    # 3. Fall back to all available CPUs
    if args.workers:
        max_workers = args.workers
    else:
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        max_workers = int(slurm_cpus) if slurm_cpus else mp.cpu_count()

    calib_years = args.years



    print("="*80)

    print("FULL MAP ALPHA CALIBRATION")

    print("="*80)

    print(f"Start time: {datetime.now().strftime('%a %b %d %H:%M:%S %Z %Y')}")

    print(f"Max workers: {max_workers}")

    print(f"CPU cores available: {mp.cpu_count()}")

    print(f"Calibration years: {calib_years}")



    # Load soil data once (shared across all scenarios)

    print(f"\n📂 Loading soil data (shared across all scenarios)...")

    start_time = time.time()

    soil_data = load_soil_data(base_path)

    load_time = time.time() - start_time

    print(f"   Soil data loaded in {load_time:.1f} seconds")



    # Define scenarios to run

    print("DEBUG: Defining scenarios...", flush=True)

    if args.all:

        # All 8 combinations

        irrigations = ['drip', 'traditional']

        crops = ['vite', 'olivo', 'agrumi', 'pesco']

        scenarios = [(irrig, crop) for irrig in irrigations for crop in crops]

        print(f"\n🔄 Running ALL 8 scenarios:", flush=True)

        for irrig, crop in scenarios:

            print(f"   - {irrig.capitalize()}/{crop}", flush=True)

    else:

        # Single scenario

        irrig_norm = {'traditional': 'traditional', 'trad': 'traditional',
                      'drip': 'drip', 'rainfed': 'rainfed'}
        irrigation = irrig_norm.get(args.irrigation, args.irrigation)

        crop = args.crop.lower()

        scenarios = [(irrigation, crop)]



    print(f"DEBUG: {len(scenarios)} scenarios defined", flush=True)



    # Track results

    completed_maps = []

    failed_scenarios = []



    total_start_time = time.time()



    print("DEBUG: Starting scenario loop...", flush=True)

    # Run each scenario

    for irrigation, crop in scenarios:

        print(f"DEBUG: Starting scenario {irrigation}/{crop}...", flush=True)

        try:

            alpha_map = run_calibration_for_scenario(

                irrigation, crop, calib_years, max_workers, base_path, soil_data,

                hydro_dir_override=args.hydro_dir, hydro_dt=args.hydro_dt,

                batch_id=args.batch_id, batch_start=args.batch_start, batch_end=args.batch_end,

                checkpoint_dir=args.checkpoint_dir

            )

            if alpha_map:

                completed_maps.append((irrigation, crop, alpha_map))

            else:

                failed_scenarios.append((irrigation, crop, "No results"))

        except Exception as e:

            print(f"\n❌ Error in {irrigation}/{crop}: {e}")

            failed_scenarios.append((irrigation, crop, str(e)))



    # Final summary

    total_time = time.time() - total_start_time



    print("\n" + "="*80)

    print("FINAL SUMMARY")

    print("="*80)

    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    print(f"End time: {datetime.now().strftime('%a %b %d %H:%M:%S %Z %Y')}")



    print(f"\n✅ Completed scenarios ({len(completed_maps)}):")

    for irrigation, crop, alpha_map in completed_maps:

        print(f"   - {irrigation.capitalize()}/{crop}: {alpha_map}")



    if failed_scenarios:

        print(f"\n❌ Failed scenarios ({len(failed_scenarios)}):")

        for irrigation, crop, reason in failed_scenarios:

            print(f"   - {irrigation.capitalize()}/{crop}: {reason}")



    print(f"\n📁 Output files generated per scenario:")

    print(f"   - alpha_map_*.tif (calibrated alpha values)")

    print(f"   - pH_final_map_*.tif (final pH, last year mean)")

    print(f"   - error_map_*.tif (calibration error)")

    print(f"   - daily_timeseries_*.npz (daily Ca, Mg, DIC, v, pH time series)")

    print(f"   - calibration_summary_*.txt (statistics)")

    print(f"\n📊 Daily time series format (.npz):")

    print(f"   - Ca_daily, Mg_daily, DIC_daily, v_daily, pH_daily: (n_pixels, n_days)")

    print(f"   - pixel_coords: (n_pixels, 2) for spatial reconstruction")

    print(f"   - map_shape, transform, crs_wkt: georeferencing info")

    if failed_scenarios:
        sys.exit(1)


if __name__ == '__main__':

    main()

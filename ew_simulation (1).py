"""
EW SIMULATION - WATNEEDS+SMEW
==============================
Runs Enhanced Weathering simulations using calibrated alpha maps.

Strategy:
  - pH_in = pH_target (pixel's observed pH) for model initialization
  - I_background reconstructed from pH_calibration (mean pH) to match calibration
  - CaCO3 = MgCO3 = 0 (clean start, CDR from difference EW - noEW)
  - 30-year spinup (no rock) to reach calibration steady state
  - 30-year EW period, rock at years 30, 40, 50 (relative: 0, 10, 20)
  - Both noEW and EW run full 60y independently; only years 30-60 saved
  - Runs noEW (baseline) + EW scenario(s) per pixel

Output:
  .npy files per variable, shape (rows, cols, 10950) float32
  Saved in Results/{crop}_{irrigation}/

Usage:
  python ew_simulation.py
  python ew_simulation.py --workers 8
"""

import sys
import gc
import numpy as np
import pandas as pd
import pyEW
from pyEW import vegetation
import rasterio
import scipy.io
import os
import multiprocessing as mp
from multiprocessing import Pool
import warnings
import argparse
from datetime import datetime
import time

try:
    import netCDF4
except ImportError:
    pass

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================
# DEFAULTS (overridden by CLI args)
# ============================================================
SPINUP_YEARS = 30                  # spinup to reach calibration steady state
SIM_YEARS    = 30                  # EW simulation period (after spinup)
TOTAL_YEARS  = SPINUP_YEARS + SIM_YEARS  # 60y total
APP_RATE     = 4000.0              # g/m2 (= 40 t/ha)
APP_YEARS    = [0, 10, 20]        # rock application years (relative to EW period start)

BASE_PATH   = '/scratch/user/lorenzo32/WATNEEDS+SMEW'
HYDRO_DT    = '4h'                 # '4h' or 'daily'
HYDRO_YEARS = 30                   # years of hydro data to load

# These are set by parse_args() before anything runs
CROP = IRRIGATION = SCENARIOS = None
SOIL_DIR = ALPHA_MAP = HYDRO_DIR = TEMP_FILE = OUTPUT_DIR = None

# Variables to save as .npy (each shape 20 x 40 x n_days)
VARS_TO_SAVE = ['pH', 'Ca', 'Mg', 'Na', 'K', 'DIC', 'HCO3', 'CO3',
                'Alk', 'CaCO3', 'MgCO3']

# ============================================================
# CROP PARAMETERS
# ============================================================
CROP_PARAMS = {
    'vite':   {'category': 'vine',      'K_max': 1450.0, 'v_min_ratio': 0.05, 'RAI': 10, 'root_d': 0.4e-3},  # Funes et al. 2022
    'olivo':  {'category': 'olive',     'K_max': 3530.0, 'v_min_ratio': 0.70, 'RAI': 10, 'root_d': 0.4e-3},  # Funes et al. 2022
    'agrumi': {'category': 'citrus',    'K_max': 5790.0, 'v_min_ratio': 0.80, 'RAI': 10, 'root_d': 0.4e-3},  # Funes et al. 2022
    'pesco':  {'category': 'deciduous', 'K_max': 2960.0, 'v_min_ratio': 0.10, 'RAI': 10, 'root_d': 0.4e-3},  # Funes et al. 2022
    'grano':  {'category': 'wheat',     'K_max': 1000.0, 'v_min_ratio': 0.00, 'RAI': 10, 'root_d': 0.3e-3},
}
CROP_NAME_MAP = {'vine': 'Vite', 'olive': 'Olivo', 'citrus': 'Agrumi', 'deciduous': 'Pesco', 'wheat': 'Grano'}

CROP_AREA_MAP = {
    'vite':   'sicily10km_vineyard_total_ha.tif',
    'olivo':  'sicily10km_olives_total_ha.tif',
    'agrumi': 'sicily10km_citrus_total_ha.tif',
    'pesco':  'sicily10km_fruits_total_ha.tif',
    'grano':  'sicily10km_wheat_total_ha.tif',
}

# ============================================================
# MINERAL / ROCK DEFINITIONS
# ============================================================
# PSD: 11 bins from sieve (>75um, SGS Table 3) + Malvern laser (<75um)
# Source: SGS Report 24/3619 + Malvern Mastersizer, Italy Mine Waste (Mt. Etna)
# Representative diameter = geometric mean of bin edges: d = sqrt(d_lo * d_hi)
D_IN_PSD = np.array([
    7.0711e-07,  # bin 1:  0-1 um (Malvern)
    2.2361e-06,  # bin 2:  1-5 um (Malvern)
    8.6603e-06,  # bin 3:  5-15 um (Malvern)
    2.1213e-05,  # bin 4:  15-30 um (Malvern)
    3.8730e-05,  # bin 5:  30-50 um (Malvern)
    6.1237e-05,  # bin 6:  50-75 um (Malvern)
    1.0607e-04,  # bin 7:  75-150 um (Sieve)
    1.7833e-04,  # bin 8:  150-212 um (Sieve)
    3.5665e-04,  # bin 9:  212-600 um (Sieve) — dominant fraction
    7.1414e-04,  # bin 10: 600-850 um (Sieve)
    9.2195e-04,  # bin 11: 850-1000 um (Sieve)
])

# Mineralogy: XRD Rietveld refinement (SGS Report Table 10, Italy mine waste)
# Reactive phases: 84.8% of rock mass. Non-reactive (muscovite 9.6%, oxides 4.8%): not modeled.
ROCK_DEFS = {
    'basalt': {
        'mineral': ["labradorite", "albite", "diopside", "anorthite"],
        'rock_f':  np.array([0.426, 0.186, 0.184, 0.052]),
        'psd':     np.array([0.0043, 0.0170, 0.0273, 0.0413, 0.0547,
                             0.0441, 0.1428, 0.1033, 0.3345, 0.0910, 0.1397]),
    },
}

# Placeholder for noEW (rock params needed by biogeochem but never applied)
ROCK_NOEW = {
    'mineral': ["labradorite", "albite", "diopside", "anorthite"],
    'rock_f':  np.array([0.426, 0.186, 0.184, 0.052]),
    'psd':     np.array([0.0043, 0.0170, 0.0273, 0.0413, 0.0547,
                         0.0441, 0.1428, 0.1033, 0.3345, 0.0910, 0.1397]),
}

# ============================================================
# DATA LOADERS (matching calibration logic)
# ============================================================
def load_soil_data():
    """Load soil maps from SOIL_DIR. Matches calibration_full_map_multi_robust.py exactly."""
    from scipy.ndimage import distance_transform_edt
    print("Loading soil data...")

    def load_tif(name, nodata_val=None):
        fpath = os.path.join(SOIL_DIR, name)
        with rasterio.open(fpath) as src:
            data = src.read(1).astype(np.float32)
            meta = {'transform': src.transform, 'crs': src.crs}
        if nodata_val is not None:
            data[data == nodata_val] = np.nan
        else:
            data[data < 0] = np.nan
        return data, meta

    def fill_nan_nn(arr, name):
        """Fill NaN pixels with nearest valid neighbor."""
        nan_mask = np.isnan(arr)
        if np.any(nan_mask) and np.any(~nan_mask):
            _, nn_idx = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
            arr[nan_mask] = arr[nn_idx[0][nan_mask], nn_idx[1][nan_mask]]
            print(f"  {name}: filled {int(np.sum(nan_mask))} NaN pixels (nearest-neighbor)")
        return arr

    # pH
    ph_map, geo = load_tif("sicily_ph_cacl2_10km.tif", nodata_val=-9999)
    target_shape = ph_map.shape

    # Bulk density  [dg/cm3] -> [g/m3]
    rho_dg, _ = load_tif("bdod_sicily_masked_10km_pH.tif")
    rho_kgm3 = rho_dg * 100          # dg/cm3 -> kg/m3
    rho_gm3  = rho_kgm3 * 1000       # -> g/m3

    # CEC  [cmol(+)/kg] -> [umol / soil column]
    cec_raw, _ = load_tif("cec_sicily_masked_10km_pH (1).tif")
    cec_tot = cec_raw * 0.01 * rho_kgm3 * 0.3 * 1e6

    # SOC  [dg/kg] -> [gOC/m3]
    soc_raw, _ = load_tif("soc_sicily_masked_10km_pH (1).tif")
    soc_in  = soc_raw * rho_gm3 / 1000

    # ADD
    add_path = os.path.join(SOIL_DIR, "ADD_map_steady_state.tif")
    if os.path.exists(add_path):
        add_map, _ = load_tif("ADD_map_steady_state.tif")
        add_map[np.isnan(add_map)] = 1.2
    else:
        print("  ADD map not found, using default 1.2")
        add_map = np.full(target_shape, 1.2, dtype=np.float32)

    # Anions [umol/L]
    anions, _ = load_tif("Anions_interpolated_umolC_L.tif")

    # Heterotrophic respiration [gC/m2/yr] -> [mol-conv / m2 d]
    r_het_raw, _ = load_tif("r_het_Sic_10km_resampled2.tif")
    r_het_map = r_het_raw * 1e6 / (12.011 * 365.25)
    r_het_map = fill_nan_nn(r_het_map, "r_het")

    # Hydraulic parameter maps (fill NaN with nearest-neighbor)
    K_s_map, _ = load_tif("K_s.tif")
    K_s_map = K_s_map / 100.0        # cm/d -> m/d
    K_s_map = fill_nan_nn(K_s_map, "K_s")
    n_map, _   = load_tif("n.tif")
    n_map = fill_nan_nn(n_map, "n")
    b_map, _   = load_tif("b.tif")
    b_map = fill_nan_nn(b_map, "b")
    s_fc_map, _ = load_tif("s_fc.tif")
    s_fc_map = fill_nan_nn(s_fc_map, "s_fc")
    s_h_map, _  = load_tif("s_h.tif")
    s_h_map = fill_nan_nn(s_h_map, "s_h")
    s_w_map, _  = load_tif("s_w.tif")
    s_w_map = fill_nan_nn(s_w_map, "s_w")

    # Texture classes (for CEC Gaines-Thomas lookup)
    tex_file = os.path.join(SOIL_DIR, "sicily_texture_classes_10km (3).csv")
    tex = pd.read_csv(tex_file, header=None).values
    if tex.shape != target_shape:
        tex = tex[:target_shape[0], :target_shape[1]]

    # 3D soil temperature climatology (365 x rows x cols)
    print(f"  Loading temperature: {TEMP_FILE}")
    try:
        import netCDF4
        with netCDF4.Dataset(TEMP_FILE) as ds:
            soil_temp = ds.variables['soil_temperature'][:]
            if hasattr(ds.variables['soil_temperature'], '_FillValue'):
                fv = ds.variables['soil_temperature']._FillValue
                soil_temp[soil_temp == fv] = np.nan
    except (ImportError, Exception):
        from scipy.io import netcdf
        with netcdf.netcdf_file(TEMP_FILE, 'r', mmap=False) as f:
            soil_temp = f.variables['soil_temperature'][:].copy()
            soil_temp[soil_temp < -900] = np.nan

    if soil_temp.shape[1:] != target_shape:
        soil_temp = np.transpose(soil_temp, (2, 0, 1))

    # Convert masked array to regular float32 (netCDF4 returns masked arrays)
    if hasattr(soil_temp, 'filled'):
        soil_temp = soil_temp.filled(np.nan)
    soil_temp = np.asarray(soil_temp, dtype=np.float32)

    # NN-fill NaN pixels per day (soil_temp may have gaps from old mask)
    from scipy.ndimage import distance_transform_edt as _edt_temp
    n_filled_temp = 0
    for d_idx in range(soil_temp.shape[0]):
        day_slice = soil_temp[d_idx]
        mask_nan = np.isnan(day_slice)
        if mask_nan.any() and (~mask_nan).any():
            _, nn_idx = _edt_temp(mask_nan, return_distances=True, return_indices=True)
            day_slice[mask_nan] = day_slice[nn_idx[0][mask_nan], nn_idx[1][mask_nan]]
            if d_idx == 0:
                n_filled_temp = int(mask_nan.sum())
    if n_filled_temp > 0:
        print(f"  Soil temperature: filled {n_filled_temp} NaN pixels/day (nearest-neighbor)")

    # Validate soil_temp shape: must be (365, rows, cols)
    if soil_temp.shape != (365, target_shape[0], target_shape[1]):
        raise ValueError(
            f"soil_temp shape {soil_temp.shape} does not match expected "
            f"(365, {target_shape[0]}, {target_shape[1]}). Check NetCDF dimension order.")

    print(f"  Loaded: {target_shape[0]}x{target_shape[1]} grid, "
          f"{np.sum(~np.isnan(ph_map))} valid pH pixels")

    return {
        'pH': ph_map, 'CEC': cec_tot, 'SOC': soc_in, 'ADD': add_map,
        'anions': anions, 'r_het': r_het_map, 'soil': tex,
        'K_s': K_s_map, 'n': n_map, 'b': b_map,
        's_fc': s_fc_map, 's_h': s_h_map, 's_w': s_w_map,
        'soil_temp_clim': soil_temp.astype(np.float32),
        'transform': geo['transform'], 'crs': geo['crs'],
    }


def load_hydro_data():
    """Load monthly .mat hydro files from HYDRO_DIR. Matches calibration loader."""
    print(f"Loading hydro data from: {HYDRO_DIR}")

    mat_files = [f for f in os.listdir(HYDRO_DIR)
                 if f.startswith('shallow_s_') and f.endswith('.mat')]
    available_years = sorted({int(f.split('_')[2]) for f in mat_files})
    if not available_years:
        raise FileNotFoundError(f"No shallow_s_*.mat in {HYDRO_DIR}")

    last_years = available_years[-HYDRO_YEARS:]
    print(f"  Available: {available_years[0]}-{available_years[-1]} ({len(available_years)}y)")
    print(f"  Using last {len(last_years)}: {last_years[0]}-{last_years[-1]}")

    steps_per_day = 6 if HYDRO_DT == '4h' else 1

    s_list, L_list, T_list, I_list = [], [], [], []
    total_months = len(last_years) * 12
    loaded = 0
    for year in last_years:
        for month in range(1, 13):
            prefix = os.path.join(HYDRO_DIR, f'shallow_{{}}_{year}_{month}.mat')
            s_list.append(scipy.io.loadmat(prefix.format('s'))['s_shallow'])
            L_list.append(scipy.io.loadmat(prefix.format('L'))['L_shallow'])
            T_list.append(scipy.io.loadmat(prefix.format('T'))['T_shallow'])
            I_file = prefix.format('I')
            if os.path.exists(I_file):
                I_list.append(scipy.io.loadmat(I_file)['I_shallow'])
            else:
                I_list.append(np.zeros_like(s_list[-1]))
            loaded += 1
            if loaded % 60 == 0 or loaded == total_months:
                print(f"  Loaded {loaded}/{total_months} months ({loaded*4} files)...",
                      flush=True)

    s_full = np.concatenate(s_list, axis=2).astype(np.float32)
    L_full = np.concatenate(L_list, axis=2).astype(np.float32)
    T_full = np.concatenate(T_list, axis=2).astype(np.float32)
    I_full = np.concatenate(I_list, axis=2).astype(np.float32)
    del s_list, L_list, T_list, I_list  # free raw .mat data (~3.5 GB)

    # Convert to m/d rates IN-PLACE (avoids float64 intermediate copies)
    # 4h: raw [mm/4h] / 1000 * 6 = m/d
    # daily: raw [mm/d] / 1000 = m/d
    flux_scale = np.float32(6.0 if HYDRO_DT == '4h' else 1.0)
    L_full /= np.float32(1000.0)
    L_full *= flux_scale
    T_full /= np.float32(1000.0)
    T_full *= flux_scale
    I_full /= np.float32(1000.0)
    I_full *= flux_scale

    # Trim or tile to needed timesteps (full 60y = spinup + EW)
    sim_timesteps = TOTAL_YEARS * 365 * steps_per_day
    actual = s_full.shape[2]
    if actual >= sim_timesteps:
        s_out = s_full[:, :, :sim_timesteps].copy()
        L_out = L_full[:, :, :sim_timesteps].copy()
        T_out = T_full[:, :, :sim_timesteps].copy()
        I_out = I_full[:, :, :sim_timesteps].copy()
    else:
        n_tiles = int(np.ceil(sim_timesteps / actual))
        s_out = np.tile(s_full, (1, 1, n_tiles))[:, :, :sim_timesteps]
        L_out = np.tile(L_full, (1, 1, n_tiles))[:, :, :sim_timesteps]
        T_out = np.tile(T_full, (1, 1, n_tiles))[:, :, :sim_timesteps]
        I_out = np.tile(I_full, (1, 1, n_tiles))[:, :, :sim_timesteps]
        print(f"  Tiled {n_tiles}x to reach {sim_timesteps} steps")
    del s_full, L_full, T_full, I_full  # free pre-tile arrays
    gc.collect()

    print(f"  Shape: {s_out.shape}, resolution: {HYDRO_DT}")
    return {
        's': s_out,
        'L': L_out,
        'T': T_out,
        'I': I_out,
        'hydro_dt': HYDRO_DT,
        'steps_per_day': steps_per_day,
    }


# ============================================================
# HELPER FUNCTIONS (from calibration)
# ============================================================
def calculate_balanced_concentrations(pH_in, r_het_0, r_aut_0, s_0, temp_soil_0,
                                      D_0, conv_mol, anions_target=None):
    """Wrapper for pyEW.pH_to_conc (kept for call-site compatibility)."""
    An = anions_target if anions_target is not None else 2000
    return pyEW.pH_to_conc(pH_in, r_het_0, r_aut_0, D_0, 0.3, temp_soil_0, conv_mol, An_0=An)


def calculate_base_I_from_leaching(conc_in, L_flux, T_flux):
    """Background weathering inputs from mean water flux. L, T in m/d."""
    factor = np.mean(L_flux + T_flux) * 1000.0   # mm/d
    return (factor * conc_in[0], factor * conc_in[1],
            factor * conc_in[2], factor * conc_in[3], factor)


def expand_hydro_to_biogeochem(s_hydro, L_hydro, T_hydro, I_hydro, hydro_dt, dt_bio):
    """Expand hydro data to biogeochem timestep. Copy from calibration."""
    steps_per_day_bio = int(1.0 / dt_bio)

    if hydro_dt == 'daily':
        n_days = len(s_hydro)
        t_d = np.arange(n_days)
        t_sub = np.linspace(0, n_days - 1, n_days * steps_per_day_bio)
        s = np.interp(t_sub, t_d, s_hydro).astype(np.float32)
        L = np.repeat(L_hydro, steps_per_day_bio)
        T = np.repeat(T_hydro, steps_per_day_bio)
        I = np.repeat(I_hydro, steps_per_day_bio)
    else:  # 4h
        spd_h = 6
        n_hydro = len(s_hydro)
        n_days = n_hydro // spd_h
        n_bio = n_days * steps_per_day_bio
        n_hydro = n_days * spd_h
        s_hydro = s_hydro[:n_hydro]
        L_hydro = L_hydro[:n_hydro]
        T_hydro = T_hydro[:n_hydro]
        I_hydro = I_hydro[:n_hydro]
        t_h = np.arange(n_hydro, dtype=np.float64) / spd_h
        t_b = np.linspace(0, n_days - dt_bio, n_bio, dtype=np.float64)
        s = np.interp(t_b, t_h, s_hydro).astype(np.float32)
        rep = steps_per_day_bio // spd_h
        L = np.repeat(L_hydro, rep).astype(np.float32)[:n_bio]
        T = np.repeat(T_hydro, rep).astype(np.float32)[:n_bio]
        I = np.repeat(I_hydro, rep).astype(np.float32)[:n_bio]

    return s, L, T, I


# ============================================================
# PIXEL WORKER
# ============================================================
def simulate_pixel(pixel_alpha, soil_data, hydro_data, crop_params):
    """Run noEW + EW scenarios for a single pixel."""
    pixel, alpha = pixel_alpha
    ii, jj = pixel

    try:
        # --- Bounds check ---
        h_rows, h_cols = hydro_data['s'].shape[:2]
        if ii >= h_rows or jj >= h_cols:
            print(f"  Skip pixel ({ii},{jj}): out of hydro grid ({h_rows},{h_cols})", flush=True)
            return None

        # --- Extract soil data (cast to float to avoid object-dtype from NN-filled maps) ---
        pH_target    = float(soil_data['pH'][ii, jj])
        soil         = str(soil_data['soil'][ii, jj])
        CEC_tot      = float(soil_data['CEC'][ii, jj])
        SOC_value    = float(soil_data['SOC'][ii, jj])
        ADD_value    = float(soil_data['ADD'][ii, jj])
        r_het_value  = float(soil_data['r_het'][ii, jj])
        anions_value = float(soil_data['anions'][ii, jj])
        K_s_px = float(soil_data['K_s'][ii, jj])
        n_px   = float(soil_data['n'][ii, jj])
        b_px   = float(soil_data['b'][ii, jj])
        s_fc_px = float(soil_data['s_fc'][ii, jj])
        s_h_px  = float(soil_data['s_h'][ii, jj])
        s_w_px  = float(soil_data['s_w'][ii, jj])

        # --- Simulation parameters ---
        dt = 1 / (24 * 2)     # 30-min timesteps
        Zr = 0.3
        conv_mol = 1e6
        conv_Al  = 1e3
        sim_days = TOTAL_YEARS * 365    # full 60y (spinup + EW)
        ew_days  = SIM_YEARS * 365      # 30y EW period to save
        spinup_days = SPINUP_YEARS * 365
        steps_per_day = int(1.0 / dt)
        # --- Hydro data ---
        spd_h = hydro_data['steps_per_day']
        sim_h_steps = sim_days * spd_h
        s_hydro = hydro_data['s'][ii, jj, :sim_h_steps].astype(np.float32)
        L_hydro = hydro_data['L'][ii, jj, :sim_h_steps].astype(np.float32)
        T_hydro = hydro_data['T'][ii, jj, :sim_h_steps].astype(np.float32)
        I_hydro = hydro_data['I'][ii, jj, :sim_h_steps].astype(np.float32)

        # Expand to biogeochem resolution
        s, L, T, I = expand_hydro_to_biogeochem(
            s_hydro, L_hydro, T_hydro, I_hydro, HYDRO_DT, dt)
        del s_hydro, L_hydro, T_hydro, I_hydro  # no longer needed

        # --- Temperature ---
        temp_365 = soil_data['soil_temp_clim'][:, ii, jj]
        temp_daily = np.tile(temp_365, TOTAL_YEARS)[:sim_days]
        temp_soil = np.repeat(temp_daily, steps_per_day)
        del temp_daily

        # --- Vegetation ---
        K_max = crop_params['K_max']
        k_v   = K_max
        RAI   = crop_params['RAI']
        root_d = crop_params['root_d']
        crop_name = CROP_NAME_MAP[crop_params['category']]
        t_days = np.arange(0, sim_days, dt)
        v_min_ratio = crop_params['v_min_ratio']
        v = vegetation.veg_mature(t_days, crop_name, K_max, v_min_ratio=v_min_ratio)
        del t_days  # no longer needed

        # --- Respiration ---
        # Pass pixel-specific hydraulic params (cluster pyEW may accept soil_params)
        soil_params = {
            's_h': float(s_h_px), 's_w': float(s_w_px), 's_fc': float(s_fc_px),
            'b': float(b_px), 'K_s': float(K_s_px), 'n': float(n_px),
        }
        [SOC_sub, r_het, r_aut, D, k_dec, n_param] = pyEW.carbon_respiration_dynamic(
            SOC_value, r_het_value, ADD_value, 1, soil,
            s, v, k_v, Zr, temp_soil, dt, conv_mol,
            soil_params=soil_params)

        # --- Initial concentrations — use time-averaged values (matches calibration) ---
        r_het_mean = float(np.mean(r_het))
        r_aut_mean = float(np.mean(r_aut))
        D_mean = float(np.mean(D[D > 0])) if np.any(D > 0) else float(D[0])
        s_mean = float(np.mean(s))
        temp_mean = float(np.mean(temp_soil))

        # Clamp D for IC only: wet pixels (drip/trad) have D→0 which makes
        # pH_to_conc produce extreme Ca (>10000 umol/L) → fsolve diverges.
        # Floor = D at s=0.3: D_0 * (1-0.3)^(10/3) * n^(4/3)
        # This only affects the starting point; D is computed dynamically.
        D_0_free = 1.3824  # pyEW.D_0()
        D_floor = D_0_free * (1 - 0.3)**(10/3) * n_px**(4/3)
        D_for_IC = max(D_mean, D_floor)
        if D_for_IC > D_mean:
            print(f"  Pixel ({ii},{jj}): IC D clamped {D_mean:.6f} → {D_for_IC:.6f} "
                  f"(floor at s=0.3)", flush=True)

        conc_in, An_conc = calculate_balanced_concentrations(
            pH_target, r_het_mean, r_aut_mean, s_mean, temp_mean, D_for_IC,
            conv_mol, anions_target=anions_value)

        # --- Background weathering (I_bg = alpha * [BC]_0 * mean(L+T)) ---
        base_I_Ca, base_I_Mg, base_I_K, base_I_Na, factor = \
            calculate_base_I_from_leaching(conc_in, L, T)

        I_background = {
            'I_Ca': alpha * base_I_Ca,
            'I_Mg': alpha * base_I_Mg,
            'I_K':  alpha * base_I_K,
            'I_Na': alpha * base_I_Na,
            'I_Si': 0.0,
            'I_An': alpha * factor * An_conc,
        }

        # --- CEC initial state (same conc_in, same pH_target — consistent with calibration) ---
        f_CEC_in, K_CEC = pyEW.conc_to_f_CEC(conc_in, pH_target, soil, conv_mol, conv_Al)

        # --- Run each scenario ---
        pixel_results = {}

        for scenario in SCENARIOS:
            if scenario == 'noEW':
                rock = ROCK_NOEW
                mass = 0.0
                t_app = [sim_days + 1]   # never applied
            else:
                rock = ROCK_DEFS[scenario]
                mass = APP_RATE
                t_app = [(SPINUP_YEARS + y) * 365 for y in APP_YEARS]

            data = pyEW.biogeochem_balance(
                n_px, s, L, T, I, v, k_v, RAI, root_d, Zr,
                r_het, r_aut, D, temp_soil,
                pH_target, conc_in, f_CEC_in, K_CEC, CEC_tot,
                0.0,    # Si_in
                0.0,    # CaCO3_in
                0.0,    # MgCO3_in
                mass,   # M_rock_in
                t_app,
                rock['mineral'], rock['rock_f'], D_IN_PSD, rock['psd'],
                0,      # SSA_in (use fractal model from PSD, not measured)
                1,      # diss_f
                dt, conv_Al, conv_mol,
                1,      # keyword_add
                I_background=I_background,
                use_mean_s_init=True,
                monitor_progress=False,
                An_in=An_conc,
                reset_carbonates_at_step=spinup_days * steps_per_day,
            )

            if data is None:
                return None

            # Convert to daily averages, then slice spinup off (keep only EW period)
            extracted = {}
            for var in VARS_TO_SAVE:
                if var in data and data[var] is not None:
                    arr = data[var]
                    try:
                        n_full_days = len(arr) // steps_per_day
                        daily = np.mean(arr[:n_full_days * steps_per_day].reshape(
                            n_full_days, steps_per_day), axis=1).astype(np.float32)
                        # Discard spinup, keep only EW period
                        extracted[var] = daily[spinup_days:spinup_days + ew_days]
                    except Exception:
                        pass

            # M_rock for EW scenarios
            if scenario != 'noEW' and 'M_rock' in data and data['M_rock'] is not None:
                arr = data['M_rock']
                n_full_days = len(arr) // steps_per_day
                daily = np.mean(arr[:n_full_days * steps_per_day].reshape(
                    n_full_days, steps_per_day), axis=1).astype(np.float32)
                extracted['M_rock'] = daily[spinup_days:spinup_days + ew_days]

            # Per-mineral weathering flux EW[j,:] and remaining mass M_min[j,:]
            if scenario != 'noEW':
                mineral_names = rock['mineral']
                for j, mname in enumerate(mineral_names):
                    # EW flux per mineral [mol-conv / m2 d] → daily
                    if 'EW' in data and data['EW'] is not None and data['EW'].ndim == 2:
                        ew_j = data['EW'][j]
                        n_full_days_j = len(ew_j) // steps_per_day
                        daily_ew = np.mean(ew_j[:n_full_days_j * steps_per_day].reshape(
                            n_full_days_j, steps_per_day), axis=1).astype(np.float32)
                        extracted[f'EW_{mname}'] = daily_ew[spinup_days:spinup_days + ew_days]
                    # Remaining mineral mass [g/m2]
                    if 'M_min' in data and data['M_min'] is not None and data['M_min'].ndim == 2:
                        mmin_j = data['M_min'][j]
                        n_full_days_j = len(mmin_j) // steps_per_day
                        daily_mm = np.mean(mmin_j[:n_full_days_j * steps_per_day].reshape(
                            n_full_days_j, steps_per_day), axis=1).astype(np.float32)
                        extracted[f'M_{mname}'] = daily_mm[spinup_days:spinup_days + ew_days]

            pixel_results[scenario] = extracted
            del data   # free the huge locals dict from biogeochem_balance

        # Free all large arrays before returning (helps pymalloc release to OS)
        del s, L, T, I, v, temp_soil, r_het, r_aut, D
        gc.collect()

        return {'pixel': pixel, 'data': pixel_results}

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n  EW ERROR pixel ({ii},{jj}): {type(e).__name__}: {e}\n{tb}", flush=True)
        return None


# ============================================================
# WORKER POOL (module-level globals avoid pickling GBs of data)
# ============================================================
_SHARED_SOIL = None
_SHARED_HYDRO = None
_SHARED_CROP = None

def _simulate_pixel_worker(pixel_alpha):
    """Worker wrapper — reads shared data from module globals (fork inherits)."""
    return simulate_pixel(pixel_alpha, _SHARED_SOIL, _SHARED_HYDRO, _SHARED_CROP)


# ============================================================
# MAIN
# ============================================================
def save_results_npy(results, output_dir, grid_shape, scenarios, sim_days):
    """Save results list to .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    rows, cols = grid_shape

    for scenario in scenarios:
        for var in VARS_TO_SAVE:
            arr3d = np.full((rows, cols, sim_days), np.nan, dtype=np.float32)
            for res in results:
                ii, jj = res['pixel']
                if var in res['data'].get(scenario, {}):
                    d = res['data'][scenario][var]
                    arr3d[ii, jj, :len(d)] = d
            fname = f"{var}_sic_{scenario}_daily.npy"
            np.save(os.path.join(output_dir, fname), arr3d)
            print(f"  Saved: {fname}  ({arr3d.nbytes / 1e6:.1f} MB)")

        if scenario != 'noEW':
            # M_rock total
            arr3d = np.full((rows, cols, sim_days), np.nan, dtype=np.float32)
            for res in results:
                ii, jj = res['pixel']
                if 'M_rock' in res['data'].get(scenario, {}):
                    d = res['data'][scenario]['M_rock']
                    arr3d[ii, jj, :len(d)] = d
            fname = f"M_rock_{scenario}_daily.npy"
            np.save(os.path.join(output_dir, fname), arr3d)
            print(f"  Saved: {fname}")

            # Per-mineral EW flux and remaining mass
            rock = ROCK_DEFS.get(scenario, ROCK_NOEW)
            for mname in rock['mineral']:
                for prefix in ['EW', 'M']:
                    key = f'{prefix}_{mname}'
                    arr3d = np.full((rows, cols, sim_days), np.nan, dtype=np.float32)
                    has_data = False
                    for res in results:
                        ii, jj = res['pixel']
                        if key in res['data'].get(scenario, {}):
                            d = res['data'][scenario][key]
                            arr3d[ii, jj, :len(d)] = d
                            has_data = True
                    if has_data:
                        fname = f"{key}_{scenario}_daily.npy"
                        np.save(os.path.join(output_dir, fname), arr3d)
                        print(f"  Saved: {fname}")


def parse_args():
    """Parse CLI args and set global config."""
    global CROP, IRRIGATION, SCENARIOS, SOIL_DIR, ALPHA_MAP, HYDRO_DIR, TEMP_FILE, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description='EW Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ew_simulation.py --crop vite --irr drip
  python ew_simulation.py --crop olivo --irr traditional --workers 48
        """)
    parser.add_argument('--crop', type=str, required=True,
                        choices=['vite', 'olivo', 'agrumi', 'pesco', 'grano'])
    parser.add_argument('--irr', type=str, required=True,
                        choices=['drip', 'traditional', 'rainfed'])
    parser.add_argument('--scenarios', nargs='+', default=['noEW', 'basalt'],
                        choices=['noEW', 'basalt'])
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--rock-dose', type=float, default=4000.0,
                        help='Rock dose g/m2 per application (default: 4000 = 40 t/ha)')
    parser.add_argument('--app-years', nargs='+', type=int, default=[0, 10, 20],
                        help='Rock application years relative to EW period')
    parser.add_argument('--base-dir', type=str,
                        default='/scratch/user/lorenzo32/WATNEEDS+SMEW')

    args = parser.parse_args()

    CROP = args.crop
    IRRIGATION = args.irr
    SCENARIOS = args.scenarios

    SOIL_DIR  = os.path.join(args.base_dir, 'soil_param')
    # Use weighted alpha map (single alpha per pixel, area-weighted across all scenarios)
    ALPHA_MAP = os.path.join(args.base_dir, 'weighted_alpha_map.tif')
    # Map irrigation name to hydro directory name (traditional -> surface)
    irr_dir = 'surface' if IRRIGATION in ('traditional', 'trad') else IRRIGATION
    HYDRO_DIR = os.path.join(args.base_dir, 'WB_interpolated_first4hours',
                             f'{CROP}_{irr_dir}')
    TEMP_FILE = os.path.join(args.base_dir, 'Sicily_Soil_Temp_3D (1).nc')
    OUTPUT_DIR = os.path.join(args.base_dir, 'Results', f'{CROP}_{IRRIGATION}')

    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    args.workers = args.workers or (int(slurm_cpus) if slurm_cpus else mp.cpu_count())

    return args


def main():
    args = parse_args()
    workers = args.workers

    print("=" * 70)
    print(f"EW SIMULATION: {CROP} / {IRRIGATION}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Spinup: {SPINUP_YEARS}y, EW period: {SIM_YEARS}y, Total: {TOTAL_YEARS}y")
    ew_app_years = [SPINUP_YEARS + y for y in args.app_years]
    print(f"Rock dose: {args.rock_dose} g/m2 ({args.rock_dose/100:.0f} t/ha)")
    print(f"Applications at EW years {args.app_years} (absolute years {ew_app_years})")
    print(f"Total rock over {SIM_YEARS}y: {args.rock_dose * len(args.app_years):.0f} g/m2 "
          f"({args.rock_dose * len(args.app_years) / 100:.0f} t/ha)")
    print(f"Hydro timestep: {HYDRO_DT}")
    print(f"Workers: {workers}")
    print(f"Alpha map: {ALPHA_MAP}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70, flush=True)

    # Update APP_RATE from args
    global APP_RATE, APP_YEARS
    APP_RATE = args.rock_dose
    APP_YEARS = args.app_years

    # 1. Load data
    soil_data  = load_soil_data()
    print(f"Soil data loaded. Grid: {soil_data['pH'].shape}", flush=True)
    hydro_data = load_hydro_data()
    print(f"Hydro data loaded. Grid: {hydro_data['s'].shape[:2]}", flush=True)

    # Validate grid shapes match
    soil_shape = soil_data['pH'].shape
    hydro_shape = hydro_data['s'].shape[:2]
    if soil_shape != hydro_shape:
        print(f"WARNING: soil grid {soil_shape} != hydro grid {hydro_shape}")
        # Use the smaller common window
        min_rows = min(soil_shape[0], hydro_shape[0])
        min_cols = min(soil_shape[1], hydro_shape[1])
        print(f"  Using common window: ({min_rows}, {min_cols})")

    with rasterio.open(ALPHA_MAP) as src:
        alpha_grid = src.read(1).astype(np.float32)
    if alpha_grid.shape != soil_data['pH'].shape:
        alpha_grid = alpha_grid[:soil_data['pH'].shape[0], :soil_data['pH'].shape[1]]

    # Valid pixel masks
    s_mean = np.nanmean(hydro_data['s'][:, :, :365 * hydro_data['steps_per_day']], axis=2)
    s_first = hydro_data['s'][:, :, 0]
    s_min = np.nanmin(hydro_data['s'][:, :, :365 * hydro_data['steps_per_day']], axis=2)
    hydro_ok = (s_mean > 0.01) & (s_first > 0.001) & (s_min > 0) & ~np.isnan(s_mean)
    soil_ok = ~(np.isnan(soil_data['pH']) | np.isnan(soil_data['CEC']) |
                np.isnan(soil_data['SOC']) | (soil_data['soil'] == 'unknown'))

    # 2. Crop area mask — only run pixels where this crop is cultivated
    crop_area_file = CROP_AREA_MAP.get(CROP)
    if crop_area_file:
        crop_area_path = os.path.join(args.base_dir, 'aree_colt', crop_area_file)
        if os.path.exists(crop_area_path):
            with rasterio.open(crop_area_path) as src:
                crop_area = src.read(1).astype(np.float64)
            crop_area[~np.isfinite(crop_area)] = 0.0
            crop_mask = crop_area > 0
            print(f"  Crop area mask ({crop_area_file}): {int(crop_mask.sum())} cultivated pixels")
        else:
            crop_mask = np.ones_like(soil_data['pH'], dtype=bool)
            print(f"  WARNING: {crop_area_path} not found, no crop mask")
    else:
        crop_mask = np.ones_like(soil_data['pH'], dtype=bool)

    alpha_valid = (alpha_grid > 0) & ~np.isnan(alpha_grid)
    n_alpha = int(alpha_valid.sum())
    n_alpha_crop = int((alpha_valid & crop_mask).sum())
    print(f"  Alpha valid: {n_alpha} total, {n_alpha_crop} within crop mask")

    # WATNEEDS reference mask (common 426-pixel domain for all scenarios)
    ref_mask_path = os.path.join(args.base_dir, 'aree_colt', 'watneeds_ref_mask.tif')
    if os.path.exists(ref_mask_path):
        with rasterio.open(ref_mask_path) as src:
            ref_mask = src.read(1).astype(bool)
        if ref_mask.shape != soil_data['pH'].shape:
            ref_mask = np.ones_like(soil_data['pH'], dtype=bool)
            print("  WARNING: ref mask shape mismatch, skipping")
        else:
            print(f"  WATNEEDS ref mask: {int(ref_mask.sum())} pixels")
    else:
        ref_mask = np.ones_like(soil_data['pH'], dtype=bool)
        print("  WARNING: watneeds_ref_mask.tif not found, skipping")

    # Build valid pixel list
    hydraulic_ok = (~np.isnan(soil_data['K_s']) & ~np.isnan(soil_data['n']) &
                    ~np.isnan(soil_data['b']) & ~np.isnan(soil_data['s_fc']) &
                    ~np.isnan(soil_data['s_h']) & ~np.isnan(soil_data['s_w']))
    alpha_ok = (alpha_grid > 0) & ~np.isnan(alpha_grid)
    final_mask = alpha_ok & hydro_ok & soil_ok & hydraulic_ok & ref_mask & crop_mask
    valid_pixels = list(zip(*np.where(final_mask)))
    print(f"Valid pixels: {len(valid_pixels)} (crop={int(crop_mask.sum())}, "
          f"alpha={int(alpha_ok.sum())}, ref_mask={int(ref_mask.sum())})")

    if not valid_pixels:
        print("No valid pixels! Check alpha map and input data.")
        sys.exit(1)

    # 3. Prepare tasks
    tasks = [(px, float(alpha_grid[px])) for px in valid_pixels]

    # 4. Run with process pool
    #    Use module-level globals so forked workers inherit shared data via COW
    #    instead of pickling GBs through partial() — fixes OOM
    global _SHARED_SOIL, _SHARED_HYDRO, _SHARED_CROP
    _SHARED_SOIL = soil_data
    _SHARED_HYDRO = hydro_data
    _SHARED_CROP = CROP_PARAMS[CROP]

    # Force memory release before forking workers (minimizes COW arena copies)
    gc.collect()

    results = []
    batch_size = max(workers * 2, 50)
    t0 = time.time()
    grid_shape = hydro_data['s'].shape[:2]
    sim_days = SIM_YEARS * 365

    with Pool(workers, maxtasksperchild=1) as pool:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(tasks) - 1) // batch_size + 1

            batch_res = pool.map(_simulate_pixel_worker, batch)

            ok = [r for r in batch_res if r is not None]
            results.extend(ok)
            elapsed = time.time() - t0
            pct = 100 * len(results) / len(tasks)
            eta = elapsed / max(len(results), 1) * (len(tasks) - len(results))
            print(f"  Batch {batch_num}/{total_batches}: {len(ok)}/{len(batch)} ok, "
                  f"total {len(results)}/{len(tasks)} ({pct:.0f}%), "
                  f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s", flush=True)

    print(f"\nDone: {len(results)} pixels in {time.time() - t0:.0f}s")

    # 5. Save results
    save_results_npy(results, OUTPUT_DIR, grid_shape, SCENARIOS, sim_days)

    print(f"\nAll output in: {OUTPUT_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()

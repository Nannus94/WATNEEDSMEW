import numpy as np
import os
import rasterio
import h5py
import scipy.io
from scipy.interpolate import NearestNDInterpolator

# --- 1. CONFIGURAZIONE GLOBALE ---
base_dir = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW"
soil_maps_dir = os.path.join(base_dir, "soil_param (1)")
# Reference mask: mainland Sicily (no small islands), 39x43 @ 10km
ref_mask_path = os.path.join(base_dir, r"Aree_coltivate\sicily_mainland_mask.tif")
# Soil parameter maps (39x43, NN-filled in calibration; here we fill on-the-fly)
n_file_path = os.path.join(soil_maps_dir, "n.tif")          # porosity [-]
s_fc_file_path = os.path.join(soil_maps_dir, "s_fc.tif")    # field capacity (relative sat) [-]
s_w_file_path = os.path.join(soil_maps_dir, "s_w.tif")      # wilting point (relative sat) [-]
ks_file_path = os.path.join(soil_maps_dir, "K_s.tif")       # K_sat [cm/d]

# Output Root — separate directory for MERIDA runs
main_output_dir = os.path.join(base_dir, "SMEW_Output_4Hour_MERIDA")
if not os.path.exists(main_output_dir):
    os.makedirs(main_output_dir)

# Parametri — MERIDA data covers 1992-2023
years = range(1992, 2024)
months = range(1, 13)
Z_layer = 0.3        

# --- MODIFICA CHIAVE: DT 4 ORE ---
DT = 4.0/24.0        # 4 ore
N_SUBSTEPS = int(1/DT) # 6 step al giorno

CROP_DB = {
    1: {'name': 'vite',   'beta': 0.964, 'p': 0.45},
    2: {'name': 'olivo',  'beta': 0.976, 'p': 0.65},
    3: {'name': 'pesco',  'beta': 0.966, 'p': 0.50},
    4: {'name': 'agrumi', 'beta': 0.976, 'p': 0.50},
    5: {'name': 'grano',  'beta': 0.961, 'p': 0.55}
}

ZR_RULES = {
    1: {'irr': 1.50, 'rain': 1.80},
    2: {'irr': 1.45, 'rain': 1.80},
    3: {'irr': 1.50, 'rain': 1.80},
    4: {'irr': 1.50, 'rain': 1.80},
    5: {'irr': 1.50, 'rain': 1.50}
}

SCENARIOS = {
    'surface': {'path': os.path.join(base_dir, r"SURFACE\giornalieri"), 'rainfed': False, 'exclude': [5], 'crop_offset': 0},
    'drip':    {'path': os.path.join(base_dir, r"DRIP15-RAINFED610\giornalieri"), 'rainfed': False, 'exclude': [5], 'crop_offset': 0},
    'rainfed': {'path': os.path.join(base_dir, r"DRIP15-RAINFED610\giornalieri"), 'rainfed': True, 'exclude': [], 'crop_offset': 5},
}

RUN_CROPS = [1,2,3,4,5]

# --- UTILS ---
def nn_fill(arr, name=None):
    """Fill NaN pixels with nearest-neighbor interpolation."""
    from scipy.ndimage import distance_transform_edt
    mask = np.isnan(arr)
    n_nan = int(np.sum(mask))
    if n_nan > 0 and np.any(~mask):
        _, nn_idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
        arr[mask] = arr[nn_idx[0][mask], nn_idx[1][mask]]
        if name:
            print(f"   {name}: filled {n_nan} NaN pixels (nearest-neighbor)")
    return arr

def fill_gaps_nearest(data_slice, valid_mask):
    data_exists_mask = ~np.isnan(data_slice)
    if not np.any(data_exists_mask): return data_slice
    missing_mask = valid_mask & (~data_exists_mask)
    if np.any(missing_mask):
        coords_y, coords_x = np.where(data_exists_mask)
        values = data_slice[data_exists_mask]
        interp = NearestNDInterpolator(list(zip(coords_y, coords_x)), values)
        missing_y, missing_x = np.where(missing_mask)
        filled_values = interp(list(zip(missing_y, missing_x)))
        data_slice[missing_mask] = filled_values
    return data_slice

def load_soil_tif(path, name):
    """Load a 39x43 soil parameter GeoTIFF and NN-fill NaNs."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    data[data < 0] = np.nan
    data[np.isinf(data)] = np.nan
    return nn_fill(data, name)

def load_mat_data(filepath, var_name):
    try: return scipy.io.loadmat(filepath)[var_name]
    except: 
        with h5py.File(filepath, 'r') as f: return np.array(f.get(var_name)).T

def crop_center_dynamic(mat_array, target_h, target_w):
    if not isinstance(mat_array, np.ndarray) or mat_array.ndim < 2: return None
    h, w = mat_array.shape[:2]
    sr = (h - target_h) // 2
    sc = (w - target_w) // 2
    if mat_array.ndim == 3: return mat_array[sr : sr + target_h, sc : sc + target_w, :]
    return mat_array[sr : sr + target_h, sc : sc + target_w]

# --- MAIN ---
print("Caricamento Reference & Soil Maps...")

# Load mainland mask as reference grid
with rasterio.open(ref_mask_path) as src:
    ref_profile = src.profile.copy()
    master_land_mask = src.read(1).astype(bool)
    tif_h, tif_w = master_land_mask.shape

# Refine land mask: exclude pixels with no rain data in WATNEEDS (29 coastal pixels
# that are land in the GIS mask but have no hydrological data and no crop area).
# Rain domain is identical across all scenarios/crops, so one reference file suffices.
_ref_bp_path = os.path.join(
    base_dir, r"DRIP15-RAINFED610\giornalieri\outputBP_2005_7_1.mat")
if os.path.exists(_ref_bp_path):
    _bp = load_mat_data(_ref_bp_path, 'outputBP')
    if _bp is not None:
        _bp_trim = _bp[
            ((_bp.shape[0] - tif_h) // 2) : ((_bp.shape[0] - tif_h) // 2) + tif_h,
            ((_bp.shape[1] - tif_w) // 2) : ((_bp.shape[1] - tif_w) // 2) + tif_w,
            :]
        _rain_valid = np.any(np.isfinite(_bp_trim), axis=2)
        _excluded = np.sum(master_land_mask & ~_rain_valid)
        master_land_mask = master_land_mask & _rain_valid
        print(f"   Rain-valid mask: excluded {_excluded} coastal pixels with no hydro data")

print(f"   Reference grid: {tif_h}x{tif_w}, mainland pixels: {np.sum(master_land_mask)}")

# Load 39x43 soil parameter maps (already aligned, NN-filled)
n_map = load_soil_tif(n_file_path, 'n (porosity)')        # porosity [-]
s_fc_map = load_soil_tif(s_fc_file_path, 's_fc')          # relative saturation at FC [-]
s_w_map = load_soil_tif(s_w_file_path, 's_w')             # relative saturation at WP [-]
ks_map = load_soil_tif(ks_file_path, 'K_s')               # K_sat [cm/d]

def apply_strict_mask(data, mask):
    d = data.copy()
    d[~mask] = np.nan
    return d

# Convert relative saturation to volumetric water content: theta = s * n
ths_map = apply_strict_mask(n_map, master_land_mask)                    # porosity = theta_sat
raw_fc = apply_strict_mask(s_fc_map * n_map, master_land_mask)          # theta_fc = s_fc * n
raw_wp = apply_strict_mask(s_w_map * n_map, master_land_mask)           # theta_wp = s_w * n

# Physical parameters [mm]
fc_mm_map = raw_fc * Z_layer * 1000.0
wp_mm_map = raw_wp * Z_layer * 1000.0
k_sat_map = apply_strict_mask(ks_map * 10.0, master_land_mask)         # cm/d -> mm/d
capacity_mm_map = apply_strict_mask(ths_map * Z_layer * 1000.0, master_land_mask)

# Power Law (Brooks-Corey b from psi_fc/psi_wp ratio)
psi_fc, psi_wp = 330.0, 15000.0
num_b = np.log(psi_wp / psi_fc)
safe_fc = np.maximum(fc_mm_map, 0.01)
safe_wp = np.maximum(wp_mm_map, 0.001)
ratio_vol = safe_fc / safe_wp
ratio_vol[ratio_vol <= 1.0] = 1.001
b_map = num_b / np.log(ratio_vol)
b_map = np.clip(b_map, 1, 30)
b_map[~master_land_mask] = np.nan
power_exp_map = 2.0 * b_map + 3.0

print("-" * 60)

# --- LOOP SCENARI ---
for sc_name, sc_data in SCENARIOS.items():
    print(f"\n>> SCENARIO: {sc_name.upper()}")
    mat_files_dir = sc_data['path']
    is_rainfed = sc_data['rainfed']
    
    crop_offset = sc_data.get('crop_offset', 0)

    for crop_code, crop_info in CROP_DB.items():
        if RUN_CROPS is not None and crop_code not in RUN_CROPS: continue
        crop_name = crop_info['name']
        if crop_code in sc_data['exclude']: continue

        # File crop code: offset for rainfed in DRIP15-RAINFED610 (crops 6-10)
        file_crop_code = crop_code + crop_offset

        folder_name = f"{crop_name}_{sc_name}"
        current_output_dir = os.path.join(main_output_dir, folder_name)
        if not os.path.exists(current_output_dir): os.makedirs(current_output_dir)

        print(f"   >> Processing: {crop_name.upper()} (dt=4h, Saving 6 steps/day)")

        # Check files
        test_file = os.path.join(mat_files_dir, f"outputPR_{years[0]}_{1}_{file_crop_code}.mat")
        if not os.path.exists(mat_files_dir) or not os.path.exists(test_file):
            print(f"   !! SKIP: Input mancanti in {mat_files_dir}")
            continue

        beta = crop_info['beta']
        p_depl = crop_info['p']
        k_root = 1.0 - (beta ** (Z_layer * 100.0))
        zr_deep = ZR_RULES[crop_code]['rain'] if is_rainfed else ZR_RULES[crop_code]['irr']
        
        # Triggers
        taw_mm_map = fc_mm_map - wp_mm_map
        trigger_point_map = fc_mm_map - (p_depl * taw_mm_map)
        fc_deep = raw_fc * zr_deep * 1000.0
        wp_deep = raw_wp * zr_deep * 1000.0
        trig_deep = fc_deep - (p_depl * (fc_deep - wp_deep))
        
        current_storage = fc_mm_map.copy()
        
        for yr in years:
            for m in months:
                f_pr = os.path.join(mat_files_dir, f"outputPR_{yr}_{m}_{file_crop_code}.mat")
                f_bw = os.path.join(mat_files_dir, f"outputBW_{yr}_{m}_{file_crop_code}.mat")
                f_et = os.path.join(mat_files_dir, f"outputET_{yr}_{m}_{file_crop_code}.mat")
                f_tb = os.path.join(mat_files_dir, f"outputTB_{yr}_{m}_{file_crop_code}.mat")
                f_s = os.path.join(mat_files_dir, f"outputS_{yr}_{m}_{file_crop_code}.mat")
                
                if not os.path.exists(f_pr): continue
                
                try:
                    # Caricamento Dati
                    ET_full = load_mat_data(f_et, 'outputET')
                    P_grid_raw = crop_center_dynamic(load_mat_data(f_pr, 'outputPR'), tif_h, tif_w)
                    ET_grid_raw = crop_center_dynamic(ET_full, tif_h, tif_w) if ET_full is not None else crop_center_dynamic(load_mat_data(f_et, 'outputET'), tif_h, tif_w)
                    
                    I_grid_raw = np.zeros_like(P_grid_raw)
                    if not is_rainfed and os.path.exists(f_bw):
                        I_grid_raw = crop_center_dynamic(load_mat_data(f_bw, 'outputBW'), tif_h, tif_w)
                    
                    S_full = load_mat_data(f_tb, 'outputTB') if os.path.exists(f_tb) else load_mat_data(f_s, 'outputS')
                    S_grid_raw = crop_center_dynamic(S_full, tif_h, tif_w)

                    if P_grid_raw is None: continue
                    n_days = P_grid_raw.shape[2]

                    # --- PRE-ALLOCAZIONE MATRICI OUTPUT (ALTA RISOLUZIONE) ---
                    # Dimensione Tempo: n_days * N_SUBSTEPS (es. 30 * 6 = 180 steps)
                    total_steps = n_days * N_SUBSTEPS
                    
                    # Usiamo float32 per ridurre la dimensione del file
                    s_out_highres = np.full((tif_h, tif_w, total_steps), np.nan, dtype=np.float32)
                    L_out_highres = np.full((tif_h, tif_w, total_steps), np.nan, dtype=np.float32)
                    T_out_highres = np.full((tif_h, tif_w, total_steps), np.nan, dtype=np.float32)
                    I_out_highres = np.full((tif_h, tif_w, total_steps), np.nan, dtype=np.float32)
                    
                    # Fix Input (Fill Gaps)
                    ET_eff_deep_fixed = np.full_like(ET_grid_raw, np.nan)
                    P_grid_fixed = np.full_like(P_grid_raw, np.nan)
                    I_grid_fixed = np.full_like(I_grid_raw, np.nan)
                    S_deep_grid_fixed = np.full_like(S_grid_raw, np.nan) if S_grid_raw is not None else None

                    for d in range(n_days):
                        et_slice = fill_gaps_nearest(ET_grid_raw[:,:,d], master_land_mask)
                        ET_eff_deep_fixed[:,:,d] = apply_strict_mask(et_slice, master_land_mask)
                        P_grid_fixed[:,:,d] = apply_strict_mask(P_grid_raw[:,:,d], master_land_mask)
                        I_grid_fixed[:,:,d] = apply_strict_mask(I_grid_raw[:,:,d], master_land_mask)
                        if S_deep_grid_fixed is not None:
                             s_slice = fill_gaps_nearest(S_grid_raw[:,:,d], master_land_mask)
                             S_deep_grid_fixed[:,:,d] = apply_strict_mask(s_slice, master_land_mask)

                    # --- SIMULAZIONE ---
                    global_step_idx = 0
                    
                    for d in range(n_days):
                        P_day = P_grid_fixed[:,:,d]
                        I_day = I_grid_fixed[:,:,d]
                        ET_eff_day = ET_eff_deep_fixed[:,:,d]
                        ET_eff_day[~np.isfinite(ET_eff_day)] = 0.0

                        # Inverse Modeling ET (per-pixel)
                        # Only apply inverse modeling on pixels with valid deep soil storage.
                        # Pixels with NaN TB (e.g. 29 coastal pixels in SURFACE) use ET directly.
                        ET_pot_day_recostructed = ET_eff_day.copy()
                        if S_deep_grid_fixed is not None:
                            S_d = S_deep_grid_fixed[:,:,d]
                            valid_S_px = np.isfinite(S_d)
                            if np.any(valid_S_px):
                                ks_deep = np.zeros_like(S_d)
                                mask_stress_d = valid_S_px & (S_d < trig_deep) & (S_d > wp_deep)
                                den_d = np.maximum(trig_deep - wp_deep, 0.1)
                                ks_deep[mask_stress_d] = (S_d[mask_stress_d] - wp_deep[mask_stress_d]) / den_d[mask_stress_d]
                                ks_deep[valid_S_px & (S_d >= trig_deep)] = 1.0
                                safe_ks = np.maximum(ks_deep, 0.05)
                                # Only apply inverse modeling where S_d is valid
                                ET_pot_day_recostructed[valid_S_px] = ET_eff_day[valid_S_px] / safe_ks[valid_S_px]

                        ET_demand_day_shallow = ET_pot_day_recostructed * k_root
                        valid_calc = master_land_mask & ~np.isnan(current_storage) & ~np.isnan(P_day)

                        # Loop Substeps (6 steps da 4h)
                        for step in range(N_SUBSTEPS):
                            # Pioggia: Tutta nel primo step (0) del giorno
                            # Questo simula un evento di 4h
                            if step == 0:
                                In_step = P_day + I_day 
                            else:
                                In_step = np.zeros_like(P_day)
                            
                            current_storage[valid_calc] += In_step[valid_calc]
                            
                            # ET Step
                            ET_demand_step = ET_demand_day_shallow * DT
                            ks_shallow = np.zeros_like(current_storage)
                            mask_no = valid_calc & (current_storage >= trigger_point_map)
                            ks_shallow[mask_no] = 1.0
                            mask_str = valid_calc & (current_storage < trigger_point_map) & (current_storage > wp_mm_map)
                            den_s = np.maximum(trigger_point_map - wp_mm_map, 0.1)
                            if np.any(mask_str):
                                ks_shallow[mask_str] = (current_storage[mask_str] - wp_mm_map[mask_str]) / den_s[mask_str]
                            
                            et_act_step = np.minimum(ET_demand_step * ks_shallow, current_storage)
                            current_storage[valid_calc] -= et_act_step[valid_calc]
                            
                            # Drainage Step
                            drainage_rate = np.zeros_like(current_storage)
                            m_ramp = valid_calc & (current_storage > trigger_point_map) & (current_storage <= fc_mm_map)
                            den_r = np.maximum(fc_mm_map - trigger_point_map, 0.1)
                            if np.any(m_ramp):
                                rel = (current_storage[m_ramp] - trigger_point_map[m_ramp]) / den_r[m_ramp]
                                drainage_rate[m_ramp] = k_sat_map[m_ramp] * (rel ** power_exp_map[m_ramp])
                            
                            m_sat = valid_calc & (current_storage > fc_mm_map)
                            if np.any(m_sat):
                                drainage_rate[m_sat] = k_sat_map[m_sat]
                            
                            L_step = drainage_rate * DT
                            water_above = np.maximum(current_storage - trigger_point_map, 0)
                            real_L = np.minimum(L_step, water_above)
                            
                            current_storage[valid_calc] -= real_L[valid_calc]
                            
                            # Runoff
                            excess = np.maximum(current_storage - capacity_mm_map, 0)
                            current_storage[valid_calc] -= excess[valid_calc]

                            # --- SALVATAGGIO RAW (NO MEDIA!) ---
                            # Calcoliamo Saturazione Relativa Istantanea
                            s_rel_inst = np.full_like(current_storage, np.nan)
                            with np.errstate(divide='ignore', invalid='ignore'):
                                s_rel_inst[valid_calc] = current_storage[valid_calc] / capacity_mm_map[valid_calc]
                            
                            # Scrittura nella matrice gigante
                            s_out_highres[:,:,global_step_idx] = s_rel_inst.astype(np.float32)
                            L_out_highres[:,:,global_step_idx] = real_L.astype(np.float32) # mm persi in queste 4h
                            T_out_highres[:,:,global_step_idx] = et_act_step.astype(np.float32) # mm persi in queste 4h
                            # Save only rain component (not irrigation) — DIC transport uses rain infiltration only
                            Rain_step = P_day if step == 0 else np.zeros_like(P_day)
                            I_out_highres[:,:,global_step_idx] = Rain_step.astype(np.float32)
                            
                            global_step_idx += 1

                    # Salvataggio su disco (I file saranno più pesanti, ma contengono la dinamica)
                    scipy.io.savemat(os.path.join(current_output_dir, f"shallow_s_{yr}_{m}.mat"), {'s_shallow': s_out_highres}, do_compression=True)
                    scipy.io.savemat(os.path.join(current_output_dir, f"shallow_L_{yr}_{m}.mat"), {'L_shallow': L_out_highres}, do_compression=True)
                    scipy.io.savemat(os.path.join(current_output_dir, f"shallow_T_{yr}_{m}.mat"), {'T_shallow': T_out_highres}, do_compression=True)
                    scipy.io.savemat(os.path.join(current_output_dir, f"shallow_I_{yr}_{m}.mat"), {'I_shallow': I_out_highres}, do_compression=True)
                    
                except Exception as e:
                    print(f"XX Errore {folder_name} {yr}-{m}: {e}")
                    import traceback
                    traceback.print_exc()

print("\nOK Completato. Ora hai file .mat con risoluzione 4 ORE.")
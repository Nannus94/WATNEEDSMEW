import numpy as np
import os
import matplotlib.pyplot as plt
import rasterio
import scipy.io
import h5py

# --- CONFIGURAZIONE ---
base_dir = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW"
main_output_dir = os.path.join(base_dir, "SMEW_Output_4Hour_MERIDA")
soil_maps_dir = os.path.join(base_dir, "soil_param (1)")
DIAGNOSIS_OUTPUT_DIR = os.path.join(base_dir, "_DIAGNOSIS_PLOTS_FINAL")

if not os.path.exists(DIAGNOSIS_OUTPUT_DIR):
    os.makedirs(DIAGNOSIS_OUTPUT_DIR)

SCENARIO_PATHS = {
    'drip':    os.path.join(base_dir, r"DRIP15-RAINFED610\giornalieri"),
    'rainfed': os.path.join(base_dir, r"DRIP15-RAINFED610\giornalieri"),
    'surface': os.path.join(base_dir, r"SURFACE\giornalieri"),
}

YEAR_START = 2020
YEAR_END = 2023  # MERIDA data: 1992-2023, rainfed: 1983-2023
PIXEL_ROW = 12
PIXEL_COL = 22
Z_layer = 0.3

# Maps Paths (aligned with New_WB_MERIDA.py — 39x43 mainland mask + soil params)
ref_mask_path = os.path.join(base_dir, r"Aree_coltivate\sicily_mainland_mask.tif")
n_file = os.path.join(soil_maps_dir, "n.tif")          # porosity [-]
s_fc_file = os.path.join(soil_maps_dir, "s_fc.tif")    # field capacity (relative sat) [-]
s_w_file = os.path.join(soil_maps_dir, "s_w.tif")      # wilting point (relative sat) [-]

CROP_MAP = {'vite': 1, 'olivo': 2, 'pesco': 3, 'agrumi': 4, 'grano': 5}
ZR_RULES = {
    1: {'irr': 1.50, 'rain': 1.80},
    2: {'irr': 1.45, 'rain': 1.80},
    3: {'irr': 1.50, 'rain': 1.80},
    4: {'irr': 1.50, 'rain': 1.80},
    5: {'irr': 1.50, 'rain': 1.50}
}

# --- FUNZIONI ---
def smart_find_file(base_folder, filename):
    p1 = os.path.join(base_folder, filename)
    if os.path.exists(p1): return p1
    p2 = os.path.join(base_folder, os.path.basename(base_folder), filename)
    if os.path.exists(p2): return p2
    return None

def load_mat(filepath, var_name):
    try: return scipy.io.loadmat(filepath)[var_name]
    except: 
        with h5py.File(filepath, 'r') as f: return np.array(f.get(var_name)).T

def print_stats(name, data):
    clean = data[np.isfinite(data)]
    if clean.size == 0: return "NaN", "NaN", "NaN"
    return f"{np.min(clean):.2f}", f"{np.mean(clean):.2f}", f"{np.max(clean):.2f}"

def crop_center(arr, target_h, target_w):
    if arr is None or arr.ndim < 2: return None
    h, w = arr.shape[:2]
    sr, sc = (h - target_h) // 2, (w - target_w) // 2
    if arr.ndim == 3: return arr[sr:sr+target_h, sc:sc+target_w, :]
    return arr[sr:sr+target_h, sc:sc+target_w]

# --- MAIN ---
def main():
    print(f"Output salvati in: {DIAGNOSIS_OUTPUT_DIR}")
    
    # Load Reference mask and soil params (39x43, same as New_WB_MERIDA.py)
    with rasterio.open(ref_mask_path) as src:
        land_mask = src.read(1).astype(bool)
        tif_h, tif_w = land_mask.shape

    with rasterio.open(n_file) as src:
        n_map = src.read(1).astype(np.float64)
    with rasterio.open(s_fc_file) as src:
        s_fc_map = src.read(1).astype(np.float64)
    with rasterio.open(s_w_file) as src:
        s_w_map = src.read(1).astype(np.float64)

    n_vol = n_map[PIXEL_ROW, PIXEL_COL]
    if not np.isfinite(n_vol) or n_vol <= 0: n_vol = 0.45

    s_fc = s_fc_map[PIXEL_ROW, PIXEL_COL]
    s_wp = s_w_map[PIXEL_ROW, PIXEL_COL]

    # Loop Scenarios
    subfolders = [f.name for f in os.scandir(main_output_dir) if f.is_dir()]
    
    for folder in subfolders:
        parts = folder.split('_')
        if len(parts) < 2: continue
        crop_name = parts[0]
        scenario = parts[1]
        
        if crop_name not in CROP_MAP or scenario not in SCENARIO_PATHS: continue

        print(f"\n>> Processing: {crop_name.upper()} - {scenario.upper()}")

        crop_code = CROP_MAP[crop_name]
        orig_mat_dir_base = SCENARIO_PATHS[scenario]
        new_out_dir = os.path.join(main_output_dir, folder)

        is_rainfed = (scenario == 'rainfed')
        # Rainfed uses crop codes 6-10 in DRIP15-RAINFED610
        file_crop_code = crop_code + 5 if is_rainfed else crop_code
        zr_deep_val = ZR_RULES[crop_code]['rain'] if is_rainfed else ZR_RULES[crop_code]['irr']
        cap_deep_mm = n_vol * zr_deep_val * 1000.0

        list_s_orig, list_s_new = [], []
        list_ET_orig, list_T_new = [], []
        list_L_new, list_L_orig = [], []
        list_I_new, list_I_orig = [], []

        # Non contiamo i giorni globali qui, accumuliamo tutto e poi creiamo gli assi
        day_count_orig = 0

        for year in range(YEAR_START, YEAR_END + 1):
            for month in range(1, 13):
                # Paths
                f_s_new = os.path.join(new_out_dir, f"shallow_s_{year}_{month}.mat")
                f_pr = smart_find_file(orig_mat_dir_base, f"outputPR_{year}_{month}_{file_crop_code}.mat")

                # Originali
                f_dp = smart_find_file(orig_mat_dir_base, f"outputDP_{year}_{month}_{file_crop_code}.mat")
                f_bw = smart_find_file(orig_mat_dir_base, f"outputBW_{year}_{month}_{file_crop_code}.mat")
                f_l_old = smart_find_file(orig_mat_dir_base, f"outputL_{year}_{month}_{file_crop_code}.mat")
                f_et = smart_find_file(orig_mat_dir_base, f"outputET_{year}_{month}_{file_crop_code}.mat")
                f_tb = smart_find_file(orig_mat_dir_base, f"outputTB_{year}_{month}_{file_crop_code}.mat")
                f_s_deep = smart_find_file(orig_mat_dir_base, f"outputS_{year}_{month}_{file_crop_code}.mat")

                if not os.path.exists(f_s_new): continue
                if f_pr is None or f_et is None: continue

                # New Data (High Res)
                s_new_grid = load_mat(f_s_new, 's_shallow')
                l_new_grid = load_mat(os.path.join(new_out_dir, f"shallow_L_{year}_{month}.mat"), 'L_shallow')
                t_new_grid = load_mat(os.path.join(new_out_dir, f"shallow_T_{year}_{month}.mat"), 'T_shallow')
                f_i_new = os.path.join(new_out_dir, f"shallow_I_{year}_{month}.mat")
                if os.path.exists(f_i_new):
                    i_new_grid = load_mat(f_i_new, 'I_shallow')
                    list_I_new.append(i_new_grid[PIXEL_ROW, PIXEL_COL, :])
                else:
                    list_I_new.append(np.zeros(s_new_grid.shape[2]))  # fallback
                
                list_s_new.append(s_new_grid[PIXEL_ROW, PIXEL_COL, :])
                list_L_new.append(l_new_grid[PIXEL_ROW, PIXEL_COL, :])
                list_T_new.append(t_new_grid[PIXEL_ROW, PIXEL_COL, :])

                # Orig Data (Daily) - outputET in WATNEEDS is actual ET
                et_raw = load_mat(f_et, 'outputET')
                if et_raw.ndim == 3:
                    n_days_file = et_raw.shape[2]
                    et_crop = crop_center(et_raw, tif_h, tif_w)
                    list_ET_orig.append(et_crop[PIXEL_ROW, PIXEL_COL, :])
                else:
                    n_days_file = et_raw.shape[1]
                    list_ET_orig.append(np.full(n_days_file, np.nan))

                day_count_orig += n_days_file

                s_deep_raw = None
                if f_tb: s_deep_raw = load_mat(f_tb, 'outputTB')
                elif f_s_deep: s_deep_raw = load_mat(f_s_deep, 'outputS')

                if s_deep_raw is not None:
                    s_deep_crop = crop_center(s_deep_raw, tif_h, tif_w)
                    s_deep_mm = s_deep_crop[PIXEL_ROW, PIXEL_COL, :]
                    list_s_orig.append(s_deep_mm / cap_deep_mm)
                else:
                    list_s_orig.append(np.full(n_days_file, np.nan))

                l_raw = None
                if f_dp: l_raw = load_mat(f_dp, 'outputDP')
                elif f_l_old:
                    l_raw = load_mat(f_l_old, 'outputL')
                    if l_raw is None: l_raw = load_mat(f_l_old, 'outputD')

                if l_raw is not None:
                    l_crop = crop_center(l_raw, tif_h, tif_w)
                    list_L_orig.append(l_crop[PIXEL_ROW, PIXEL_COL, :])
                else:
                    list_L_orig.append(np.full(n_days_file, np.nan))

                # Infiltration orig = PR + BW (mm/day)
                pr_raw = load_mat(f_pr, 'outputPR')
                pr_crop = crop_center(pr_raw, tif_h, tif_w)
                pr_day = pr_crop[PIXEL_ROW, PIXEL_COL, :] if pr_crop is not None and pr_crop.ndim == 3 else np.zeros(n_days_file)
                bw_day = np.zeros(n_days_file)
                if f_bw:
                    bw_raw = load_mat(f_bw, 'outputBW')
                    bw_crop = crop_center(bw_raw, tif_h, tif_w)
                    if bw_crop is not None:
                        bw_day = bw_crop[PIXEL_ROW, PIXEL_COL, :]
                list_I_orig.append(np.where(np.isfinite(pr_day), pr_day, 0) + np.where(np.isfinite(bw_day), bw_day, 0))

        if day_count_orig == 0:
            print("XX Dati mancanti.")
            continue

        # Concatenazione
        s_new_raw = np.concatenate(list_s_new)
        L_new_raw = np.concatenate(list_L_new)
        T_new_raw = np.concatenate(list_T_new)
        I_new_raw = np.concatenate(list_I_new)
        
        s_orig = np.concatenate(list_s_orig)
        L_orig = np.concatenate(list_L_orig)
        ET_orig = np.concatenate(list_ET_orig)
        I_orig = np.concatenate(list_I_orig)  # mm/day (PR + BW)

        # --- AGGREGATE SHALLOW TO DAILY (L, T, I in mm/day) ---
        # Shallow: 6 steps/day (4h). Orig: daily. len(s_new_raw) = 6 * n_days
        n_days = len(s_orig)
        n_steps = n_days * 6
        if len(s_new_raw) != n_steps:
            print(f"    !! Length mismatch: s_new={len(s_new_raw)}, expected {n_steps}. Trimming/padding.")
            n_steps = min(len(s_new_raw), n_steps)
            n_days_eff = n_steps // 6
        else:
            n_days_eff = n_days
        s_new = np.nanmean(s_new_raw[:n_days_eff*6].reshape(n_days_eff, 6), axis=1)
        L_new = np.nansum(L_new_raw[:n_days_eff*6].reshape(n_days_eff, 6), axis=1)  # mm/day
        T_new = np.nansum(T_new_raw[:n_days_eff*6].reshape(n_days_eff, 6), axis=1)  # mm/day
        I_new = np.nansum(I_new_raw[:n_days_eff*6].reshape(n_days_eff, 6), axis=1)  # mm/day
        s_orig = s_orig[:n_days_eff]
        L_orig = L_orig[:n_days_eff]
        ET_orig = ET_orig[:n_days_eff]
        I_orig = I_orig[:n_days_eff]

        time_daily = np.arange(n_days_eff)

        # Cumulate (mm)
        cum_L_new = np.nancumsum(np.nan_to_num(L_new))
        cum_L_orig = np.nancumsum(np.nan_to_num(L_orig))

        # Stats (daily, same units)
        print(f"{'VAR':<12} | {'MIN':<10} | {'MEAN':<10} | {'MAX':<10}")
        print("-" * 52)
        print(f"{'S Shallow':<12} | {print_stats('S', s_new)[0]:<10} | {print_stats('S', s_new)[1]:<10} | {print_stats('S', s_new)[2]:<10}")
        print(f"{'S Deep':<12} | {print_stats('S', s_orig)[0]:<10} | {print_stats('S', s_orig)[1]:<10} | {print_stats('S', s_orig)[2]:<10}")
        print(f"{'L Shallow mm/d':<12} | {print_stats('L', L_new)[0]:<10} | {print_stats('L', L_new)[1]:<10} | {print_stats('L', L_new)[2]:<10}")
        print(f"{'L Deep mm/d':<12} | {print_stats('L', L_orig)[0]:<10} | {print_stats('L', L_orig)[1]:<10} | {print_stats('L', L_orig)[2]:<10}")
        print(f"{'T Shallow mm/d':<12} | {print_stats('T', T_new)[0]:<10} | {print_stats('T', T_new)[1]:<10} | {print_stats('T', T_new)[2]:<10}")
        print(f"{'ET Deep mm/d':<12} | {print_stats('ET', ET_orig)[0]:<10} | {print_stats('ET', ET_orig)[1]:<10} | {print_stats('ET', ET_orig)[2]:<10}")
        print(f"{'I Shallow mm/d':<12} | {print_stats('I', I_new)[0]:<10} | {print_stats('I', I_new)[1]:<10} | {print_stats('I', I_new)[2]:<10}")
        print(f"{'I Orig mm/d':<12} | {print_stats('I', I_orig)[0]:<10} | {print_stats('I', I_orig)[1]:<10} | {print_stats('I', I_orig)[2]:<10}")
        print("-" * 52)

        # PLOT (5 pannelli: S, L, cum L, ET, I)
        fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
        fig.suptitle(f"{crop_name.upper()} - {scenario.upper()} ({YEAR_START}-{YEAR_END})\nPixel ({PIXEL_ROW}, {PIXEL_COL})", fontsize=14, fontweight='bold')

        # 1. MOISTURE (both daily)
        ax = axes[0]
        ax.plot(time_daily, s_new, color='#009E73', label='Shallow (daily mean)', linewidth=0.8)
        ax.plot(time_daily, s_orig, color='gray', linestyle=':', label=f'Deep ({zr_deep_val}m)', alpha=0.9, linewidth=1.5)
        
        ax.axhline(s_fc, color='blue', linestyle='--', label='FC')
        ax.axhline(s_wp, color='red', linestyle='--', label='WP')
        ax.set_ylabel('Saturation (0-1)')
        ax.legend(loc='upper right', ncol=2, fontsize='small')
        ax.set_title("1. Soil Moisture Saturation")
        ax.grid(True, alpha=0.3)

        # 2. LEACHING (both mm/day)
        ax = axes[1]
        ax.plot(time_daily, L_new, color='#D55E00', label='L Shallow (mm/day)', linewidth=0.8)
        ax.plot(time_daily, L_orig, color='gray', linestyle=':', label='L Deep (mm/day)', alpha=0.8)
        ax.set_ylabel('Leaching (mm/day)')
        ax.legend(loc='upper right')
        ax.set_title("2. Leaching Flux")
        ax.grid(True, alpha=0.3)

        # 3. CUMULATIVE LEACHING
        ax = axes[2]
        ax.plot(time_daily, cum_L_new, color='#D55E00', linewidth=2, label='Cum. L Shallow')
        ax.plot(time_daily, cum_L_orig, color='black', linestyle='--', linewidth=1.5, label='Cum. L Deep')
        ax.set_ylabel('Cumulative Leaching (mm)')
        ax.legend(loc='upper left')
        ax.set_title("3. Cumulative Leaching (Total Loss)")
        ax.grid(True, alpha=0.3)

        # 4. ET (both mm/day)
        ax = axes[3]
        ax.plot(time_daily, T_new, color='#0072B2', label='T Shallow (mm/day)', linewidth=0.8)
        ax.plot(time_daily, ET_orig, color='gray', linestyle=':', label='ET Deep (mm/day)', alpha=0.8)
        ax.set_ylabel('Flux (mm/day)')
        ax.legend(loc='upper right')
        ax.set_title("4. Evapotraspiration")
        ax.grid(True, alpha=0.3)

        # 5. Infiltration (both mm/day)
        ax = axes[4]
        ax.plot(time_daily, I_new, color='#56B4E9', label='I Shallow (mm/day)', linewidth=0.8)
        ax.plot(time_daily, I_orig, color='gray', linestyle=':', label='I Orig PR+BW (mm/day)', alpha=0.8)
        ax.set_ylabel('Infiltration (mm/day)')
        ax.set_xlabel('Days from start')
        ax.legend(loc='upper right')
        ax.set_title("5. Infiltration (Rain only)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"Diagnosis_HR_{crop_name}_{scenario}_Pix{PIXEL_ROW}-{PIXEL_COL}.png"
        out_path = os.path.join(DIAGNOSIS_OUTPUT_DIR, filename)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"OK Plot Salvato: {filename}")

if __name__ == "__main__":
    main()
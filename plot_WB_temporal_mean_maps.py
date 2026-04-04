"""
Plot temporal mean maps of s, L, I, T from shallow WB outputs.
For each scenario (crop_scenario folder), computes pixel-wise temporal mean:
- s: mean saturation [0-1]
- L, I, T: aggregate 6 steps/day -> daily (mm/day), then mean over time
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
import scipy.io
import h5py

# --- CONFIGURATION ---
base_dir = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW"
main_output_dir = os.path.join(base_dir, "SMEW_Output_4Hour_Raw_Steps")
ref_map_path = os.path.join(base_dir, r"Aree_coltivate\sicilia_cellarea_10km.tif")
OUTPUT_DIR = os.path.join(base_dir, "WB_TemporalMean_Maps")

YEAR_START = 1983
YEAR_END = 2023
N_SUBSTEPS = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_mat(filepath, var_name):
    try:
        return scipy.io.loadmat(filepath)[var_name]
    except Exception:
        with h5py.File(filepath, 'r') as f:
            return np.array(f.get(var_name)).T


def compute_temporal_mean_maps(folder_path):
    """
    Load all shallow_*.mat from folder, compute temporal mean per pixel.
    Returns: maps_s, maps_L, maps_I, maps_T (H, W) in mm/day for L,I,T
    """
    list_s, list_L, list_I, list_T = [], [], [], []
    
    for year in range(YEAR_START, YEAR_END + 1):
        for month in range(1, 13):
            f_s = os.path.join(folder_path, f"shallow_s_{year}_{month}.mat")
            if not os.path.exists(f_s):
                continue
            try:
                s_grid = load_mat(f_s, 's_shallow')  # (H, W, steps)
                L_grid = load_mat(os.path.join(folder_path, f"shallow_L_{year}_{month}.mat"), 'L_shallow')
                T_grid = load_mat(os.path.join(folder_path, f"shallow_T_{year}_{month}.mat"), 'T_shallow')
                f_i = os.path.join(folder_path, f"shallow_I_{year}_{month}.mat")
                I_grid = load_mat(f_i, 'I_shallow') if os.path.exists(f_i) else np.zeros_like(L_grid)
            except Exception as e:
                print(f"    Skip {year}-{month}: {e}")
                continue
            
            n_days = s_grid.shape[2] // N_SUBSTEPS
            n_steps = n_days * N_SUBSTEPS
            
            # s: mean over all steps
            list_s.append(s_grid[:, :, :n_steps])
            # L, I, T: reshape (H,W,n_days,6), sum axis=3 -> mm/day
            L_daily = np.nansum(L_grid[:, :, :n_steps].reshape(*L_grid.shape[:2], n_days, N_SUBSTEPS), axis=3)
            I_daily = np.nansum(I_grid[:, :, :n_steps].reshape(*I_grid.shape[:2], n_days, N_SUBSTEPS), axis=3)
            T_daily = np.nansum(T_grid[:, :, :n_steps].reshape(*T_grid.shape[:2], n_days, N_SUBSTEPS), axis=3)
            list_L.append(L_daily)
            list_I.append(I_daily)
            list_T.append(T_daily)
    
    if len(list_s) == 0:
        return None, None, None, None
    
    s_all = np.concatenate(list_s, axis=2)
    L_all = np.concatenate(list_L, axis=2)
    I_all = np.concatenate(list_I, axis=2)
    T_all = np.concatenate(list_T, axis=2)
    
    map_s = np.nanmean(s_all, axis=2)
    map_L = np.nanmean(L_all, axis=2)
    map_I = np.nanmean(I_all, axis=2)
    map_T = np.nanmean(T_all, axis=2)
    
    return map_s, map_L, map_I, map_T


def main():
    print(f"📂 Output: {OUTPUT_DIR}")
    
    with rasterio.open(ref_map_path) as src:
        ref_data = src.read(1).astype(float)
        ref_profile = src.profile.copy()
    
    land_mask = np.isfinite(ref_data) & (ref_data > 0)
    tif_h, tif_w = ref_data.shape
    
    subfolders = sorted([f.name for f in os.scandir(main_output_dir) if f.is_dir()])
    
    for folder in subfolders:
        parts = folder.split('_')
        if len(parts) < 2:
            continue
        crop_name = parts[0]
        scenario = parts[1]
        
        folder_path = os.path.join(main_output_dir, folder)
        print(f"\n🔄 Processing: {crop_name.upper()} - {scenario.upper()}")
        
        map_s, map_L, map_I, map_T = compute_temporal_mean_maps(folder_path)
        if map_s is None:
            print(f"   ⚠️ No data found.")
            continue
        
        # Apply land mask (NaN for sea)
        map_s = np.where(land_mask, map_s, np.nan)
        map_L = np.where(land_mask, map_L, np.nan)
        map_I = np.where(land_mask, map_I, np.nan)
        map_T = np.where(land_mask, map_T, np.nan)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
        fig.suptitle(f"Temporal Mean — {crop_name.upper()} {scenario.upper()}\n({YEAR_START}-{YEAR_END})", fontsize=14)
        
        # 1. s (saturation)
        ax = axes[0, 0]
        im0 = ax.imshow(map_s, cmap='YlGnBu', vmin=0, vmax=1, origin='upper')
        ax.set_title("s (saturation) [-]")
        plt.colorbar(im0, ax=ax, label='[-]')
        ax.set_axis_off()
        
        # 2. L (leaching mm/day)
        ax = axes[0, 1]
        L_valid = map_L[np.isfinite(map_L) & (map_L > 0)]
        vmax_L = np.percentile(L_valid, 98) if L_valid.size > 0 else 5.0
        im1 = ax.imshow(map_L, cmap='Oranges', vmin=0, vmax=max(vmax_L, 0.01), origin='upper')
        ax.set_title("L (leaching) [mm/day]")
        plt.colorbar(im1, ax=ax, label='mm/day')
        ax.set_axis_off()
        
        # 3. I (infiltration mm/day)
        ax = axes[1, 0]
        I_valid = map_I[np.isfinite(map_I) & (map_I > 0)]
        vmax_I = np.percentile(I_valid, 98) if I_valid.size > 0 else 10.0
        im2 = ax.imshow(map_I, cmap='Blues', vmin=0, vmax=max(vmax_I, 0.01), origin='upper')
        ax.set_title("I (infiltration) [mm/day]")
        plt.colorbar(im2, ax=ax, label='mm/day')
        ax.set_axis_off()
        
        # 4. T (transpiration mm/day)
        ax = axes[1, 1]
        T_valid = map_T[np.isfinite(map_T) & (map_T > 0)]
        vmax_T = np.percentile(T_valid, 98) if T_valid.size > 0 else 5.0
        im3 = ax.imshow(map_T, cmap='Greens', vmin=0, vmax=max(vmax_T, 0.01), origin='upper')
        ax.set_title("T (transpiration) [mm/day]")
        plt.colorbar(im3, ax=ax, label='mm/day')
        ax.set_axis_off()
        
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"WB_mean_{crop_name}_{scenario}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {out_path}")
        
        # Stats
        for name, arr in [('s', map_s), ('L', map_L), ('I', map_I), ('T', map_T)]:
            v = arr[np.isfinite(arr)]
            if v.size > 0:
                print(f"      {name}: min={np.nanmin(v):.4f}, mean={np.nanmean(v):.4f}, max={np.nanmax(v):.4f}")
    
    print(f"\n✅ Done. Maps in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

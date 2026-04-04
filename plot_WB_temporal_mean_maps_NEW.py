"""
Plot temporal mean maps of s, L, I, T from shallow bucket output (SMEW_Output_4Hour_MERIDA).

All data from MERIDA climate + AIDA crops, processed through New_WB_MERIDA.py (30cm bucket, 4h).
Grid: (39, 43), 4h resolution (6 steps/day).
Variables: s_shallow (saturation [-]), L_shallow (leaching mm/4h), T_shallow (transpiration mm/4h),
           I_shallow (rain infiltration mm/4h).
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io
import h5py
import rasterio

# --- CONFIGURATION ---
base_dir = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW"
ref_mask_path = os.path.join(base_dir, r"Aree_coltivate\sicily_mainland_mask.tif")
OUTPUT_DIR = os.path.join(base_dir, "WB_TemporalMean_Maps_NEW")

# Shallow bucket output (from New_WB_MERIDA.py)
SHALLOW_DIR = os.path.join(base_dir, "SMEW_Output_4Hour_MERIDA")

YEAR_START = 1992
YEAR_END = 2023
TARGET_SHAPE = (39, 43)

# Scenarios: folder names in SMEW_Output_4Hour_MERIDA
SCENARIOS = [
    'vite_drip', 'vite_surface', 'vite_rainfed',
    'olivo_drip', 'olivo_surface', 'olivo_rainfed',
    'pesco_drip', 'pesco_surface', 'pesco_rainfed',
    'agrumi_drip', 'agrumi_surface', 'agrumi_rainfed',
    'grano_rainfed',
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_mat(filepath, varname):
    """Load .mat file, try scipy first then h5py."""
    try:
        return scipy.io.loadmat(filepath)[varname]
    except Exception:
        with h5py.File(filepath, 'r') as f:
            return np.array(f[varname]).T


def compute_maps_shallow(scenario_name):
    """
    Compute temporal mean maps from shallow bucket output.
    Aggregates 4h steps to daily (mean for s, sum for L/T/I), then temporal mean.
    Returns: map_s, map_L, map_I, map_T as (39, 43) arrays in mm/day.
    """
    data_dir = os.path.join(SHALLOW_DIR, scenario_name)
    if not os.path.exists(data_dir):
        return None, None, None, None

    sum_s = np.zeros(TARGET_SHAPE, dtype=np.float64)
    sum_L = np.zeros(TARGET_SHAPE, dtype=np.float64)
    sum_T = np.zeros(TARGET_SHAPE, dtype=np.float64)
    sum_I = np.zeros(TARGET_SHAPE, dtype=np.float64)
    count = np.zeros(TARGET_SHAPE, dtype=np.float64)

    for year in range(YEAR_START, YEAR_END + 1):
        for month in range(1, 13):
            f_s = os.path.join(data_dir, f"shallow_s_{year}_{month}.mat")
            if not os.path.exists(f_s):
                continue
            try:
                s = load_mat(f_s, 's_shallow')
                L = load_mat(os.path.join(data_dir, f"shallow_L_{year}_{month}.mat"), 'L_shallow')
                T = load_mat(os.path.join(data_dir, f"shallow_T_{year}_{month}.mat"), 'T_shallow')
                I = load_mat(os.path.join(data_dir, f"shallow_I_{year}_{month}.mat"), 'I_shallow')
            except Exception as e:
                print(f"    Skip {year}-{month}: {e}")
                continue

            n_steps = s.shape[2]
            n_days = n_steps // 6

            for d in range(n_days):
                sl = slice(d * 6, (d + 1) * 6)
                s_day = np.nanmean(s[:, :, sl], axis=2)
                L_day = np.nansum(L[:, :, sl], axis=2)
                T_day = np.nansum(T[:, :, sl], axis=2)
                I_day = np.nansum(I[:, :, sl], axis=2)

                valid = np.isfinite(s_day)
                sum_s[valid] += s_day[valid]
                sum_L[valid] += L_day[valid]
                sum_T[valid] += T_day[valid]
                sum_I[valid] += I_day[valid]
                count[valid] += 1

    if np.max(count) == 0:
        return None, None, None, None

    count[count == 0] = np.nan
    return sum_s / count, sum_L / count, sum_I / count, sum_T / count


def plot_maps(map_s, map_L, map_I, map_T, label, land_mask):
    """Plot 2x2 panel of temporal mean maps."""
    map_s = np.where(land_mask, map_s, np.nan)
    map_L = np.where(land_mask, map_L, np.nan)
    map_I = np.where(land_mask, map_I, np.nan)
    map_T = np.where(land_mask, map_T, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle(f"Temporal Mean Water Balance (30cm bucket) -- {label}", fontsize=14)

    ax = axes[0, 0]
    im0 = ax.imshow(map_s, cmap='YlGnBu', vmin=0, vmax=1, origin='upper')
    ax.set_title("s (saturation) [-]")
    plt.colorbar(im0, ax=ax)
    ax.set_axis_off()

    ax = axes[0, 1]
    L_valid = map_L[np.isfinite(map_L) & (map_L > 0)]
    vmax_L = np.percentile(L_valid, 98) if L_valid.size > 0 else 5.0
    im1 = ax.imshow(map_L, cmap='Oranges', vmin=0, vmax=max(vmax_L, 0.01), origin='upper')
    ax.set_title("L (leaching) [mm/d]")
    plt.colorbar(im1, ax=ax)
    ax.set_axis_off()

    ax = axes[1, 0]
    I_valid = map_I[np.isfinite(map_I) & (map_I > 0)]
    vmax_I = np.percentile(I_valid, 98) if I_valid.size > 0 else 10.0
    im2 = ax.imshow(map_I, cmap='Blues', vmin=0, vmax=max(vmax_I, 0.01), origin='upper')
    ax.set_title("I (rain infiltration) [mm/d]")
    plt.colorbar(im2, ax=ax)
    ax.set_axis_off()

    ax = axes[1, 1]
    T_valid = map_T[np.isfinite(map_T) & (map_T > 0)]
    vmax_T = np.percentile(T_valid, 98) if T_valid.size > 0 else 5.0
    im3 = ax.imshow(map_T, cmap='Greens', vmin=0, vmax=max(vmax_T, 0.01), origin='upper')
    ax.set_title("T (transpiration) [mm/d]")
    plt.colorbar(im3, ax=ax)
    ax.set_axis_off()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"WB_mean_{label}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {out_path}")

    for name, arr in [('s', map_s), ('L', map_L), ('I', map_I), ('T', map_T)]:
        v = arr[np.isfinite(arr)]
        if v.size > 0:
            print(f"      {name:6s}: min={np.nanmin(v):.4f}, mean={np.nanmean(v):.4f}, max={np.nanmax(v):.4f}")


def main():
    print(f"Output: {OUTPUT_DIR}")

    with rasterio.open(ref_mask_path) as src:
        land_mask = src.read(1).astype(bool)

    for scenario in SCENARIOS:
        print(f"\nProcessing: {scenario}")
        map_s, map_L, map_I, map_T = compute_maps_shallow(scenario)

        if map_s is None:
            print(f"   No data found for {scenario}")
            continue

        plot_maps(map_s, map_L, map_I, map_T, scenario, land_mask)

    print(f"\nDone. Maps in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

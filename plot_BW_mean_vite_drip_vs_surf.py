"""
Plot temporal mean BW (irrigation) maps for vite: DRIP vs SURFACE
from the MERIDA/AIDA dataset (DRIP15-RAINFED610 and SURFACE folders).

Vite = crop 1 (file suffix _1).
Grid: (41, 45) raw — no trimming applied.
"""

import numpy as np
import os
import scipy.io
import h5py
import matplotlib.pyplot as plt

base_dir = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW"

DRIP_DIR = os.path.join(base_dir, r"DRIP15-RAINFED610\giornalieri")
SURF_DIR = os.path.join(base_dir, r"SURFACE\giornalieri")

YEAR_START = 1992
YEAR_END = 2023
CROP_CODE = 1  # vite
TARGET_SHAPE = (41, 45)


def load_mat(filepath, varname):
    try:
        return scipy.io.loadmat(filepath)[varname]
    except Exception:
        with h5py.File(filepath, 'r') as f:
            return np.array(f[varname]).T


def compute_mean_bw(data_dir, crop_code):
    """Compute temporal mean BW [mm/d] over full period."""
    total_bw = np.zeros(TARGET_SHAPE, dtype=np.float64)
    total_days = np.zeros(TARGET_SHAPE, dtype=np.float64)

    for year in range(YEAR_START, YEAR_END + 1):
        for month in range(1, 13):
            fname = os.path.join(data_dir, f"outputBW_{year}_{month}_{crop_code}.mat")
            if not os.path.exists(fname):
                continue
            try:
                data = load_mat(fname, 'outputBW')
                bw = data  # raw grid, no trim
                ndays = bw.shape[2]
                daily_sum = np.nansum(bw, axis=2)
                total_bw += daily_sum
                total_days += ndays
            except Exception as e:
                print(f"  Skip {year}-{month}: {e}")

    total_days[total_days == 0] = np.nan
    return total_bw / total_days


def main():
    print("Computing temporal mean BW for vite...")
    mean_drip = compute_mean_bw(DRIP_DIR, CROP_CODE)
    mean_surf = compute_mean_bw(SURF_DIR, CROP_CODE)

    # Stats
    for label, arr in [("DRIP", mean_drip), ("SURFACE", mean_surf)]:
        v = arr[np.isfinite(arr)]
        n_irr = np.sum(v > 0)
        print(f"\n{label}:")
        print(f"  Pixels with BW > 0: {n_irr}")
        print(f"  Mean BW (all pixels):      {np.nanmean(v):.3f} mm/d")
        if n_irr > 0:
            print(f"  Mean BW (irrigated only):  {np.nanmean(v[v > 0]):.3f} mm/d")
            print(f"  Max BW:                    {np.nanmax(v):.2f} mm/d")

    # --- Plot ---
    vmax_global = max(np.nanmax(mean_drip[np.isfinite(mean_drip)]),
                      np.nanmax(mean_surf[np.isfinite(mean_surf)]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Temporal Mean Irrigation (BW) — Vite (MERIDA/AIDA dataset, 1992-2023)", fontsize=13)

    ax = axes[0]
    im = ax.imshow(mean_drip, cmap='Blues', vmin=0, vmax=vmax_global, origin='upper')
    n_drip = np.sum(np.isfinite(mean_drip) & (mean_drip > 0))
    ax.set_title(f"DRIP  ({n_drip} irrigated pixels)")
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, label="mm/d", shrink=0.8)

    ax = axes[1]
    im = ax.imshow(mean_surf, cmap='Blues', vmin=0, vmax=vmax_global, origin='upper')
    n_surf = np.sum(np.isfinite(mean_surf) & (mean_surf > 0))
    ax.set_title(f"SURFACE  ({n_surf} irrigated pixels)")
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, label="mm/d", shrink=0.8)

    plt.tight_layout()
    out_path = os.path.join(base_dir, "BW_mean_vite_DRIP_vs_SURFACE_MERIDA.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

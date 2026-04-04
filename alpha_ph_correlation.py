import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.stats import pearsonr
from scipy.ndimage import distance_transform_edt

# --- CONFIGURATION ---
CALIB_DIR   = r"C:\Users\Latitude 5511\Downloads"
PH_FILE     = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\soil_param (1)\sicily_ph_cacl2_10km.tif"
MAINLAND_FILE = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\Aree_coltivate\sicily_mainland_mask.tif"

# Crop area map for each scenario (used for cultivated pixel overlay)
CROP_AREA = {
    "Agrumi Drip":        r"C:\Users\Latitude 5511\Downloads\sicily10km_citrus_total_ha.tif",
    "Grano Rainfed":      r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\Aree_coltivate\sicily10km_wheat_total_ha.tif",
    "Olivo Drip":         r"C:\Users\Latitude 5511\Downloads\sicily10km_olives_total_ha.tif",
    "Olivo Traditional":  r"C:\Users\Latitude 5511\Downloads\sicily10km_olives_total_ha.tif",
    "Vite Drip":          r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\Aree_coltivate\sicily10km_vineyard_total_ha.tif",
}

# Each entry: (label, alpha_glob_pattern, output_png)
SCENARIOS = [
    ("Agrumi Drip",
     "FULL_AGRUMI_DRIP_30Y_alpha_map_MERGED*.tif",
     "alpha_ph_correlation_agrumi_drip.png"),

    ("Grano Rainfed",
     "FULL_GRANO_RAINFED_30Y_alpha_map_MERGED*.tif",
     "alpha_ph_correlation_grano_rainfed.png"),

    ("Olivo Drip",
     "FULL_OLIVO_DRIP_30Y_alpha_map_MERGED*.tif",
     "alpha_ph_correlation_olivo_drip.png"),

    ("Olivo Traditional",
     "FULL_OLIVO_TRADITIONAL_30Y_alpha_map_MERGED*.tif",
     "alpha_ph_correlation_olivo_traditional.png"),

    ("Vite Drip",
     "FULL_VITE_DRIP_30Y_alpha_map_MERGED*.tif",
     "alpha_ph_correlation_vite_drip.png"),
]


def nn_fill(arr):
    """Nearest-neighbour fill of NaN — same logic as calibration script."""
    arr = arr.copy()
    mask = np.isnan(arr)
    if mask.any() and (~mask).any():
        _, idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
        arr[mask] = arr[idx[0][mask], idx[1][mask]]
    return arr


def load_tif(path):
    with rasterio.open(path) as src:
        d = src.read(1).astype(float)
        if src.nodata is not None:
            d[d == src.nodata] = np.nan
    return d


def find_latest(directory, pattern):
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def analyze_scenario(label, alpha_file, ph_raw, ph_filled, mainland, output_path,
                     crop_area_file=None):
    print(f"\n--- {label} ---")
    print(f"  Alpha : {os.path.basename(alpha_file)}")

    alpha_map = load_tif(alpha_file)
    alpha_map[alpha_map <= 0] = np.nan

    # Crop mask
    crop_mask = None
    if crop_area_file and os.path.exists(crop_area_file):
        crop_area = load_tif(crop_area_file)
        crop_area[~np.isfinite(crop_area)] = 0.0
        crop_mask = (crop_area > 0) & mainland
        crop_rows, crop_cols = np.where(crop_mask)
    else:
        crop_rows, crop_cols = np.array([]), np.array([])

    # Use NN-filled pH for scatter (matches calibration target)
    valid_mask = ~np.isnan(alpha_map.flatten()) & ~np.isnan(ph_filled.flatten())
    # Restrict to crop mask if available
    if crop_mask is not None:
        valid_mask = valid_mask & crop_mask.flatten()

    alpha  = alpha_map.flatten()[valid_mask]
    ph_vec = ph_filled.flatten()[valid_mask]
    print(f"  Valid pixels (in crop mask): {len(alpha)}")

    # pH display map: NN-filled, masked to mainland
    ph_display = ph_filled.copy()
    ph_display[~mainland] = np.nan

    # ── Layout: 2 rows × 2 cols ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"{label} — Alpha Calibration Diagnostics", fontsize=13, fontweight='bold')

    # Panel [0,0]: Alpha vs NN-interpolated pH (scatter)
    ax = axes[0, 0]
    if len(alpha) > 0:
        sc = ax.scatter(ph_vec, alpha, alpha=0.5, s=12,
                        c=alpha, cmap='plasma',
                        vmin=np.nanpercentile(alpha, 2),
                        vmax=np.nanpercentile(alpha, 98),
                        edgecolors='none')
        plt.colorbar(sc, ax=ax, label='Alpha')
        corr, _ = pearsonr(ph_vec, alpha)
        ax.set_title(f"Alpha vs NN-interpolated pH  (r = {corr:.2f})")
    else:
        ax.set_title("Alpha vs pH  (no data)")
    ax.set_xlabel("Soil pH (CaCl₂) — NN-interpolated")
    ax.set_ylabel("Calibrated Alpha")
    ax.grid(True, alpha=0.3)

    # Panel [0,1]: Alpha histogram
    ax = axes[0, 1]
    if len(alpha) > 0:
        ax.hist(alpha, bins=50, color='purple', alpha=0.7, edgecolor='black', linewidth=0.3)
        for g in [0.5, 1.0, 1.5, 2.5, 6.0]:
            ax.axvline(g, color='orange', linestyle=':', alpha=0.6, linewidth=1)
        ax.text(0.97, 0.97,
                f"n = {len(alpha)}\nmean = {np.mean(alpha):.2f}\nmedian = {np.median(alpha):.2f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.set_title("Alpha Distribution")
    ax.set_xlabel("Alpha Value")
    ax.set_ylabel("Frequency (Pixels)")
    ax.grid(True, alpha=0.3)

    # Panel [1,0]: NN-interpolated target pH map + cultivated pixels overlay
    ax = axes[1, 0]
    vmin_ph = np.nanpercentile(ph_display[mainland], 2)
    vmax_ph = np.nanpercentile(ph_display[mainland], 98)
    im_ph = ax.imshow(ph_display, cmap='RdYlGn', vmin=vmin_ph, vmax=vmax_ph,
                      origin='upper', aspect='auto')
    plt.colorbar(im_ph, ax=ax, label='pH (CaCl₂)', shrink=0.85)
    if len(crop_rows) > 0:
        n_cult = int(crop_mask.sum()) if crop_mask is not None else 0
        ax.scatter(crop_cols, crop_rows, s=10, c='black', marker='o',
                   alpha=0.6, linewidths=0, label=f'Cultivated ({n_cult} px)')
        ax.legend(fontsize=7, loc='lower right')
    ax.set_title("NN-interpolated target pH\n+ cultivated pixels")
    ax.set_xlabel("Col (W→E)")
    ax.set_ylabel("Row (N→S)")

    # Panel [1,1]: Spatial alpha map + cultivated pixels overlay
    ax = axes[1, 1]
    alpha_display = alpha_map.copy()
    alpha_display[~mainland] = np.nan
    vmin_a = np.nanpercentile(alpha_map[~np.isnan(alpha_map)], 2)
    vmax_a = np.nanpercentile(alpha_map[~np.isnan(alpha_map)], 98)
    im_a = ax.imshow(alpha_display, cmap='plasma', vmin=vmin_a, vmax=vmax_a,
                     origin='upper', aspect='auto')
    plt.colorbar(im_a, ax=ax, label='Alpha', shrink=0.85)
    if len(crop_rows) > 0:
        # Failed pixels (crop_mask but no alpha)
        if crop_mask is not None:
            fail_mask = crop_mask & np.isnan(alpha_map)
            fail_rows, fail_cols = np.where(fail_mask)
            if len(fail_rows) > 0:
                ax.scatter(fail_cols, fail_rows, s=30, c='red', marker='x',
                           label=f'Failed ({len(fail_rows)})', zorder=4, linewidths=1.2)
        ax.scatter(crop_cols, crop_rows, s=8, c='white', marker='o',
                   alpha=0.3, linewidths=0, label='Cultivated')
        ax.legend(fontsize=7, loc='lower right')
    ax.set_title("Spatial Alpha Map\n(red × = calibration failures)")
    ax.set_xlabel("Col (W→E)")
    ax.set_ylabel("Row (N→S)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    # Load shared maps once
    ph_raw  = load_tif(PH_FILE)
    ph_raw[(ph_raw <= 0) | (ph_raw >= 14)] = np.nan
    ph_filled = nn_fill(ph_raw)

    mainland = load_tif(MAINLAND_FILE).astype(bool)

    for label, alpha_pat, out_name in SCENARIOS:
        alpha_file = find_latest(CALIB_DIR, alpha_pat)
        if alpha_file is None:
            print(f"\n--- {label} --- SKIPPED (alpha map not found in {CALIB_DIR})")
            continue

        crop_area_file = CROP_AREA.get(label)
        output_path = os.path.join(CALIB_DIR, out_name)
        analyze_scenario(label, alpha_file, ph_raw, ph_filled, mainland,
                         output_path, crop_area_file=crop_area_file)

    print("\nDone.")

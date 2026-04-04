import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from scipy.ndimage import distance_transform_edt

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
CROP       = 'olivo'
IRRIGATION = 'traditional'
LABEL      = f'{CROP.capitalize()} {IRRIGATION.capitalize()}'

NPZ_PATH    = r"C:\Users\Latitude 5511\Downloads\FULL_OLIVO_TRADITIONAL_30Y_timeseries_MERGED (2).npz"
ALPHA_PATH  = r"C:\Users\Latitude 5511\Downloads\FULL_OLIVO_TRADITIONAL_30Y_alpha_map_MERGED (3).tif"
PH_RAW_PATH = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\soil_param (1)\sicily_ph_cacl2_10km.tif"

CROP_AREA_PATH  = r"C:\Users\Latitude 5511\Downloads\sicily10km_olives_total_ha.tif"
WATNEEDS_MASK   = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\Aree_coltivate\watneeds_ref_mask.tif"
MAINLAND_PATH   = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\Aree_coltivate\sicily_mainland_mask.tif"

OUTDIR = r"C:\Users\Latitude 5511\Downloads"
YEARS  = 30
MAX_TIMESERIES_PLOTS = 10


# ── HELPERS ───────────────────────────────────────────────────────────────────
def nn_fill(arr):
    """Nearest-neighbour fill of NaN — same logic as calibration script."""
    arr = arr.copy()
    mask = np.isnan(arr)
    if mask.any() and (~mask).any():
        _, idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
        arr[mask] = arr[idx[0][mask], idx[1][mask]]
    return arr


def load_tif(path, nodata_to_nan=True):
    with rasterio.open(path) as src:
        d = src.read(1).astype(float)
        if nodata_to_nan and src.nodata is not None:
            d[d == src.nodata] = np.nan
    return d


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # --- Load data ---
    data        = np.load(NPZ_PATH, allow_pickle=True)
    pH_ts       = data['pH_daily']            # (n_pixels, n_days)
    Ca_ts       = data['Ca_daily']
    Mg_ts       = data['Mg_daily']
    pixel_coords = data['pixel_coords']       # (n_pixels, 2)
    map_shape   = tuple(data['map_shape'])    # (39, 43)

    alpha_map = load_tif(ALPHA_PATH)
    alpha_map[alpha_map <= 0] = np.nan

    # Raw ESDAC pH map
    ph_raw = load_tif(PH_RAW_PATH)
    ph_raw[(ph_raw <= 0) | (ph_raw >= 14)] = np.nan

    # NN-interpolated pH (what calibration actually uses as target)
    ph_filled = nn_fill(ph_raw)

    # Masks
    mainland  = load_tif(MAINLAND_PATH).astype(bool)
    crop_area = load_tif(CROP_AREA_PATH)
    crop_mask = (crop_area > 0) & mainland      # cultivated pixels

    # Mask filled pH to mainland only (don't show sea pixels)
    ph_filled_display = ph_filled.copy()
    ph_filled_display[~mainland] = np.nan

    n_pixels = len(pixel_coords)
    n_days   = pH_ts.shape[1]
    t_years  = np.linspace(0, YEARS, n_days)

    alphas  = np.array([alpha_map[r, c] for r, c in pixel_coords])
    # Use NN-filled pH as target (matches what calibration used)
    targets = np.array([ph_filled[r, c] for r, c in pixel_coords])

    ph_final = np.array([np.nanmean(pH_ts[i, -365:]) for i in range(n_pixels)])

    # Build maps
    sim_ph_map = np.full(map_shape, np.nan)
    err_map    = np.full(map_shape, np.nan)
    for i, (r, c) in enumerate(pixel_coords):
        sim_ph_map[r, c] = ph_final[i]
        err_map[r, c]    = ph_final[i] - ph_filled[r, c]

    crop_rows, crop_cols = np.where(crop_mask)

    # =========================================================================
    # FIGURE 1 — pH maps: NN-interpolated target | simulated | error
    # =========================================================================
    vmin_ph = np.nanpercentile(ph_filled_display[mainland], 2)
    vmax_ph = np.nanpercentile(ph_filled_display[mainland], 98)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{LABEL} 30Y Calibration — pH Maps', fontsize=14, fontweight='bold')

    # Panel 1: NN-interpolated target pH + cultivated pixels overlay
    im0 = axes[0].imshow(ph_filled_display, cmap='RdYlGn', vmin=vmin_ph, vmax=vmax_ph,
                         origin='upper', aspect='auto')
    axes[0].scatter(crop_cols, crop_rows, s=12, c='black', marker='o',
                    alpha=0.6, linewidths=0, label=f'Cultivated {CROP} ({crop_mask.sum()} px)')
    fig.colorbar(im0, ax=axes[0], shrink=0.75, label='pH')
    axes[0].set_title('Target pH (NN-interpolated ESDAC)\n+ cultivated pixels', fontsize=10)
    axes[0].legend(fontsize=7, loc='lower right')

    # Panel 2: simulated last-year pH (only calibrated pixels)
    im1 = axes[1].imshow(sim_ph_map, cmap='RdYlGn', vmin=vmin_ph, vmax=vmax_ph,
                         origin='upper', aspect='auto')
    axes[1].scatter(crop_cols, crop_rows, s=12, c='black', marker='o',
                    alpha=0.2, linewidths=0)
    fig.colorbar(im1, ax=axes[1], shrink=0.75, label='pH')
    axes[1].set_title(f'Simulated pH (last-year avg)\n{n_pixels}/{crop_mask.sum()} pixels calibrated', fontsize=10)

    # Gray overlay for uncalibrated crop pixels
    uncalib = crop_mask & np.isnan(sim_ph_map)
    if uncalib.any():
        ov = np.zeros((*map_shape, 4))
        ov[uncalib] = [0.5, 0.5, 0.5, 0.7]
        axes[1].imshow(ov, origin='upper', aspect='auto')

    # Panel 3: error map (sim - target)
    err_abs_max = np.nanpercentile(np.abs(err_map[~np.isnan(err_map)]), 95)
    im2 = axes[2].imshow(err_map, cmap='RdBu_r', vmin=-err_abs_max, vmax=err_abs_max,
                         origin='upper', aspect='auto')
    axes[2].scatter(crop_cols, crop_rows, s=12, c='black', marker='o',
                    alpha=0.2, linewidths=0)
    fig.colorbar(im2, ax=axes[2], shrink=0.75, label='pH error (sim - target)')
    axes[2].set_title('Calibration error\n(sim pH − target pH)', fontsize=10)

    for ax in axes:
        ax.set_xlabel('Col (W→E)')
        ax.set_ylabel('Row (N→S)')

    plt.tight_layout()
    out1 = f"{OUTDIR}\\{CROP}_{IRRIGATION}_30Y_pH_maps.png"
    fig.savefig(out1, dpi=200, bbox_inches='tight')
    print(f"Saved: {out1}")
    plt.close(fig)

    # =========================================================================
    # FIGURE 2 — Cultivated pixel map: area [ha] + calibration status
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    fig2.suptitle(f'{LABEL} — Cultivated Pixel Map', fontsize=13, fontweight='bold')

    # Background: crop area [ha] for cultivated pixels
    area_display = crop_area.copy()
    area_display[~crop_mask] = np.nan
    im_area = ax2.imshow(area_display, cmap='YlGn', origin='upper', aspect='auto')
    fig2.colorbar(im_area, ax=ax2, shrink=0.75, label=f'{CROP} area [ha/pixel]')

    # Overlay calibrated (blue) vs failed (red) pixels
    calib_rows, calib_cols = np.where(crop_mask & ~np.isnan(sim_ph_map))
    fail_rows,  fail_cols  = np.where(crop_mask & np.isnan(sim_ph_map))
    ax2.scatter(calib_cols, calib_rows, s=25, c='steelblue', marker='o',
                label=f'Calibrated ({len(calib_rows)})', zorder=3, edgecolors='none')
    ax2.scatter(fail_cols,  fail_rows,  s=40, c='red', marker='x',
                label=f'Failed ({len(fail_rows)})', zorder=4, linewidths=1.2)

    ax2.set_xlabel('Col (W→E)')
    ax2.set_ylabel('Row (N→S)')
    ax2.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    out2 = f"{OUTDIR}\\{CROP}_{IRRIGATION}_30Y_cultivated_map.png"
    fig2.savefig(out2, dpi=200, bbox_inches='tight')
    print(f"Saved: {out2}")
    plt.close(fig2)

    # =========================================================================
    # FIGURE 3..N — Ca & Mg time series (evenly sampled across alpha range)
    # =========================================================================
    valid_idx    = np.where(~np.isnan(alphas) & ~np.isnan(ph_final))[0]
    sort_by_alpha = valid_idx[np.argsort(alphas[valid_idx])]
    if len(sort_by_alpha) > MAX_TIMESERIES_PLOTS:
        pick     = np.linspace(0, len(sort_by_alpha) - 1, MAX_TIMESERIES_PLOTS, dtype=int)
        selected = sort_by_alpha[pick]
    else:
        selected = sort_by_alpha

    for i in selected:
        r, c = pixel_coords[i]
        fig3, axes3 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig3.suptitle(f'{LABEL} — Pixel ({r},{c})  alpha={alphas[i]:.3f}  target pH={targets[i]:.2f}',
                      fontsize=12)

        axes3[0].plot(t_years, pH_ts[i], color='purple', lw=1)
        axes3[0].axhline(targets[i], color='k', ls='--', label=f'Target {targets[i]:.2f}')
        axes3[0].set_ylabel('pH')
        axes3[0].legend(fontsize=8)
        axes3[0].grid(True, alpha=0.3)

        axes3[1].plot(t_years, Ca_ts[i], color='steelblue', lw=1, label='Ca')
        axes3[1].plot(t_years, Mg_ts[i], color='darkorange', lw=1, label='Mg')
        axes3[1].set_ylabel('Concentration [µmol/L]')
        axes3[1].set_xlabel('Simulation Year')
        axes3[1].legend(fontsize=8)
        axes3[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out3 = f"{OUTDIR}\\{CROP}_{IRRIGATION}_30Y_ts_{r}_{c}.png"
        fig3.savefig(out3, dpi=150, bbox_inches='tight')
        print(f"Saved: {out3}")
        plt.close(fig3)

    # =========================================================================
    # Console summary
    # =========================================================================
    valid_mask_s = ~np.isnan(ph_final) & ~np.isnan(targets)
    rmse = np.sqrt(np.nanmean((ph_final[valid_mask_s] - targets[valid_mask_s])**2))
    bias = np.nanmean(ph_final[valid_mask_s] - targets[valid_mask_s])
    print(f"\n--- SUMMARY: {LABEL} ---")
    print(f"Calibrated pixels : {n_pixels} / {int(crop_mask.sum())}")
    print(f"Failed pixels     : {int(crop_mask.sum()) - n_pixels}")
    print(f"pH RMSE (last yr) : {rmse:.4f}")
    print(f"pH bias           : {bias:.4f}")
    print(f"Alpha range       : {np.nanmin(alphas):.3f} – {np.nanmax(alphas):.3f}  (mean {np.nanmean(alphas):.3f})")


if __name__ == "__main__":
    main()

"""
Compute weighted-mean alpha map from all calibrated scenarios.

For each pixel: alpha_weighted = sum(alpha_i * ha_i) / sum(ha_i)
where i = each crop × irrigation scenario present at that pixel.

Output: weighted_alpha_map.tif (39x43, same georeferencing as alpha maps)
"""
import numpy as np
import rasterio
import os
import sys

# ── PATHS ─────────────────────────────────────────────────────────────────────
DL = r"C:\Users\Latitude 5511\Downloads"
AREA_DIR = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\Aree_coltivate"
WN_FILE = os.path.join(AREA_DIR, "watneeds_ref_mask.tif")
OUT_DIR = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW"

# ── SCENARIO DEFINITIONS ──────────────────────────────────────────────────────
# (crop, irrigation, alpha_file, area_ha_file)
# area_ha_file: pixel-specific ha for this exact crop × irrigation type
# Using _i_ha (irrigated) and _r_ha (rainfed) maps from CLC/ISTAT
# Irrigated area split 50/50 between drip and traditional

SCENARIOS = [
    # Olivo: irrigated split 50/50 drip/trad, rainfed from _r_ha
    ('olivo', 'drip',
     f'{DL}/FULL_OLIVO_DRIP_30Y_alpha_map_MERGED (7).tif',
     f'{AREA_DIR}/sicily10km_olives_i_ha.tif', 0.50),       # 50% of irrigated = drip

    ('olivo', 'traditional',
     f'{DL}/FULL_OLIVO_TRADITIONAL_30Y_alpha_map_MERGED (3).tif',
     f'{AREA_DIR}/sicily10km_olives_i_ha.tif', 0.50),       # 50% of irrigated = traditional

    ('olivo', 'rainfed',
     f'{DL}/FULL_OLIVO_RAINFED_30Y_alpha_map_MERGED.tif',
     f'{AREA_DIR}/sicily10km_olives_r_ha.tif', 1.00),       # 100% of rainfed

    # Vite: only drip calibrated, use irrigated ha; rainfed not yet calibrated
    ('vite', 'drip',
     f'{DL}/FULL_VITE_DRIP_30Y_alpha_map_MERGED (9).tif',
     f'{AREA_DIR}/sicily10km_vineyard_i_ha.tif', 1.00),     # all irrigated = drip (for now)

    # Grano: rainfed only (0% irrigated)
    ('grano', 'rainfed',
     f'{DL}/FULL_GRANO_RAINFED_30Y_alpha_map_MERGED (1).tif',
     f'{AREA_DIR}/sicily10km_wheat_r_ha.tif', 1.00),        # 100% rainfed
]
# Format: (crop, irrigation, alpha_path, area_ha_path, fraction_of_that_area)


def load_tif(path):
    with rasterio.open(path) as src:
        d = src.read(1).astype(np.float64)
        meta = {'transform': src.transform, 'crs': src.crs,
                'height': src.height, 'width': src.width}
        if src.nodata is not None:
            d[d == src.nodata] = np.nan
    return d, meta


def main():
    # Load WATNEEDS ref mask
    with rasterio.open(WN_FILE) as src:
        wn = src.read(1).astype(bool)

    # Get grid shape from first alpha file
    first_alpha, geo = load_tif(SCENARIOS[0][2])
    grid_shape = first_alpha.shape
    print(f"Grid: {grid_shape}")
    print(f"WATNEEDS mask: {wn.sum()} pixels")

    # Accumulate weighted sum and total weight per pixel
    alpha_sum = np.zeros(grid_shape, dtype=np.float64)
    weight_sum = np.zeros(grid_shape, dtype=np.float64)

    print(f"\n{'Scenario':<25s} {'Alpha px':>9s} {'Area px':>9s} {'Frac':>6s} {'Ha total':>10s} {'Contrib px':>11s}")
    print("-" * 75)

    for crop, irr, alpha_path, area_path, frac in SCENARIOS:
        # Load alpha
        if not os.path.exists(alpha_path):
            print(f"  WARNING: {os.path.basename(alpha_path)} not found — skipping")
            continue
        alpha, _ = load_tif(alpha_path)
        alpha[alpha <= 0] = np.nan

        # Load area [ha] for this specific irrigation type
        if not os.path.exists(area_path):
            print(f"  WARNING: {os.path.basename(area_path)} not found — skipping")
            continue
        area_ha, _ = load_tif(area_path)
        area_ha[~np.isfinite(area_ha)] = 0.0
        area_ha[area_ha < 0] = 0.0

        # Weight = area_ha * fraction (e.g. 50% of irrigated ha for drip)
        ha_this = area_ha * frac

        # Only count pixels where both alpha and area are valid
        valid = ~np.isnan(alpha) & (ha_this > 0) & wn
        n_alpha = int(np.sum(~np.isnan(alpha) & wn))
        n_area = int(np.sum(area_ha > 0))
        n_contrib = int(valid.sum())
        ha_total = float(np.sum(ha_this[valid]))

        alpha_sum[valid] += alpha[valid] * ha_this[valid]
        weight_sum[valid] += ha_this[valid]

        label = f"{crop}_{irr}"
        print(f"{label:<25s} {n_alpha:>9d} {n_area:>9d} {frac:>6.2f} {ha_total:>10.0f} {n_contrib:>11d}")

    # Compute weighted mean
    alpha_weighted = np.full(grid_shape, np.nan, dtype=np.float32)
    has_weight = weight_sum > 0
    alpha_weighted[has_weight] = (alpha_sum[has_weight] / weight_sum[has_weight]).astype(np.float32)

    n_valid = int(np.sum(~np.isnan(alpha_weighted)))
    print(f"\n{'='*75}")
    print(f"Weighted alpha map: {n_valid} valid pixels")
    a = alpha_weighted[~np.isnan(alpha_weighted)]
    if len(a) > 0:
        print(f"  Mean:   {np.mean(a):.3f}")
        print(f"  Median: {np.median(a):.3f}")
        print(f"  Std:    {np.std(a):.3f}")
        print(f"  Range:  [{np.min(a):.3f}, {np.max(a):.3f}]")

    # Save as GeoTIFF
    out_path = os.path.join(OUT_DIR, "weighted_alpha_map.tif")
    profile = {
        'driver': 'GTiff', 'height': grid_shape[0], 'width': grid_shape[1],
        'count': 1, 'dtype': np.float32, 'crs': geo['crs'],
        'transform': geo['transform'], 'compress': 'lzw', 'nodata': np.nan,
    }
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(alpha_weighted, 1)
    print(f"\nSaved: {out_path}")

    # Also save weight map (total ha per pixel — useful for CDR scaling)
    weight_path = os.path.join(OUT_DIR, "weighted_alpha_total_ha.tif")
    weight_map = weight_sum.astype(np.float32)
    weight_map[weight_map == 0] = np.nan
    with rasterio.open(weight_path, 'w', **profile) as dst:
        dst.write(weight_map, 1)
    print(f"Saved: {weight_path}")

    # ── PLOT ──────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Weighted Alpha Map (area-weighted across all crop × irrigation scenarios)",
                 fontsize=13, fontweight='bold')

    # Panel 1: weighted alpha
    a_display = alpha_weighted.copy()
    a_display[~wn] = np.nan
    im0 = axes[0].imshow(a_display, cmap='plasma', origin='upper', aspect='auto')
    fig.colorbar(im0, ax=axes[0], shrink=0.75, label='Alpha')
    axes[0].set_title(f'Weighted Alpha ({n_valid} px)')

    # Panel 2: total cultivated ha per pixel
    w_display = weight_sum.copy()
    w_display[w_display == 0] = np.nan
    w_display[~wn] = np.nan
    im1 = axes[1].imshow(w_display, cmap='YlGn', origin='upper', aspect='auto')
    fig.colorbar(im1, ax=axes[1], shrink=0.75, label='Total ha')
    axes[1].set_title('Total cultivated area [ha]')

    # Panel 3: number of scenarios contributing per pixel
    n_scenarios = np.zeros(grid_shape, dtype=int)
    for crop, irr, alpha_path, area_path, frac in SCENARIOS:
        if not os.path.exists(alpha_path) or not os.path.exists(area_path):
            continue
        alpha, _ = load_tif(alpha_path)
        alpha[alpha <= 0] = np.nan
        area_ha, _ = load_tif(area_path)
        area_ha[~np.isfinite(area_ha)] = 0.0
        valid = ~np.isnan(alpha) & (area_ha * frac > 0) & wn
        n_scenarios[valid] += 1

    n_display = n_scenarios.astype(float)
    n_display[n_display == 0] = np.nan
    n_display[~wn] = np.nan
    im2 = axes[2].imshow(n_display, cmap='viridis', vmin=1, vmax=5,
                          origin='upper', aspect='auto')
    fig.colorbar(im2, ax=axes[2], shrink=0.75, label='N scenarios')
    axes[2].set_title('Scenarios contributing per pixel')

    for ax in axes:
        ax.set_xlabel('Col')
        ax.set_ylabel('Row')

    plt.tight_layout()
    out_png = os.path.join(DL, "weighted_alpha_map.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_png}")
    plt.close()


if __name__ == "__main__":
    main()

"""
EW Simulation Diagnostics:
  1. Seasonal mineral weathering flux (DOY climatology, mean + IQR) — like the reference figure
  2. Per-pixel time series: Ca, Mg, CaCO3 evolution for 3 random cultivated pixels

Run on cluster:
  python plot_ew_diagnostics.py --base-dir /scratch/user/lorenzo32/WATNEEDS+SMEW
  python plot_ew_diagnostics.py --crop olivo --irr drip
  python plot_ew_diagnostics.py --all
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio

# ── CONFIG ────────────────────────────────────────────────────────────────────
SIM_YEARS = 30
N_DAYS = SIM_YEARS * 365
conv_mol = 1e6

ALL_SCENARIOS = [
    ('olivo', 'drip'), ('olivo', 'traditional'), ('olivo', 'rainfed'),
    ('vite', 'drip'), ('vite', 'traditional'), ('vite', 'rainfed'),
    ('agrumi', 'drip'), ('agrumi', 'traditional'),
    ('pesco', 'drip'), ('pesco', 'traditional'),
    ('grano', 'rainfed'),
]

MINERALS = ['labradorite', 'albite', 'diopside', 'anorthite']
MINERAL_COLORS = {
    'labradorite': '#d62728', 'albite': '#9467bd',
    'diopside': '#2ca02c', 'anorthite': '#ff7f0e',
}

CROP_AREA_FILES = {
    'olivo': 'sicily10km_olives_total_ha.tif',
    'vite': 'sicily10km_vineyard_total_ha.tif',
    'agrumi': 'sicily10km_citrus_total_ha.tif',
    'pesco': 'sicily10km_fruits_total_ha.tif',
    'grano': 'sicily10km_wheat_total_ha.tif',
}

MONTH_STARTS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
MONTH_LABELS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']


def load_npy(rdir, var, scenario):
    for pattern in [f'{var}_sic_{scenario}_daily.npy', f'{var}_{scenario}_daily.npy']:
        fpath = os.path.join(rdir, pattern)
        if os.path.exists(fpath):
            return np.load(fpath)
    return None


def smooth(x, w=7):
    """Optional running mean. w=1 returns raw data."""
    if w <= 1:
        return x
    import pandas as pd
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().values


def get_valid_pixels(rdir, crop, area_dir):
    """Get mask of pixels that have results AND are cultivated."""
    ph = load_npy(rdir, 'pH', 'basalt')
    if ph is None:
        return None, None
    has_result = ~np.all(np.isnan(ph), axis=2)

    area_file = os.path.join(area_dir, CROP_AREA_FILES.get(crop, ''))
    if os.path.exists(area_file):
        with rasterio.open(area_file) as src:
            ha = src.read(1).astype(float)
        ha[~np.isfinite(ha)] = 0
        cultivated = ha > 0
    else:
        cultivated = has_result

    return has_result & cultivated, ha


def plot_mineral_seasonal(rdir, crop, irr, area_dir, out_dir):
    """
    Figure 1: Seasonal evolution of per-mineral weathering flux + total DIC effect.
    DOY climatology (mean + 25-75% IQR) across all pixels and years.
    """
    valid_mask, ha = get_valid_pixels(rdir, crop, area_dir)
    if valid_mask is None or not valid_mask.any():
        print(f"  No valid pixels for {crop}/{irr}")
        return

    n_years = N_DAYS // 365

    fig, ax = plt.subplots(figsize=(12, 6))

    # Per-mineral weathering flux
    for mname in MINERALS:
        ew_arr = load_npy(rdir, f'EW_{mname}', 'basalt')
        if ew_arr is None:
            continue

        # Spatial mean per day (across valid pixels)
        daily_spatial_mean = np.nanmean(ew_arr[valid_mask], axis=0)  # (n_days,)
        if len(daily_spatial_mean) < n_years * 365:
            continue

        # Reshape to (n_years, 365), compute DOY stats
        ts = daily_spatial_mean[:n_years * 365].reshape(n_years, 365)
        doy_mean = smooth(np.nanmean(ts, axis=0))
        doy_q25 = smooth(np.nanpercentile(ts, 25, axis=0))
        doy_q75 = smooth(np.nanpercentile(ts, 75, axis=0))

        doy = np.arange(365)
        color = MINERAL_COLORS[mname]
        ax.fill_between(doy, doy_q25, doy_q75, color=color, alpha=0.15)
        ax.plot(doy, doy_mean, color=color, lw=2, label=mname.capitalize())

    # Total effective CDR: DIC(EW) - DIC(noEW) as dashed line
    dic_ew = load_npy(rdir, 'DIC', 'basalt')
    dic_noew = load_npy(rdir, 'DIC', 'noEW')
    if dic_ew is not None and dic_noew is not None:
        delta_dic_spatial = np.nanmean(dic_ew[valid_mask], axis=0) - \
                            np.nanmean(dic_noew[valid_mask], axis=0)
        ts_dic = delta_dic_spatial[:n_years * 365].reshape(n_years, 365)
        dic_mean = smooth(np.nanmean(ts_dic, axis=0))
        dic_q25 = smooth(np.nanpercentile(ts_dic, 25, axis=0))
        dic_q75 = smooth(np.nanpercentile(ts_dic, 75, axis=0))

        ax.fill_between(np.arange(365), dic_q25, dic_q75, color='steelblue', alpha=0.1)
        ax.plot(np.arange(365), dic_mean, color='steelblue', lw=2, ls='--',
                label='Total DIC increase (EW−noEW)')

    ax.set_xticks(MONTH_STARTS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=11)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Weathering flux / DIC difference (spatial mean)', fontsize=11)
    ax.set_title(f'Seasonal Mineral Weathering vs Total DIC Effect\n'
                 f'{crop.capitalize()} {irr.capitalize()} — Basalt (Mt. Etna)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlim(0, 364)

    outpath = os.path.join(out_dir, f'mineral_seasonal_{crop}_{irr}.png')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_pixel_timeseries(rdir, crop, irr, area_dir, out_dir, n_pixels=3):
    """
    Figure 2: Per-pixel time series of Ca, Mg, CaCO3 for random cultivated pixels.
    Shows noEW (dashed) vs basalt (solid) for each variable.
    """
    valid_mask, ha = get_valid_pixels(rdir, crop, area_dir)
    if valid_mask is None or not valid_mask.any():
        return

    # Pick n_pixels random cultivated pixels
    rows, cols = np.where(valid_mask)
    np.random.seed(42)
    if len(rows) > n_pixels:
        idx = np.random.choice(len(rows), n_pixels, replace=False)
    else:
        idx = np.arange(len(rows))

    # Load variables
    vars_to_plot = [
        ('Ca', 'Ca', 'umol/L'),
        ('Mg', 'Mg', 'umol/L'),
        ('CaCO3', 'CaCO3', 'mol-conv/m2'),
        ('pH', 'pH', ''),
    ]

    for px_idx in idx:
        r, c = rows[px_idx], cols[px_idx]
        ha_val = ha[r, c] if ha is not None else 0

        fig, axes = plt.subplots(len(vars_to_plot), 1, figsize=(14, 3.5 * len(vars_to_plot)), sharex=True)
        fig.suptitle(f'{crop.capitalize()} {irr.capitalize()} — Pixel ({r},{c})  '
                     f'[{ha_val:.0f} ha]',
                     fontsize=13, fontweight='bold')

        t_years = np.arange(N_DAYS) / 365.0

        for ax, (var, label, unit) in zip(axes, vars_to_plot):
            arr_ew = load_npy(rdir, var, 'basalt')
            arr_noew = load_npy(rdir, var, 'noEW')

            if arr_ew is not None:
                ts_ew = arr_ew[r, c, :N_DAYS]
                ts_ew_smooth = ts_ew
                ax.plot(t_years[:len(ts_ew_smooth)], ts_ew_smooth,
                        color='steelblue', lw=1.2, label=f'{label} (basalt)')

            if arr_noew is not None:
                ts_noew = arr_noew[r, c, :N_DAYS]
                ts_noew_smooth = ts_noew
                ax.plot(t_years[:len(ts_noew_smooth)], ts_noew_smooth,
                        color='gray', lw=1, ls='--', alpha=0.7, label=f'{label} (noEW)')

            # Mark rock application years
            for app_yr in [0, 10, 20]:
                ax.axvline(app_yr, color='red', ls=':', alpha=0.3, lw=0.8)

            ax.set_ylabel(f'{label} [{unit}]' if unit else label, fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Year (EW period)', fontsize=11)
        axes[0].text(0.5, 1.02, 'Red dashed = rock application', transform=axes[0].transAxes,
                     fontsize=8, ha='center', color='red', alpha=0.6)

        plt.tight_layout()
        outpath = os.path.join(out_dir, f'pixel_ts_{crop}_{irr}_{r}_{c}.png')
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outpath}")


def process_scenario(base, crop, irr, area_dir, out_dir):
    rdir = os.path.join(base, 'Results', f'{crop}_{irr}')
    if not os.path.isdir(rdir):
        print(f"\n  {crop}/{irr}: Results not found, skipping")
        return

    print(f"\n{'='*60}")
    print(f"  {crop.upper()} / {irr.upper()}")
    print(f"{'='*60}")

    plot_mineral_seasonal(rdir, crop, irr, area_dir, out_dir)
    plot_pixel_timeseries(rdir, crop, irr, area_dir, out_dir)


def main():
    parser = argparse.ArgumentParser(description='EW simulation diagnostic plots')
    parser.add_argument('--base-dir', type=str,
                        default='/scratch/user/lorenzo32/WATNEEDS+SMEW')
    parser.add_argument('--crop', type=str, default=None)
    parser.add_argument('--irr', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    base = args.base_dir
    area_dir = os.path.join(base, 'aree_coltivate')
    out_dir = os.path.join(base, 'Results')
    os.makedirs(out_dir, exist_ok=True)

    if args.all:
        for crop, irr in ALL_SCENARIOS:
            process_scenario(base, crop, irr, area_dir, out_dir)
    elif args.crop and args.irr:
        process_scenario(base, args.crop, args.irr, area_dir, out_dir)
    else:
        # Default: all olivo scenarios
        for irr in ['drip', 'traditional', 'rainfed']:
            process_scenario(base, 'olivo', irr, area_dir, out_dir)

    print(f"\nDone.")


if __name__ == '__main__':
    main()

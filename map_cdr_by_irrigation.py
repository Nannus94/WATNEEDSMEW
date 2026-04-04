"""
CDR maps aggregated by irrigation type across ALL crops.

For each pixel:
  CDR_irrigated = sum over crops [ CDR_crop_drip * ha_irrig/2 + CDR_crop_trad * ha_irrig/2 ]
  CDR_rainfed   = sum over crops [ CDR_crop_rainfed * ha_rain ]
  CDR_total     = CDR_irrigated + CDR_rainfed

Also shows % contribution of each crop × irrigation scenario.

Run on cluster:
  python map_cdr_by_irrigation.py --base-dir /scratch/user/lorenzo32/WATNEEDS+SMEW
"""
import os
import sys
import argparse
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio

# ── CONFIG ────────────────────────────────────────────────────────────────────
SIM_YEARS = 30
N_DAYS = SIM_YEARS * 365
conv_mol = 1e6
MM_CO2 = 44.01
IRR_SPLIT = 0.5  # 50% drip, 50% traditional for irrigated area

# Scenario definitions: (crop, irrigation, area_type)
# area_type: 'irrig' uses _i_ha * IRR_SPLIT, 'rain' uses _r_ha
SCENARIOS = [
    ('olivo',  'drip',        'irrig'),
    ('olivo',  'traditional', 'irrig'),
    ('olivo',  'rainfed',     'rain'),
    ('vite',   'drip',        'irrig'),
    ('vite',   'traditional', 'irrig'),
    ('vite',   'rainfed',     'rain'),
    ('agrumi', 'drip',        'irrig'),
    ('agrumi', 'traditional', 'irrig'),
    ('agrumi', 'rainfed',     'rain'),
    ('pesco',  'drip',        'irrig'),
    ('pesco',  'traditional', 'irrig'),
    ('pesco',  'rainfed',     'rain'),
    ('grano',  'rainfed',     'rain'),
]

CROP_AREA = {
    'olivo':  {'irrig': 'sicily10km_olives_i_ha.tif',   'rain': 'sicily10km_olives_r_ha.tif',   'total': 'sicily10km_olives_total_ha.tif'},
    'vite':   {'irrig': 'sicily10km_vineyard_i_ha.tif',  'rain': 'sicily10km_vineyard_r_ha.tif',  'total': 'sicily10km_vineyard_total_ha.tif'},
    'agrumi': {'irrig': 'sicily10km_citrus_i_ha.tif',    'rain': 'sicily10km_citrus_r_ha.tif',    'total': 'sicily10km_citrus_total_ha.tif'},
    'pesco':  {'irrig': 'sicily10km_fruits_i_ha.tif',    'rain': 'sicily10km_fruits_r_ha.tif',    'total': 'sicily10km_fruits_total_ha.tif'},
    'grano':  {'irrig': None,                             'rain': 'sicily10km_wheat_r_ha.tif',     'total': 'sicily10km_wheat_total_ha.tif'},
}

COLORS_CROP = {
    'olivo': '#8c564b', 'vite': '#9467bd', 'agrumi': '#ff7f0e',
    'pesco': '#e377c2', 'grano': '#bcbd22',
}


def load_npy(rdir, var, scenario):
    for pattern in [f'{var}_sic_{scenario}_daily.npy', f'{var}_{scenario}_daily.npy']:
        fpath = os.path.join(rdir, pattern)
        if os.path.exists(fpath):
            return np.load(fpath)
    return None


def load_tif(fpath):
    if fpath is None or not os.path.exists(fpath):
        return None
    with rasterio.open(fpath) as src:
        d = src.read(1).astype(np.float64)
    d[~np.isfinite(d)] = 0.0
    d[d < 0] = 0.0
    return d


def load_hydro_L(base, crop, irr, years=30):
    irr_dir = 'surface' if irr in ('traditional', 'trad') else irr
    hydro_dir = os.path.join(base, 'WB_interpolated_first4hours', f'{crop}_{irr_dir}')
    if not os.path.isdir(hydro_dir):
        return None
    L_list = []
    mat_files = sorted([f for f in os.listdir(hydro_dir) if f.startswith('shallow_L_')])
    available_years = sorted({int(f.split('_')[2]) for f in mat_files})
    for year in available_years[-years:]:
        for month in range(1, 13):
            fpath = os.path.join(hydro_dir, f'shallow_L_{year}_{month}.mat')
            if not os.path.exists(fpath):
                continue
            mat = scipy.io.loadmat(fpath)
            key = [k for k in mat if not k.startswith('_')][0]
            L_list.append(mat[key].astype(np.float32))
    if not L_list:
        return None
    L_full = np.concatenate(L_list, axis=2) / 1000.0 * 6.0
    spd = 6
    nd = L_full.shape[2] // spd
    L_daily = np.mean(L_full[:, :, :nd * spd].reshape(L_full.shape[0], L_full.shape[1], nd, spd), axis=3)
    if L_daily.shape[2] < N_DAYS:
        nt = int(np.ceil(N_DAYS / L_daily.shape[2]))
        L_daily = np.tile(L_daily, (1, 1, nt))[:, :, :N_DAYS]
    else:
        L_daily = L_daily[:, :, :N_DAYS]
    return L_daily


def compute_cdr_per_ha(base, crop, irr):
    """Returns CDR total [t CO2/ha/yr] per pixel, or None."""
    rdir = os.path.join(base, 'Results', f'{crop}_{irr}')
    if not os.path.isdir(rdir):
        return None

    dic_ew = load_npy(rdir, 'DIC', 'basalt')
    dic_noew = load_npy(rdir, 'DIC', 'noEW')
    caco3_ew = load_npy(rdir, 'CaCO3', 'basalt')
    caco3_noew = load_npy(rdir, 'CaCO3', 'noEW')
    mgco3_ew = load_npy(rdir, 'MgCO3', 'basalt')
    mgco3_noew = load_npy(rdir, 'MgCO3', 'noEW')

    if dic_ew is None or dic_noew is None:
        return None

    rows, cols = dic_ew.shape[:2]
    L_daily = load_hydro_L(base, crop, irr)

    # DIC leaching CDR
    cdr_dic = np.full((rows, cols), np.nan, dtype=np.float64)
    if L_daily is not None:
        delta_dic = dic_ew - dic_noew
        for i in range(rows):
            for j in range(cols):
                if np.isnan(dic_ew[i, j, 0]):
                    continue
                ddic = delta_dic[i, j, :]
                L_px = L_daily[i, j, :len(ddic)]
                cum = np.nansum(ddic * L_px * 1000.0)
                cdr_dic[i, j] = cum / conv_mol * MM_CO2 / 1e6 * 1e4 / SIM_YEARS

    # Pedogenic carbonate CDR
    cdr_carb = np.full((rows, cols), np.nan, dtype=np.float64)
    if caco3_ew is not None and caco3_noew is not None:
        ly = slice(-365, None)
        fy = slice(0, 365)
        d_caco3 = (np.nanmean(caco3_ew[:, :, ly], axis=2) - np.nanmean(caco3_ew[:, :, fy], axis=2)) - \
                  (np.nanmean(caco3_noew[:, :, ly], axis=2) - np.nanmean(caco3_noew[:, :, fy], axis=2))
        d_mgco3 = np.zeros_like(d_caco3)
        if mgco3_ew is not None and mgco3_noew is not None:
            d_mgco3 = (np.nanmean(mgco3_ew[:, :, ly], axis=2) - np.nanmean(mgco3_ew[:, :, fy], axis=2)) - \
                      (np.nanmean(mgco3_noew[:, :, ly], axis=2) - np.nanmean(mgco3_noew[:, :, fy], axis=2))
        cdr_carb = (d_caco3 + d_mgco3) / conv_mol * MM_CO2 / 1e6 * 1e4 / SIM_YEARS

    # Total
    cdr = np.where(np.isnan(cdr_dic) & np.isnan(cdr_carb), np.nan,
                   np.nansum([np.nan_to_num(cdr_dic), np.nan_to_num(cdr_carb)], axis=0))
    return cdr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default='/scratch/user/lorenzo32/WATNEEDS+SMEW')
    args = parser.parse_args()
    base = args.base_dir
    area_dir = os.path.join(base, 'aree_coltivate')
    out_dir = os.path.join(base, 'Results')

    grid_shape = (39, 43)

    # ── Compute CDR per scenario and accumulate by irrigation type ──
    # Maps: CDR [t CO2/yr] per pixel (CDR_per_ha * ha)
    cdr_irrigated = np.zeros(grid_shape, dtype=np.float64)
    cdr_rainfed = np.zeros(grid_shape, dtype=np.float64)
    ha_irrigated = np.zeros(grid_shape, dtype=np.float64)
    ha_rainfed = np.zeros(grid_shape, dtype=np.float64)

    # Per-scenario contribution tracking
    contributions = {}  # (crop, irr) → regional CDR [t CO2/yr]

    print(f"{'Crop':>10s} {'Irrigation':>14s}  {'CDR/ha/yr':>10s} {'Ha':>10s} {'Regional':>12s}  Status")
    print('-' * 70)

    for crop, irr, area_type in SCENARIOS:
        # Load CDR per ha
        cdr = compute_cdr_per_ha(base, crop, irr)
        if cdr is None:
            print(f"{crop:>10s} {irr:>14s}  {'':>10s} {'':>10s} {'':>12s}  NO RESULTS")
            contributions[(crop, irr)] = 0.0
            continue

        # Load area
        area_file = CROP_AREA[crop].get(area_type)
        ha_map = load_tif(os.path.join(area_dir, area_file)) if area_file else None
        if ha_map is None:
            print(f"{crop:>10s} {irr:>14s}  {'':>10s} {'':>10s} {'':>12s}  NO AREA MAP")
            contributions[(crop, irr)] = 0.0
            continue

        # Scale irrigated area by split fraction
        if area_type == 'irrig':
            ha_map = ha_map * IRR_SPLIT

        valid = ~np.isnan(cdr) & (ha_map > 0)
        if not valid.any():
            contributions[(crop, irr)] = 0.0
            continue

        # CDR in t CO2/yr per pixel
        cdr_pixel = np.where(valid, cdr * ha_map, 0.0)
        regional = np.sum(cdr_pixel)
        contributions[(crop, irr)] = regional

        # Accumulate by irrigation type
        if irr == 'rainfed':
            cdr_rainfed += cdr_pixel
            ha_rainfed += np.where(valid, ha_map, 0.0)
        else:
            cdr_irrigated += cdr_pixel
            ha_irrigated += np.where(valid, ha_map, 0.0)

        mean_cdr = np.mean(cdr[valid])
        total_ha = np.sum(ha_map[valid])
        print(f"{crop:>10s} {irr:>14s}  {mean_cdr:>10.3f} {total_ha:>10.0f} {regional:>12.1f}  OK")

    cdr_total = cdr_irrigated + cdr_rainfed
    ha_total = ha_irrigated + ha_rainfed

    # ── Summary ──
    total_irr = np.sum(cdr_irrigated)
    total_rain = np.sum(cdr_rainfed)
    total_all = total_irr + total_rain

    print(f"\n{'='*70}")
    print(f"REGIONAL CDR SUMMARY (t CO2/yr)")
    print(f"{'='*70}")
    print(f"  Irrigated (drip + traditional): {total_irr:>12.1f}")
    print(f"  Rainfed:                        {total_rain:>12.1f}")
    print(f"  TOTAL:                          {total_all:>12.1f}")

    # ── % contribution per scenario ──
    print(f"\n--- Contribution by scenario ---")
    print(f"{'Crop':>10s} {'Irrigation':>14s}  {'t CO2/yr':>12s}  {'% of total':>10s}")
    print('-' * 50)
    for (crop, irr), val in sorted(contributions.items(), key=lambda x: -x[1]):
        pct = 100 * val / total_all if total_all > 0 else 0
        if val > 0:
            print(f"{crop:>10s} {irr:>14s}  {val:>12.1f}  {pct:>9.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Regional CDR by Irrigation Type — Sicily, Mt. Etna Basalt\n'
                 f'Total: {total_all:.0f} t CO2/yr  |  Irrigated: {total_irr:.0f}  |  Rainfed: {total_rain:.0f}',
                 fontsize=14, fontweight='bold')

    # Shared vmax for CDR maps
    vmax_cdr = np.percentile(cdr_total[cdr_total > 0], 95) if np.any(cdr_total > 0) else 1

    # Panel 1: CDR irrigated [t CO2/yr per pixel]
    ax = fig.add_subplot(gs[0, 0])
    display = np.where(cdr_irrigated > 0, cdr_irrigated, np.nan)
    im = ax.imshow(display, cmap='OrRd', vmin=0, vmax=vmax_cdr, origin='upper', aspect='auto')
    fig.colorbar(im, ax=ax, shrink=0.75, label='t CO2/yr')
    ax.set_title(f'(a) CDR Irrigated\n(drip + traditional, {total_irr:.0f} t/yr)')

    # Panel 2: CDR rainfed
    ax = fig.add_subplot(gs[0, 1])
    display = np.where(cdr_rainfed > 0, cdr_rainfed, np.nan)
    im = ax.imshow(display, cmap='YlGn', vmin=0, vmax=vmax_cdr, origin='upper', aspect='auto')
    fig.colorbar(im, ax=ax, shrink=0.75, label='t CO2/yr')
    ax.set_title(f'(b) CDR Rainfed\n({total_rain:.0f} t/yr)')

    # Panel 3: CDR total
    ax = fig.add_subplot(gs[0, 2])
    display = np.where(cdr_total > 0, cdr_total, np.nan)
    im = ax.imshow(display, cmap='YlOrRd', vmin=0, vmax=vmax_cdr, origin='upper', aspect='auto')
    fig.colorbar(im, ax=ax, shrink=0.75, label='t CO2/yr')
    ax.set_title(f'(c) CDR Total\n({total_all:.0f} t/yr)')

    # Panel 4: Pie chart — % contribution by crop
    ax = fig.add_subplot(gs[1, 0])
    crop_totals = {}
    for (crop, irr), val in contributions.items():
        crop_totals[crop] = crop_totals.get(crop, 0) + val
    crop_totals = {k: v for k, v in crop_totals.items() if v > 0}
    if crop_totals:
        labels = [f'{c.capitalize()}\n{v:.0f} t/yr' for c, v in crop_totals.items()]
        colors = [COLORS_CROP.get(c, 'gray') for c in crop_totals.keys()]
        ax.pie(crop_totals.values(), labels=labels, colors=colors, autopct='%1.1f%%',
               textprops={'fontsize': 9})
        ax.set_title('(d) CDR by Crop')

    # Panel 5: Stacked bar — irrigated vs rainfed per crop
    ax = fig.add_subplot(gs[1, 1])
    crops = sorted(set(c for c, i in contributions.keys() if contributions[(c, i)] > 0))
    irr_vals = [sum(contributions.get((c, i), 0) for i in ['drip', 'traditional']) for c in crops]
    rain_vals = [contributions.get((c, 'rainfed'), 0) for c in crops]

    x = np.arange(len(crops))
    ax.bar(x, irr_vals, color='#d62728', alpha=0.7, label='Irrigated', edgecolor='black', linewidth=0.5)
    ax.bar(x, rain_vals, bottom=irr_vals, color='#1f77b4', alpha=0.7, label='Rainfed', edgecolor='black', linewidth=0.5)
    for j in range(len(crops)):
        total_j = irr_vals[j] + rain_vals[j]
        if total_j > 0:
            ax.text(j, total_j + 200, f'{total_j:.0f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in crops])
    ax.set_ylabel('CDR (t CO2/yr)')
    ax.legend(fontsize=9)
    ax.set_title('(e) CDR by Crop: Irrigated vs Rainfed')
    ax.grid(axis='y', alpha=0.3)

    # Panel 6: % contribution table as text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    sorted_contribs = sorted(contributions.items(), key=lambda x: -x[1])
    lines = ['Scenario              t CO2/yr    %\n' + '-' * 40]
    for (crop, irr), val in sorted_contribs:
        if val > 0:
            pct = 100 * val / total_all if total_all > 0 else 0
            lines.append(f'{crop:>8s} {irr:>12s}  {val:>9.0f}  {pct:>5.1f}%')
    lines.append('-' * 40)
    lines.append(f'{"TOTAL":>21s}  {total_all:>9.0f}  100.0%')
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('(f) Scenario Contributions')

    plt.tight_layout()
    outpath = os.path.join(out_dir, 'cdr_map_by_irrigation_type.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == '__main__':
    main()

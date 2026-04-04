"""
Compare CDR across ALL crop × irrigation scenarios.
Shows both:
  - Unweighted CDR (t CO2/ha/yr per pixel, averaged across valid pixels)
  - Weighted CDR (t CO2/yr, using actual cultivated ha per pixel)

Run on cluster:
  python compare_all_cdr.py --base-dir /scratch/user/lorenzo32/WATNEEDS+SMEW
"""
import os
import sys
import argparse
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio

# ── CONFIG ────────────────────────────────────────────────────────────────────
SIM_YEARS = 30
N_DAYS = SIM_YEARS * 365
Zr = 0.3
conv_mol = 1e6
MM_CO2 = 44.01

# All crop × irrigation scenarios
ALL_SCENARIOS = [
    ('olivo',  'drip'),
    ('olivo',  'traditional'),
    ('olivo',  'rainfed'),
    ('vite',   'drip'),
    ('vite',   'traditional'),
    ('vite',   'rainfed'),
    ('agrumi', 'drip'),
    ('agrumi', 'traditional'),
    ('agrumi', 'rainfed'),
    ('pesco',  'drip'),
    ('pesco',  'traditional'),
    ('pesco',  'rainfed'),
    ('grano',  'rainfed'),
]

# Crop area map files (total ha for unweighted; _i_ha and _r_ha for weighted)
CROP_AREA_FILES = {
    'olivo':  {'total': 'sicily10km_olives_total_ha.tif',
               'irrig': 'sicily10km_olives_i_ha.tif',
               'rain':  'sicily10km_olives_r_ha.tif'},
    'vite':   {'total': 'sicily10km_vineyard_total_ha.tif',
               'irrig': 'sicily10km_vineyard_i_ha.tif',
               'rain':  'sicily10km_vineyard_r_ha.tif'},
    'agrumi': {'total': 'sicily10km_citrus_total_ha.tif',
               'irrig': 'sicily10km_citrus_i_ha.tif',
               'rain':  'sicily10km_citrus_r_ha.tif'},
    'pesco':  {'total': 'sicily10km_fruits_total_ha.tif',
               'irrig': 'sicily10km_fruits_i_ha.tif',
               'rain':  'sicily10km_fruits_r_ha.tif'},
    'grano':  {'total': 'sicily10km_wheat_total_ha.tif',
               'irrig': None,
               'rain':  'sicily10km_wheat_r_ha.tif'},
}

COLORS_IRR = {'drip': '#2ca02c', 'traditional': '#d62728', 'rainfed': '#1f77b4'}
COLORS_CROP = {'olivo': '#8c564b', 'vite': '#9467bd', 'agrumi': '#ff7f0e',
               'pesco': '#e377c2', 'grano': '#bcbd22'}

# Drip/traditional split for irrigated area (50/50 assumption)
IRR_SPLIT = 0.5


def load_npy(base, crop, irr, var, scenario):
    rdir = os.path.join(base, 'Results', f'{crop}_{irr}')
    for pattern in [f'{var}_sic_{scenario}_daily.npy', f'{var}_{scenario}_daily.npy']:
        fpath = os.path.join(rdir, pattern)
        if os.path.exists(fpath):
            return np.load(fpath)
    return None


def load_hydro_L(base, crop, irr, years=30):
    irr_dir = 'surface' if irr in ('traditional', 'trad') else irr
    hydro_dir = os.path.join(base, 'WB_interpolated_first4hours', f'{crop}_{irr_dir}')
    if not os.path.isdir(hydro_dir):
        return None

    mat_files = sorted([f for f in os.listdir(hydro_dir) if f.startswith('shallow_L_')])
    available_years = sorted({int(f.split('_')[2]) for f in mat_files})
    last_years = available_years[-years:]

    L_list = []
    for year in last_years:
        for month in range(1, 13):
            fpath = os.path.join(hydro_dir, f'shallow_L_{year}_{month}.mat')
            if not os.path.exists(fpath):
                continue
            mat = scipy.io.loadmat(fpath)
            key = [k for k in mat if not k.startswith('_')][0]
            L_list.append(mat[key].astype(np.float32))
    if not L_list:
        return None

    L_full = np.concatenate(L_list, axis=2) / 1000.0 * 6.0  # mm/4h → m/d
    spd = 6
    nd = L_full.shape[2] // spd
    L_daily = np.mean(L_full[:, :, :nd * spd].reshape(
        L_full.shape[0], L_full.shape[1], nd, spd), axis=3)

    if L_daily.shape[2] < N_DAYS:
        nt = int(np.ceil(N_DAYS / L_daily.shape[2]))
        L_daily = np.tile(L_daily, (1, 1, nt))[:, :, :N_DAYS]
    else:
        L_daily = L_daily[:, :, :N_DAYS]
    return L_daily


def load_area_tif(fpath):
    if fpath is None or not os.path.exists(fpath):
        return None
    with rasterio.open(fpath) as src:
        d = src.read(1).astype(np.float64)
    d[~np.isfinite(d)] = 0.0
    d[d < 0] = 0.0
    return d


def compute_cdr(base, crop, irr):
    """Compute CDR per pixel [t CO2/ha/yr] for one scenario."""
    rdir = os.path.join(base, 'Results', f'{crop}_{irr}')
    if not os.path.isdir(rdir):
        return None

    dic_ew = load_npy(base, crop, irr, 'DIC', 'basalt')
    dic_noew = load_npy(base, crop, irr, 'DIC', 'noEW')
    caco3_ew = load_npy(base, crop, irr, 'CaCO3', 'basalt')
    caco3_noew = load_npy(base, crop, irr, 'CaCO3', 'noEW')
    mgco3_ew = load_npy(base, crop, irr, 'MgCO3', 'basalt')
    mgco3_noew = load_npy(base, crop, irr, 'MgCO3', 'noEW')
    m_rock = load_npy(base, crop, irr, 'M_rock', 'basalt')

    if dic_ew is None or dic_noew is None:
        return None

    rows, cols = dic_ew.shape[:2]
    L_daily = load_hydro_L(base, crop, irr)

    # DIC leaching CDR
    cdr_dic = np.full((rows, cols), np.nan, dtype=np.float32)
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
    cdr_carb = np.full((rows, cols), np.nan, dtype=np.float32)
    if caco3_ew is not None and caco3_noew is not None:
        last_yr = slice(-365, None)
        first_yr = slice(0, 365)
        d_caco3 = (np.nanmean(caco3_ew[:, :, last_yr], axis=2) -
                   np.nanmean(caco3_ew[:, :, first_yr], axis=2)) - \
                  (np.nanmean(caco3_noew[:, :, last_yr], axis=2) -
                   np.nanmean(caco3_noew[:, :, first_yr], axis=2))
        d_mgco3 = np.zeros_like(d_caco3)
        if mgco3_ew is not None and mgco3_noew is not None:
            d_mgco3 = (np.nanmean(mgco3_ew[:, :, last_yr], axis=2) -
                       np.nanmean(mgco3_ew[:, :, first_yr], axis=2)) - \
                      (np.nanmean(mgco3_noew[:, :, last_yr], axis=2) -
                       np.nanmean(mgco3_noew[:, :, first_yr], axis=2))
        cdr_carb = (d_caco3 + d_mgco3) / conv_mol * MM_CO2 / 1e6 * 1e4 / SIM_YEARS

    # Total
    cdr_total = np.full((rows, cols), np.nan, dtype=np.float32)
    valid = ~np.isnan(cdr_dic) | ~np.isnan(cdr_carb)
    cdr_total[valid] = np.nansum([
        np.where(np.isnan(cdr_dic), 0, cdr_dic),
        np.where(np.isnan(cdr_carb), 0, cdr_carb)
    ], axis=0)[valid]

    # Rock dissolution %
    rock_pct = np.full((rows, cols), np.nan, dtype=np.float32)
    if m_rock is not None:
        rock_final = np.nanmean(m_rock[:, :, -365:], axis=2)
        rock_pct = (3 * 4000.0 - rock_final) / (3 * 4000.0) * 100

    return {
        'cdr_dic': cdr_dic, 'cdr_carb': cdr_carb,
        'cdr_total': cdr_total, 'rock_pct': rock_pct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str,
                        default='/scratch/user/lorenzo32/WATNEEDS+SMEW')
    args = parser.parse_args()
    base = args.base_dir
    area_dir = os.path.join(base, 'aree_coltivate')

    # ── Compute CDR for all available scenarios ──
    all_results = {}
    for crop, irr in ALL_SCENARIOS:
        rdir = os.path.join(base, 'Results', f'{crop}_{irr}')
        if not os.path.isdir(rdir):
            continue
        print(f"Processing: {crop}/{irr}...")
        res = compute_cdr(base, crop, irr)
        if res is not None:
            all_results[(crop, irr)] = res

    if not all_results:
        print("No results found!")
        sys.exit(1)

    # ── Load area maps ──
    area_maps = {}
    for crop, info in CROP_AREA_FILES.items():
        area_maps[crop] = {}
        for key in ['total', 'irrig', 'rain']:
            fname = info[key]
            if fname:
                area_maps[crop][key] = load_area_tif(os.path.join(area_dir, fname))

    # ── SUMMARY TABLES ──
    print(f"\n{'='*90}")
    print(f"CDR COMPARISON — ALL SCENARIOS")
    print(f"{'='*90}")

    # Load WATNEEDS ref mask for coverage check
    wn_path = os.path.join(area_dir, 'watneeds_ref_mask.tif')
    wn_mask = load_area_tif(wn_path)
    if wn_mask is not None:
        wn_mask = wn_mask > 0
    else:
        wn_mask = np.ones((39, 43), dtype=bool)

    # Table 1: Unweighted (per-ha, spatial mean across valid pixels) + coverage
    print(f"\n--- UNWEIGHTED CDR (spatial mean, t CO2/ha/yr) ---")
    print(f"{'Crop':>10s} {'Irrigation':>14s}  {'CDR_DIC':>10s} {'CDR_carb':>10s} {'CDR_total':>10s} {'Rock%':>8s} {'Valid':>7s} {'Cult.':>7s} {'Miss.':>7s}")
    print('-' * 90)

    for crop, irr in ALL_SCENARIOS:
        # Count cultivated pixels for this crop inside WATNEEDS mask
        total_ha = area_maps.get(crop, {}).get('total')
        n_cultivated = int(np.sum((total_ha > 0) & wn_mask)) if total_ha is not None else -1

        if (crop, irr) not in all_results:
            print(f"{crop:>10s} {irr:>14s}  {'NO RESULTS':>10s} {'':>10s} {'':>10s} {'':>8s} {'':>7s} {n_cultivated:>7d} {'ALL':>7s}")
            continue

        r = all_results[(crop, irr)]
        d = r['cdr_dic'][~np.isnan(r['cdr_dic'])]
        c = r['cdr_carb'][~np.isnan(r['cdr_carb'])]
        t = r['cdr_total'][~np.isnan(r['cdr_total'])]
        rk = r['rock_pct'][~np.isnan(r['rock_pct'])]
        n_valid = len(t)
        n_missing = n_cultivated - n_valid if n_cultivated >= 0 else -1
        miss_str = f"{n_missing:>7d}" if n_missing >= 0 else "   ?"
        flag = " ***" if n_missing > 0 else ""

        print(f"{crop:>10s} {irr:>14s}  {np.mean(d):>10.3f} {np.mean(c):>10.4f} {np.mean(t):>10.3f} "
              f"{np.mean(rk):>7.1f}% {n_valid:>7d} {n_cultivated:>7d} {miss_str}{flag}")

    # Detailed missing pixel report
    print(f"\n--- MISSING PIXEL DETAILS ---")
    any_missing = False
    for crop, irr in ALL_SCENARIOS:
        if (crop, irr) not in all_results:
            continue
        r = all_results[(crop, irr)]
        cdr = r['cdr_total']
        total_ha = area_maps.get(crop, {}).get('total')
        if total_ha is None:
            continue

        cultivated = (total_ha > 0) & wn_mask
        has_result = ~np.isnan(cdr)
        missing = cultivated & ~has_result

        if missing.any():
            any_missing = True
            rows_m, cols_m = np.where(missing)
            print(f"\n  {crop}/{irr}: {int(missing.sum())} cultivated pixels without CDR results:")
            for rm, cm in zip(rows_m[:10], cols_m[:10]):
                print(f"    [{rm:2d},{cm:2d}]  {total_ha[rm, cm]:.1f} ha")
            if len(rows_m) > 10:
                print(f"    ... and {len(rows_m) - 10} more")

    if not any_missing:
        print("  All cultivated pixels have CDR results for all available scenarios.")

    # Table 2: Weighted by cultivated area
    print(f"\n--- WEIGHTED CDR (using cultivated ha per pixel) ---")
    print(f"{'Crop':>10s} {'Irrigation':>14s}  {'CDR/ha/yr':>10s} {'Area (ha)':>12s} {'Regional CDR':>14s}")
    print(f"{'':>10s} {'':>14s}  {'t CO2':>10s} {'':>12s} {'t CO2/yr':>14s}")
    print('-' * 65)

    grand_total = 0.0

    for crop, irr in ALL_SCENARIOS:
        if (crop, irr) not in all_results:
            continue
        r = all_results[(crop, irr)]
        cdr = r['cdr_total']

        # Get the appropriate area map
        if irr == 'rainfed':
            ha_map = area_maps.get(crop, {}).get('rain')
        else:
            ha_map = area_maps.get(crop, {}).get('irrig')
            if ha_map is not None:
                ha_map = ha_map * IRR_SPLIT  # 50% drip, 50% traditional

        if ha_map is None:
            ha_map = area_maps.get(crop, {}).get('total')
            if ha_map is not None and irr != 'rainfed':
                ha_map = ha_map * 0.5  # rough split

        if ha_map is None:
            print(f"{crop:>10s} {irr:>14s}  {'no area map':>10s}")
            continue

        valid = ~np.isnan(cdr) & (ha_map > 0)
        if not valid.any():
            continue

        cdr_per_ha = np.mean(cdr[valid])
        total_ha = np.sum(ha_map[valid])
        regional = np.sum(cdr[valid] * ha_map[valid])
        grand_total += regional

        print(f"{crop:>10s} {irr:>14s}  {cdr_per_ha:>10.3f} {total_ha:>12.0f} {regional:>14.1f}")

    print(f"{'':>10s} {'':>14s}  {'':>10s} {'':>12s} {'─'*14}")
    print(f"{'':>10s} {'TOTAL':>14s}  {'':>10s} {'':>12s} {grand_total:>14.1f}")

    # ── PLOTS ──
    out_dir = os.path.join(base, 'Results')
    os.makedirs(out_dir, exist_ok=True)

    # ── Plot 1: Grouped bar chart — CDR per crop, grouped by irrigation ──
    crops_available = sorted(set(c for c, i in all_results.keys()))
    irrs_available = sorted(set(i for c, i in all_results.keys()),
                            key=lambda x: ['drip', 'traditional', 'rainfed'].index(x)
                            if x in ['drip', 'traditional', 'rainfed'] else 99)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('CDR Comparison Across All Scenarios — Mt. Etna Basalt (40 t/ha × 3)',
                 fontsize=13, fontweight='bold')

    # Panel 1: Unweighted (per ha)
    ax = axes[0]
    x = np.arange(len(crops_available))
    width = 0.25
    for i_irr, irr in enumerate(irrs_available):
        vals = []
        errs = []
        for crop in crops_available:
            if (crop, irr) in all_results:
                t = all_results[(crop, irr)]['cdr_total']
                v = t[~np.isnan(t)]
                vals.append(np.mean(v) if len(v) > 0 else 0)
                errs.append(np.std(v) if len(v) > 0 else 0)
            else:
                vals.append(0)
                errs.append(0)
        offset = (i_irr - len(irrs_available) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                      color=COLORS_IRR.get(irr, 'gray'), alpha=0.7,
                      label=irr.capitalize(), edgecolor='black', linewidth=0.5)
        for j, v in enumerate(vals):
            if v > 0:
                ax.text(x[j] + offset, v + errs[j] + 0.005, f'{v:.2f}',
                        ha='center', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in crops_available])
    ax.set_ylabel('CDR (t CO2/ha/yr)', fontsize=11)
    ax.set_title('(a) Unweighted CDR per ha')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Weighted (regional total)
    ax = axes[1]
    for i_irr, irr in enumerate(irrs_available):
        vals = []
        for crop in crops_available:
            if (crop, irr) in all_results:
                r = all_results[(crop, irr)]
                cdr = r['cdr_total']
                if irr == 'rainfed':
                    ha_map = area_maps.get(crop, {}).get('rain')
                else:
                    ha_map = area_maps.get(crop, {}).get('irrig')
                    if ha_map is not None:
                        ha_map = ha_map * IRR_SPLIT
                if ha_map is None:
                    ha_map = area_maps.get(crop, {}).get('total')
                    if ha_map is not None:
                        ha_map = ha_map * (1.0 if irr == 'rainfed' else 0.5)
                if ha_map is not None:
                    valid = ~np.isnan(cdr) & (ha_map > 0)
                    vals.append(np.sum(cdr[valid] * ha_map[valid]) if valid.any() else 0)
                else:
                    vals.append(0)
            else:
                vals.append(0)
        offset = (i_irr - len(irrs_available) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      color=COLORS_IRR.get(irr, 'gray'), alpha=0.7,
                      label=irr.capitalize(), edgecolor='black', linewidth=0.5)
        for j, v in enumerate(vals):
            if v > 0:
                ax.text(x[j] + offset, v + 500, f'{v:.0f}',
                        ha='center', fontsize=7, fontweight='bold', rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in crops_available])
    ax.set_ylabel('Regional CDR (t CO2/yr)', fontsize=11)
    ax.set_title('(b) Weighted CDR (× cultivated ha)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(out_dir, 'all_scenarios_cdr_comparison.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out1}")
    plt.close()

    # ── Plot 2: Stacked bar — total regional CDR by crop ──
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.suptitle('Total Regional CDR by Crop — Sicily, Mt. Etna Basalt',
                  fontsize=13, fontweight='bold')

    bottom = np.zeros(len(crops_available))
    for irr in irrs_available:
        vals = []
        for crop in crops_available:
            if (crop, irr) in all_results:
                r = all_results[(crop, irr)]
                cdr = r['cdr_total']
                if irr == 'rainfed':
                    ha_map = area_maps.get(crop, {}).get('rain')
                else:
                    ha_map = area_maps.get(crop, {}).get('irrig')
                    if ha_map is not None:
                        ha_map = ha_map * IRR_SPLIT
                if ha_map is None:
                    ha_map = area_maps.get(crop, {}).get('total')
                    if ha_map is not None:
                        ha_map = ha_map * (1.0 if irr == 'rainfed' else 0.5)
                if ha_map is not None:
                    valid = ~np.isnan(cdr) & (ha_map > 0)
                    vals.append(np.sum(cdr[valid] * ha_map[valid]) if valid.any() else 0)
                else:
                    vals.append(0)
            else:
                vals.append(0)

        ax2.bar(crops_available, vals, bottom=bottom,
                color=COLORS_IRR.get(irr, 'gray'), alpha=0.7,
                label=irr.capitalize(), edgecolor='black', linewidth=0.5)
        # Label each segment
        for j, v in enumerate(vals):
            if v > 500:
                ax2.text(j, bottom[j] + v / 2, f'{v:.0f}',
                         ha='center', va='center', fontsize=8, fontweight='bold')
        bottom += np.array(vals)

    # Total labels on top
    for j, b in enumerate(bottom):
        if b > 0:
            ax2.text(j, b + 500, f'Total: {b:.0f}', ha='center', fontsize=9, fontweight='bold')

    ax2.set_xticklabels([c.capitalize() for c in crops_available])
    ax2.set_ylabel('Regional CDR (t CO2/yr)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(out_dir, 'all_scenarios_cdr_stacked.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")
    plt.close()

    # ── Plot 3: Spatial CDR maps — all scenarios in a grid ──
    # Rows = crops, Columns = irrigation types
    all_crops = ['olivo', 'vite', 'agrumi', 'pesco', 'grano']
    all_irrs = ['drip', 'traditional', 'rainfed']

    # Find global vmin/vmax across all scenarios for consistent colorbar
    all_cdr_vals = []
    for (crop, irr), r in all_results.items():
        v = r['cdr_total'][~np.isnan(r['cdr_total'])]
        if len(v) > 0:
            all_cdr_vals.extend(v.tolist())
    if all_cdr_vals:
        vmax_cdr = np.percentile(all_cdr_vals, 95)
    else:
        vmax_cdr = 1.0

    fig3, axes3 = plt.subplots(len(all_crops), len(all_irrs),
                                figsize=(5 * len(all_irrs), 4 * len(all_crops)))
    fig3.suptitle('Spatial CDR (t CO2/ha/yr) — All Crop × Irrigation Scenarios\n'
                   'Basalt (Mt. Etna), 40 t/ha × 3 applications over 30 years',
                   fontsize=14, fontweight='bold', y=1.01)

    for i_crop, crop in enumerate(all_crops):
        for i_irr, irr in enumerate(all_irrs):
            ax = axes3[i_crop, i_irr]

            if (crop, irr) in all_results:
                cdr = all_results[(crop, irr)]['cdr_total'].copy()
                n_valid = int(np.sum(~np.isnan(cdr)))

                # Get cultivated mask for overlay
                total_ha = area_maps.get(crop, {}).get('total')
                if total_ha is not None:
                    cult_mask = total_ha > 0
                    # Grey out non-cultivated pixels
                    cdr_display = np.where(cult_mask, cdr, np.nan)
                else:
                    cdr_display = cdr

                im = ax.imshow(cdr_display, cmap='YlOrRd', vmin=0, vmax=vmax_cdr,
                               origin='upper', aspect='auto')

                # Mark missing cultivated pixels (cultivated but no CDR)
                if total_ha is not None:
                    missing = cult_mask & np.isnan(cdr)
                    if missing.any():
                        rm, cm = np.where(missing)
                        ax.scatter(cm, rm, s=15, c='black', marker='x', linewidths=0.8,
                                   alpha=0.7, zorder=5)

                mean_cdr = np.nanmean(cdr[~np.isnan(cdr)]) if n_valid > 0 else 0
                ax.set_title(f'{crop.capitalize()} {irr.capitalize()}\n'
                             f'n={n_valid}, mean={mean_cdr:.3f} t/ha/yr', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No results', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')
                ax.set_title(f'{crop.capitalize()} {irr.capitalize()}\n(not available)', fontsize=9)

            if i_crop == len(all_crops) - 1:
                ax.set_xlabel('Col', fontsize=8)
            if i_irr == 0:
                ax.set_ylabel('Row', fontsize=8)
            ax.tick_params(labelsize=7)

    # Shared colorbar
    fig3.subplots_adjust(right=0.92)
    cbar_ax = fig3.add_axes([0.94, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=vmax_cdr))
    fig3.colorbar(sm, cax=cbar_ax, label='CDR (t CO2/ha/yr)')

    plt.tight_layout(rect=[0, 0, 0.92, 0.97])
    out3 = os.path.join(out_dir, 'all_scenarios_cdr_spatial.png')
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"Saved: {out3}")
    plt.close()

    # ── Plot 4: Spatial rock dissolution % maps ──
    fig4, axes4 = plt.subplots(len(all_crops), len(all_irrs),
                                figsize=(5 * len(all_irrs), 4 * len(all_crops)))
    fig4.suptitle('Rock Dissolution (%) — All Crop × Irrigation Scenarios\n'
                   '30 years, 3 × 40 t/ha Mt. Etna Basalt',
                   fontsize=14, fontweight='bold', y=1.01)

    for i_crop, crop in enumerate(all_crops):
        for i_irr, irr in enumerate(all_irrs):
            ax = axes4[i_crop, i_irr]

            if (crop, irr) in all_results:
                rock = all_results[(crop, irr)]['rock_pct'].copy()
                n_valid = int(np.sum(~np.isnan(rock)))

                total_ha = area_maps.get(crop, {}).get('total')
                if total_ha is not None:
                    rock_display = np.where(total_ha > 0, rock, np.nan)
                else:
                    rock_display = rock

                im = ax.imshow(rock_display, cmap='Blues', vmin=0, vmax=100,
                               origin='upper', aspect='auto')

                mean_rock = np.nanmean(rock[~np.isnan(rock)]) if n_valid > 0 else 0
                ax.set_title(f'{crop.capitalize()} {irr.capitalize()}\n'
                             f'mean={mean_rock:.1f}% dissolved', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No results', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')
                ax.set_title(f'{crop.capitalize()} {irr.capitalize()}\n(not available)', fontsize=9)

            if i_crop == len(all_crops) - 1:
                ax.set_xlabel('Col', fontsize=8)
            if i_irr == 0:
                ax.set_ylabel('Row', fontsize=8)
            ax.tick_params(labelsize=7)

    fig4.subplots_adjust(right=0.92)
    cbar_ax4 = fig4.add_axes([0.94, 0.15, 0.015, 0.7])
    sm4 = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=100))
    fig4.colorbar(sm4, cax=cbar_ax4, label='Rock dissolved (%)')

    plt.tight_layout(rect=[0, 0, 0.92, 0.97])
    out4 = os.path.join(out_dir, 'all_scenarios_rock_dissolution_spatial.png')
    plt.savefig(out4, dpi=150, bbox_inches='tight')
    print(f"Saved: {out4}")
    plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 5: COMPOSITE CDR MAP — dominant crop per pixel + area-weighted
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- Building composite CDR maps ---")
    rows_g, cols_g = 39, 43
    all_crops_list = ['olivo', 'vite', 'agrumi', 'pesco', 'grano']

    # For each pixel, find dominant crop (most total ha) and its best CDR
    dominant_crop_idx = np.full((rows_g, cols_g), -1, dtype=int)
    dominant_crop_ha = np.zeros((rows_g, cols_g), dtype=np.float64)
    for ic, crop in enumerate(all_crops_list):
        total_ha = area_maps.get(crop, {}).get('total')
        if total_ha is None:
            continue
        ha = total_ha.copy()
        ha[~np.isfinite(ha)] = 0.0
        better = ha > dominant_crop_ha
        dominant_crop_idx[better] = ic
        dominant_crop_ha[better] = ha[better]

    # (a) Dominant-crop CDR: for each pixel, use the CDR from the dominant crop's
    #     best available irrigation scenario (drip > traditional > rainfed)
    irr_priority = ['drip', 'traditional', 'rainfed']
    composite_cdr_dominant = np.full((rows_g, cols_g), np.nan, dtype=np.float32)
    composite_crop_label = np.full((rows_g, cols_g), '', dtype=object)

    for i in range(rows_g):
        for j in range(cols_g):
            ci = dominant_crop_idx[i, j]
            if ci < 0:
                continue
            crop = all_crops_list[ci]
            for irr in irr_priority:
                if (crop, irr) in all_results:
                    val = all_results[(crop, irr)]['cdr_total'][i, j]
                    if np.isfinite(val):
                        composite_cdr_dominant[i, j] = val
                        composite_crop_label[i, j] = f"{crop[0].upper()}"
                        break

    # (b) Area-weighted CDR: weighted average across ALL crops at each pixel
    composite_cdr_weighted = np.full((rows_g, cols_g), np.nan, dtype=np.float32)
    composite_total_ha = np.zeros((rows_g, cols_g), dtype=np.float64)

    # Accumulate (CDR * ha) and (ha) across all scenarios
    weighted_sum = np.zeros((rows_g, cols_g), dtype=np.float64)
    ha_sum = np.zeros((rows_g, cols_g), dtype=np.float64)

    for crop, irr in ALL_SCENARIOS:
        if (crop, irr) not in all_results:
            continue
        cdr = all_results[(crop, irr)]['cdr_total']

        # Get appropriate area for this scenario
        if irr == 'rainfed':
            ha_map = area_maps.get(crop, {}).get('rain')
        else:
            ha_map = area_maps.get(crop, {}).get('irrig')
            if ha_map is not None:
                ha_map = ha_map * IRR_SPLIT

        if ha_map is None:
            ha_map = area_maps.get(crop, {}).get('total')
            if ha_map is not None:
                ha_map = ha_map * (1.0 if irr == 'rainfed' else 0.5)

        if ha_map is None:
            continue

        valid = np.isfinite(cdr) & (ha_map > 0)
        weighted_sum[valid] += cdr[valid] * ha_map[valid]
        ha_sum[valid] += ha_map[valid]

    has_data = ha_sum > 0
    composite_cdr_weighted[has_data] = (weighted_sum[has_data] / ha_sum[has_data]).astype(np.float32)
    composite_total_ha[has_data] = ha_sum[has_data]

    # Plot both composite maps side by side
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, 7))
    fig5.suptitle('Composite CDR Maps — Sicily, Mt. Etna Basalt (40 t/ha × 3)',
                  fontsize=13, fontweight='bold')

    vmax_comp = np.nanpercentile(composite_cdr_dominant[np.isfinite(composite_cdr_dominant)], 95) \
        if np.any(np.isfinite(composite_cdr_dominant)) else 0.5

    # (a) Dominant crop
    im5a = ax5a.imshow(composite_cdr_dominant, cmap='YlOrRd', vmin=0, vmax=vmax_comp,
                       origin='upper', aspect='auto')
    fig5.colorbar(im5a, ax=ax5a, label='CDR (t CO2/ha/yr)', shrink=0.8)
    n_dom = int(np.sum(np.isfinite(composite_cdr_dominant)))
    mean_dom = np.nanmean(composite_cdr_dominant)
    ax5a.set_title(f'(a) Dominant crop per pixel\nn={n_dom}, mean={mean_dom:.3f} t/ha/yr', fontsize=10)
    ax5a.set_xlabel('Col'); ax5a.set_ylabel('Row')

    # (b) Area-weighted average
    im5b = ax5b.imshow(composite_cdr_weighted, cmap='YlOrRd', vmin=0, vmax=vmax_comp,
                       origin='upper', aspect='auto')
    fig5.colorbar(im5b, ax=ax5b, label='CDR (t CO2/ha/yr)', shrink=0.8)
    n_wt = int(np.sum(np.isfinite(composite_cdr_weighted)))
    mean_wt = np.nanmean(composite_cdr_weighted)
    ax5b.set_title(f'(b) Area-weighted CDR (all crops)\nn={n_wt}, mean={mean_wt:.3f} t/ha/yr', fontsize=10)
    ax5b.set_xlabel('Col'); ax5b.set_ylabel('Row')

    plt.tight_layout()
    out5 = os.path.join(out_dir, 'composite_cdr_maps.png')
    plt.savefig(out5, dpi=150, bbox_inches='tight')
    print(f"Saved: {out5}")
    plt.close()

    # ── Plot 5b: Dominant crop identity map ──
    fig5c, ax5c = plt.subplots(figsize=(10, 7))
    crop_id_map = np.full((rows_g, cols_g), np.nan, dtype=np.float32)
    for i in range(rows_g):
        for j in range(cols_g):
            ci = dominant_crop_idx[i, j]
            if ci >= 0 and dominant_crop_ha[i, j] > 0:
                crop_id_map[i, j] = ci

    from matplotlib.colors import ListedColormap, BoundaryNorm
    crop_colors = ['#8c564b', '#9467bd', '#ff7f0e', '#e377c2', '#bcbd22']
    cmap_crop = ListedColormap(crop_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm_crop = BoundaryNorm(bounds, cmap_crop.N)

    im5c = ax5c.imshow(crop_id_map, cmap=cmap_crop, norm=norm_crop,
                       origin='upper', aspect='auto')
    cbar5c = fig5c.colorbar(im5c, ax=ax5c, ticks=range(5), shrink=0.8)
    cbar5c.ax.set_yticklabels([c.capitalize() for c in all_crops_list])
    ax5c.set_title('Dominant Crop per Pixel (by total cultivated ha)', fontsize=12)
    ax5c.set_xlabel('Col'); ax5c.set_ylabel('Row')

    out5c = os.path.join(out_dir, 'dominant_crop_map.png')
    plt.savefig(out5c, dpi=150, bbox_inches='tight')
    print(f"Saved: {out5c}")
    plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 6: CDR PARTITIONING — DIC leaching vs pedogenic carbonates
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- CDR partitioning maps ---")

    # Aggregate partitioning using dominant-crop approach
    comp_cdr_dic = np.full((rows_g, cols_g), np.nan, dtype=np.float32)
    comp_cdr_carb = np.full((rows_g, cols_g), np.nan, dtype=np.float32)

    for i in range(rows_g):
        for j in range(cols_g):
            ci = dominant_crop_idx[i, j]
            if ci < 0:
                continue
            crop = all_crops_list[ci]
            for irr in irr_priority:
                if (crop, irr) in all_results:
                    r = all_results[(crop, irr)]
                    dic_v = r['cdr_dic'][i, j]
                    carb_v = r['cdr_carb'][i, j]
                    if np.isfinite(dic_v) or np.isfinite(carb_v):
                        comp_cdr_dic[i, j] = dic_v if np.isfinite(dic_v) else 0.0
                        comp_cdr_carb[i, j] = carb_v if np.isfinite(carb_v) else 0.0
                        break

    # Fraction of CDR from DIC leaching
    total_part = np.where(np.isfinite(comp_cdr_dic), comp_cdr_dic, 0) + \
                 np.where(np.isfinite(comp_cdr_carb), comp_cdr_carb, 0)
    frac_dic = np.full((rows_g, cols_g), np.nan, dtype=np.float32)
    valid_part = (total_part > 0) & (np.isfinite(comp_cdr_dic))
    frac_dic[valid_part] = comp_cdr_dic[valid_part] / total_part[valid_part] * 100

    fig6, axes6 = plt.subplots(1, 3, figsize=(20, 6))
    fig6.suptitle('CDR Partitioning — DIC Leaching vs Pedogenic Carbonates\n'
                  'Dominant crop per pixel, Mt. Etna Basalt',
                  fontsize=13, fontweight='bold')

    im6a = axes6[0].imshow(comp_cdr_dic, cmap='YlOrRd', vmin=0, vmax=vmax_comp,
                           origin='upper', aspect='auto')
    fig6.colorbar(im6a, ax=axes6[0], label='t CO2/ha/yr', shrink=0.8)
    axes6[0].set_title(f'(a) DIC leaching CDR\nmean={np.nanmean(comp_cdr_dic):.3f}', fontsize=10)

    vmax_carb = max(np.nanpercentile(comp_cdr_carb[np.isfinite(comp_cdr_carb)], 95), 0.01) \
        if np.any(np.isfinite(comp_cdr_carb)) else 0.01
    im6b = axes6[1].imshow(comp_cdr_carb, cmap='PuBu', vmin=-vmax_carb, vmax=vmax_carb,
                           origin='upper', aspect='auto')
    fig6.colorbar(im6b, ax=axes6[1], label='t CO2/ha/yr', shrink=0.8)
    axes6[1].set_title(f'(b) Pedogenic carbonate CDR\nmean={np.nanmean(comp_cdr_carb):.4f}', fontsize=10)

    im6c = axes6[2].imshow(frac_dic, cmap='RdYlBu_r', vmin=0, vmax=100,
                           origin='upper', aspect='auto')
    fig6.colorbar(im6c, ax=axes6[2], label='% of total CDR', shrink=0.8)
    axes6[2].set_title(f'(c) DIC fraction of total CDR\nmean={np.nanmean(frac_dic):.1f}%', fontsize=10)

    for ax in axes6:
        ax.set_xlabel('Col'); ax.set_ylabel('Row')

    plt.tight_layout()
    out6 = os.path.join(out_dir, 'cdr_partitioning_maps.png')
    plt.savefig(out6, dpi=150, bbox_inches='tight')
    print(f"Saved: {out6}")
    plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 7: DELTA-pH MAP — pH(EW) - pH(noEW), last year average
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- Delta-pH maps ---")

    all_irrs_plot = ['drip', 'traditional', 'rainfed']

    # Per-scenario delta-pH grid (same layout as spatial CDR)
    fig7, axes7 = plt.subplots(len(all_crops_list), len(all_irrs_plot),
                               figsize=(5 * len(all_irrs_plot), 4 * len(all_crops_list)))
    fig7.suptitle('pH Change from Enhanced Weathering — pH(EW) − pH(noEW)\n'
                  'Last-year average, Mt. Etna Basalt (40 t/ha × 3)',
                  fontsize=14, fontweight='bold', y=1.01)

    all_dpH = []
    dpH_results = {}
    for crop, irr in ALL_SCENARIOS:
        if (crop, irr) not in all_results:
            continue
        pH_ew = load_npy(base, crop, irr, 'pH', 'basalt')
        pH_noew = load_npy(base, crop, irr, 'pH', 'noEW')
        if pH_ew is None or pH_noew is None:
            continue
        last_yr = slice(-365, None)
        dpH = np.nanmean(pH_ew[:, :, last_yr], axis=2) - np.nanmean(pH_noew[:, :, last_yr], axis=2)
        dpH_results[(crop, irr)] = dpH
        v = dpH[np.isfinite(dpH)]
        if len(v) > 0:
            all_dpH.extend(v.tolist())

    if all_dpH:
        dpH_absmax = max(abs(np.percentile(all_dpH, 5)), abs(np.percentile(all_dpH, 95)))
    else:
        dpH_absmax = 0.5

    for i_crop, crop in enumerate(all_crops_list):
        for i_irr, irr in enumerate(all_irrs_plot):
            ax = axes7[i_crop, i_irr]
            if (crop, irr) in dpH_results:
                dpH = dpH_results[(crop, irr)]
                n_v = int(np.sum(np.isfinite(dpH)))
                mean_v = np.nanmean(dpH)
                im = ax.imshow(dpH, cmap='RdBu_r', vmin=-dpH_absmax, vmax=dpH_absmax,
                               origin='upper', aspect='auto')
                ax.set_title(f'{crop.capitalize()} {irr.capitalize()}\n'
                             f'n={n_v}, mean ΔpH={mean_v:+.3f}', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No results', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')
                ax.set_title(f'{crop.capitalize()} {irr.capitalize()}\n(not available)', fontsize=9)
            if i_crop == len(all_crops_list) - 1:
                ax.set_xlabel('Col', fontsize=8)
            if i_irr == 0:
                ax.set_ylabel('Row', fontsize=8)
            ax.tick_params(labelsize=7)

    fig7.subplots_adjust(right=0.92)
    cbar_ax7 = fig7.add_axes([0.94, 0.15, 0.015, 0.7])
    sm7 = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-dpH_absmax, vmax=dpH_absmax))
    fig7.colorbar(sm7, cax=cbar_ax7, label='ΔpH (EW − noEW)')

    plt.tight_layout(rect=[0, 0, 0.92, 0.97])
    out7 = os.path.join(out_dir, 'delta_pH_spatial.png')
    plt.savefig(out7, dpi=150, bbox_inches='tight')
    print(f"Saved: {out7}")
    plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 8: IRRIGATION EFFECT MAP — CDR(drip) - CDR(rainfed) for olivo, vite
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- Irrigation effect maps ---")

    crops_with_3irr = [c for c in all_crops_list
                       if all((c, irr) in all_results for irr in all_irrs_plot)]

    if crops_with_3irr:
        n_crops_3 = len(crops_with_3irr)
        fig8, axes8 = plt.subplots(n_crops_3, 2, figsize=(12, 5 * n_crops_3),
                                   squeeze=False)
        fig8.suptitle('Irrigation Effect on CDR — Δ from Rainfed Baseline\n'
                      'Mt. Etna Basalt (40 t/ha × 3)',
                      fontsize=13, fontweight='bold')

        all_delta_irr = []
        for crop in crops_with_3irr:
            cdr_rain = all_results[(crop, 'rainfed')]['cdr_total']
            for irr in ['drip', 'traditional']:
                delta = all_results[(crop, irr)]['cdr_total'] - cdr_rain
                v = delta[np.isfinite(delta)]
                if len(v) > 0:
                    all_delta_irr.extend(v.tolist())

        if all_delta_irr:
            delta_absmax = max(abs(np.percentile(all_delta_irr, 5)),
                               abs(np.percentile(all_delta_irr, 95)))
        else:
            delta_absmax = 0.1

        for ic, crop in enumerate(crops_with_3irr):
            cdr_rain = all_results[(crop, 'rainfed')]['cdr_total']
            for ii, irr in enumerate(['drip', 'traditional']):
                ax = axes8[ic, ii]
                delta = all_results[(crop, irr)]['cdr_total'] - cdr_rain
                n_v = int(np.sum(np.isfinite(delta)))
                mean_d = np.nanmean(delta)
                im = ax.imshow(delta, cmap='RdBu', vmin=-delta_absmax, vmax=delta_absmax,
                               origin='upper', aspect='auto')
                fig8.colorbar(im, ax=ax, label='Δ CDR (t CO2/ha/yr)', shrink=0.8)
                ax.set_title(f'{crop.capitalize()}: {irr.capitalize()} − Rainfed\n'
                             f'n={n_v}, mean Δ={mean_d:+.4f}', fontsize=10)
                ax.set_xlabel('Col'); ax.set_ylabel('Row')

        plt.tight_layout()
        out8 = os.path.join(out_dir, 'irrigation_effect_cdr.png')
        plt.savefig(out8, dpi=150, bbox_inches='tight')
        print(f"Saved: {out8}")
        plt.close()
    else:
        print("  No crops with all 3 irrigation types available — skipping.")

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 9: MISSING PIXEL COVERAGE REPORT
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- Missing pixel coverage report ---")
    print(f"\n{'='*100}")
    print(f"MISSING PIXEL REPORT — Cultivated ha without CDR results")
    print(f"{'='*100}")
    print(f"{'Crop':>10s} {'Irrigation':>14s}  {'Cult.px':>8s} {'Valid':>8s} {'Miss.px':>8s} "
          f"{'Total ha':>12s} {'Valid ha':>12s} {'Missing ha':>12s} {'Miss%':>8s}")
    print('-' * 100)

    total_missing_ha = 0.0
    total_cult_ha = 0.0

    for crop, irr in ALL_SCENARIOS:
        total_ha = area_maps.get(crop, {}).get('total')
        if total_ha is None:
            print(f"{crop:>10s} {irr:>14s}  {'no area map':>8s}")
            continue

        cult_mask = (total_ha > 0) & wn_mask
        n_cult = int(cult_mask.sum())
        ha_cult = float(total_ha[cult_mask].sum())
        total_cult_ha += ha_cult

        if (crop, irr) not in all_results:
            print(f"{crop:>10s} {irr:>14s}  {n_cult:>8d} {'N/A':>8s} {n_cult:>8d} "
                  f"{ha_cult:>12.0f} {'0':>12s} {ha_cult:>12.0f} {'100.0%':>8s}")
            total_missing_ha += ha_cult
            continue

        cdr = all_results[(crop, irr)]['cdr_total']
        has_result = np.isfinite(cdr)
        valid_mask = cult_mask & has_result
        missing_mask = cult_mask & ~has_result

        n_valid = int(valid_mask.sum())
        n_miss = int(missing_mask.sum())
        ha_valid = float(total_ha[valid_mask].sum())
        ha_miss = float(total_ha[missing_mask].sum())
        total_missing_ha += ha_miss
        pct = ha_miss / ha_cult * 100 if ha_cult > 0 else 0

        flag = " ***" if pct > 5 else ""
        print(f"{crop:>10s} {irr:>14s}  {n_cult:>8d} {n_valid:>8d} {n_miss:>8d} "
              f"{ha_cult:>12.0f} {ha_valid:>12.0f} {ha_miss:>12.0f} {pct:>7.1f}%{flag}")

    print('-' * 100)
    pct_total = total_missing_ha / total_cult_ha * 100 if total_cult_ha > 0 else 0
    print(f"{'TOTAL':>10s} {'':>14s}  {'':>8s} {'':>8s} {'':>8s} "
          f"{total_cult_ha:>12.0f} {'':>12s} {total_missing_ha:>12.0f} {pct_total:>7.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 10: CDR vs MEAN LEACHING SCATTER
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- CDR vs mean leaching scatter ---")

    fig10, (ax10a, ax10b) = plt.subplots(1, 2, figsize=(16, 7))
    fig10.suptitle('CDR vs Mean Leaching Rate — Per Pixel\n'
                   'Mt. Etna Basalt (40 t/ha × 3, 30 years)',
                   fontsize=13, fontweight='bold')

    marker_irr = {'drip': 'o', 'traditional': 's', 'rainfed': '^'}

    for crop, irr in ALL_SCENARIOS:
        if (crop, irr) not in all_results:
            continue
        cdr = all_results[(crop, irr)]['cdr_total']
        L_daily = load_hydro_L(base, crop, irr)
        if L_daily is None:
            continue

        # Mean daily leaching per pixel (m/d → mm/d)
        mean_L = np.nanmean(L_daily, axis=2) * 1000.0  # mm/d

        valid = np.isfinite(cdr) & np.isfinite(mean_L) & (mean_L > 0)
        if not valid.any():
            continue

        # Panel (a): colored by crop
        ax10a.scatter(mean_L[valid], cdr[valid], s=12, alpha=0.4,
                      color=COLORS_CROP.get(crop, 'gray'),
                      marker=marker_irr.get(irr, 'o'),
                      label=f'{crop} {irr}')

        # Panel (b): colored by irrigation
        ax10b.scatter(mean_L[valid], cdr[valid], s=12, alpha=0.4,
                      color=COLORS_IRR.get(irr, 'gray'),
                      marker=marker_irr.get(irr, 'o'),
                      label=f'{crop} {irr}')

    # Deduplicate legends
    for ax, title in [(ax10a, '(a) Colored by crop'), (ax10b, '(b) Colored by irrigation')]:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=2, markerscale=1.5)
        ax.set_xlabel('Mean leaching (mm/d)', fontsize=11)
        ax.set_ylabel('CDR (t CO2/ha/yr)', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out10 = os.path.join(out_dir, 'cdr_vs_leaching_scatter.png')
    plt.savefig(out10, dpi=150, bbox_inches='tight')
    print(f"Saved: {out10}")
    plt.close()

    print(f"\nDone.")


if __name__ == '__main__':
    main()

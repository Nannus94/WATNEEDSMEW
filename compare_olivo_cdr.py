"""
Compare CDR across 3 olivo irrigation scenarios (drip, traditional, rainfed).
Computes: DIC leaching CDR + pedogenic carbonate CDR.
Uses weighted_alpha approach (same alpha for all scenarios).

Run on cluster:
  python compare_olivo_cdr.py --base-dir /scratch/user/lorenzo32/WATNEEDS+SMEW

Output: comparison plots saved to Results/
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
Zr = 0.3          # m
conv_mol = 1e6    # internal unit conversion
MM_CO2 = 44.01    # g/mol CO2
MM_C = 12.011     # g/mol C

SCENARIOS_IRR = ['drip', 'traditional', 'rainfed']
COLORS = {'drip': '#2ca02c', 'traditional': '#d62728', 'rainfed': '#1f77b4'}


def load_npy(base, crop, irr, var, scenario):
    """Load a .npy result file. Tries {var}_sic_{scenario} first, then {var}_{scenario}."""
    rdir = os.path.join(base, 'Results', f'{crop}_{irr}')
    # Chemistry vars use _sic_ prefix, rock/mineral vars don't
    for pattern in [f'{var}_sic_{scenario}_daily.npy', f'{var}_{scenario}_daily.npy']:
        fpath = os.path.join(rdir, pattern)
        if os.path.exists(fpath):
            return np.load(fpath)
    print(f"  WARNING: {var} for {scenario} not found in {rdir}")
    return None


def load_hydro_L(base, crop, irr, years=30):
    """Load leaching flux L from hydro data, tile to match EW simulation period."""
    irr_dir = 'surface' if irr in ('traditional', 'trad') else irr
    hydro_dir = os.path.join(base, 'WB_interpolated_first4hours', f'{crop}_{irr_dir}')

    if not os.path.isdir(hydro_dir):
        print(f"  WARNING: {hydro_dir} not found")
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

    L_full = np.concatenate(L_list, axis=2)  # (rows, cols, 4h_steps)

    # Convert from mm/4h to m/d: /1000 * 6
    L_full = L_full / 1000.0 * 6.0

    # Average to daily
    steps_per_day = 6
    n_days_avail = L_full.shape[2] // steps_per_day
    L_daily = np.mean(L_full[:, :, :n_days_avail * steps_per_day].reshape(
        L_full.shape[0], L_full.shape[1], n_days_avail, steps_per_day), axis=3)

    # Tile to cover SIM_YEARS (the EW period = last 30y of 60y simulation)
    if L_daily.shape[2] < N_DAYS:
        n_tiles = int(np.ceil(N_DAYS / L_daily.shape[2]))
        L_daily = np.tile(L_daily, (1, 1, n_tiles))[:, :, :N_DAYS]
    else:
        L_daily = L_daily[:, :, :N_DAYS]

    return L_daily


def compute_cdr_per_pixel(base, crop, irr, n_map):
    """
    Compute CDR per pixel [t CO2 / ha / yr] for one crop x irrigation scenario.

    CDR = DIC_leaching_CDR + pedogenic_carbonate_CDR

    DIC leaching CDR:
      For each day: flux = (DIC_EW - DIC_noEW) * L * 1000 [umol/m2/d]
      Integrate over 30 years, convert to t CO2/ha/yr

    Pedogenic carbonate CDR:
      delta_CaCO3 = CaCO3_EW(end) - CaCO3_EW(start) - [CaCO3_noEW(end) - CaCO3_noEW(start)]
      Each mol CaCO3 stores 1 mol CO2
    """
    print(f"\n  Computing CDR: {crop}/{irr}")

    # Load DIC
    dic_ew = load_npy(base, crop, irr, 'DIC', 'basalt')
    dic_noew = load_npy(base, crop, irr, 'DIC', 'noEW')
    if dic_ew is None or dic_noew is None:
        return None, None, None

    # Load carbonates
    caco3_ew = load_npy(base, crop, irr, 'CaCO3', 'basalt')
    caco3_noew = load_npy(base, crop, irr, 'CaCO3', 'noEW')
    mgco3_ew = load_npy(base, crop, irr, 'MgCO3', 'basalt')
    mgco3_noew = load_npy(base, crop, irr, 'MgCO3', 'noEW')

    # Load M_rock
    m_rock = load_npy(base, crop, irr, 'M_rock', 'basalt')

    # Load hydro L
    L_daily = load_hydro_L(base, crop, irr)
    if L_daily is None:
        print(f"  WARNING: No hydro L for {crop}/{irr}, DIC leaching CDR unavailable")

    rows, cols = dic_ew.shape[:2]

    # ── DIC leaching CDR ──
    cdr_dic = np.full((rows, cols), np.nan, dtype=np.float32)
    if L_daily is not None:
        # delta_DIC [umol/L] per day
        delta_dic = dic_ew - dic_noew  # (rows, cols, n_days)

        # DIC flux: delta_DIC * L * 1000 * n * Zr [umol/m2/d]
        # But biogeochem already accounts for Zr in the bucket model.
        # The leaching flux removes DIC * L * 1000 umol/m2/d from the bucket.
        # So CDR_DIC = sum_t(delta_DIC[t] * L[t]) * 1000 [umol/m2 over 30y]
        # Convert: umol/m2 / conv_mol = mol/m2, * MM_CO2/1e6 = t CO2/m2, * 1e4 = t CO2/ha
        for i in range(rows):
            for j in range(cols):
                if np.isnan(dic_ew[i, j, 0]):
                    continue
                ddic = delta_dic[i, j, :]
                L_px = L_daily[i, j, :len(ddic)]
                # Cumulative DIC leaching over 30 years [umol/m2]
                cum_dic = np.nansum(ddic * L_px * 1000.0)
                # Convert to t CO2 / ha / yr
                cdr_dic[i, j] = cum_dic / conv_mol * MM_CO2 / 1e6 * 1e4 / SIM_YEARS

    # ── Pedogenic carbonate CDR ──
    cdr_carb = np.full((rows, cols), np.nan, dtype=np.float32)
    if caco3_ew is not None and caco3_noew is not None:
        # CaCO3 and MgCO3 in internal units (mol-conv/m2 = umol/m2 with conv_mol=1e6)
        # delta = [EW_final - EW_start] - [noEW_final - noEW_start]
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

        # Each mol carbonate stores 1 mol CO2
        # Convert: mol-conv/m2 / conv_mol = mol/m2, * MM_CO2 / 1e6 = t CO2/m2, * 1e4 = t CO2/ha
        cdr_carb = (d_caco3 + d_mgco3) / conv_mol * MM_CO2 / 1e6 * 1e4 / SIM_YEARS

    # ── Rock dissolution ──
    rock_dissolved_pct = np.full((rows, cols), np.nan, dtype=np.float32)
    if m_rock is not None:
        rock_final = np.nanmean(m_rock[:, :, -365:], axis=2)  # g/m2
        total_applied = 3 * 4000.0  # g/m2
        rock_dissolved_pct = (total_applied - rock_final) / total_applied * 100

    # ── Total CDR ──
    cdr_total = np.full((rows, cols), np.nan, dtype=np.float32)
    valid = ~np.isnan(cdr_dic) & ~np.isnan(cdr_carb)
    cdr_total[valid] = cdr_dic[valid] + cdr_carb[valid]

    # Also handle case where only one component is available
    only_dic = ~np.isnan(cdr_dic) & np.isnan(cdr_carb)
    cdr_total[only_dic] = cdr_dic[only_dic]
    only_carb = np.isnan(cdr_dic) & ~np.isnan(cdr_carb)
    cdr_total[only_carb] = cdr_carb[only_carb]

    return {
        'cdr_dic': cdr_dic,
        'cdr_carb': cdr_carb,
        'cdr_total': cdr_total,
        'rock_dissolved_pct': rock_dissolved_pct,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare CDR across olivo irrigation scenarios')
    parser.add_argument('--base-dir', type=str,
                        default='/scratch/user/lorenzo32/WATNEEDS+SMEW')
    parser.add_argument('--crop', type=str, default='olivo')
    args = parser.parse_args()

    crop = args.crop
    base = args.base_dir

    # Load olive area map
    area_path = os.path.join(base, 'aree_colt', 'sicily10km_olives_total_ha.tif')
    if os.path.exists(area_path):
        with rasterio.open(area_path) as src:
            olive_ha = src.read(1).astype(np.float64)
        olive_ha[~np.isfinite(olive_ha)] = 0.0
    else:
        print(f"WARNING: {area_path} not found")
        olive_ha = None

    # Load porosity for n
    n_path = os.path.join(base, 'soil_param', 'n.tif')
    if os.path.exists(n_path):
        with rasterio.open(n_path) as src:
            n_map = src.read(1).astype(np.float32)
    else:
        n_map = np.full((39, 43), 0.43, dtype=np.float32)

    # Compute CDR for each irrigation scenario
    results = {}
    for irr in SCENARIOS_IRR:
        res_dir = os.path.join(base, 'Results', f'{crop}_{irr}')
        if not os.path.isdir(res_dir):
            print(f"\n  {crop}/{irr}: Results directory not found, skipping")
            continue
        results[irr] = compute_cdr_per_pixel(base, crop, irr, n_map)

    if not results:
        print("No results to compare!")
        sys.exit(1)

    # ── SUMMARY TABLE ──
    print(f"\n{'='*80}")
    print(f"CDR COMPARISON: {crop.upper()} — Drip vs Traditional vs Rainfed")
    print(f"{'='*80}")
    print(f"\n{'Irrigation':>15s}  {'CDR_DIC':>12s}  {'CDR_carb':>12s}  {'CDR_total':>12s}  {'Rock diss%':>12s}  {'Valid px':>10s}")
    print(f"{'':>15s}  {'t CO2/ha/yr':>12s}  {'t CO2/ha/yr':>12s}  {'t CO2/ha/yr':>12s}  {'%':>12s}  {'':>10s}")
    print('-' * 80)

    for irr in SCENARIOS_IRR:
        if irr not in results or results[irr] is None:
            print(f"{irr:>15s}  {'N/A':>12s}")
            continue
        r = results[irr]
        for key in ['cdr_dic', 'cdr_carb', 'cdr_total', 'rock_dissolved_pct']:
            arr = r[key]
            v = arr[~np.isnan(arr)]
        cdr_d = r['cdr_dic'][~np.isnan(r['cdr_dic'])]
        cdr_c = r['cdr_carb'][~np.isnan(r['cdr_carb'])]
        cdr_t = r['cdr_total'][~np.isnan(r['cdr_total'])]
        rock = r['rock_dissolved_pct'][~np.isnan(r['rock_dissolved_pct'])]
        n_valid = len(cdr_t)

        print(f"{irr:>15s}  {np.mean(cdr_d):>12.3f}  {np.mean(cdr_c):>12.4f}  {np.mean(cdr_t):>12.3f}  "
              f"{np.mean(rock):>12.1f}  {n_valid:>10d}")

    # ── Regional CDR (with olive ha weighting) ──
    if olive_ha is not None:
        print(f"\n--- Regional CDR (weighted by olive cultivated area) ---")
        print(f"{'Irrigation':>15s}  {'CDR/ha/yr':>12s}  {'Total ha':>12s}  {'Regional CDR':>14s}")
        print(f"{'':>15s}  {'t CO2':>12s}  {'':>12s}  {'t CO2/yr':>14s}")
        print('-' * 60)

        for irr in SCENARIOS_IRR:
            if irr not in results or results[irr] is None:
                continue
            r = results[irr]
            cdr = r['cdr_total']
            valid = ~np.isnan(cdr) & (olive_ha > 0)
            if not valid.any():
                continue

            # CDR per ha per year (spatial mean)
            cdr_per_ha = np.mean(cdr[valid])
            # Total olive ha in valid pixels
            total_ha = np.sum(olive_ha[valid])
            # Regional CDR = sum(CDR_per_ha[pixel] * olive_ha[pixel])
            regional_cdr = np.sum(cdr[valid] * olive_ha[valid])

            print(f"{irr:>15s}  {cdr_per_ha:>12.3f}  {total_ha:>12.0f}  {regional_cdr:>14.1f}")

    # ── PLOTS ──
    out_dir = os.path.join(base, 'Results')

    # Plot 1: CDR comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{crop.capitalize()} — CDR Comparison Across Irrigation Scenarios',
                 fontsize=13, fontweight='bold')

    for ax, (key, label, unit) in zip(axes, [
        ('cdr_dic', 'DIC Leaching CDR', 't CO2/ha/yr'),
        ('cdr_carb', 'Pedogenic Carbonate CDR', 't CO2/ha/yr'),
        ('cdr_total', 'Total CDR', 't CO2/ha/yr'),
    ]):
        means = []
        stds = []
        irr_labels = []
        for irr in SCENARIOS_IRR:
            if irr not in results or results[irr] is None:
                continue
            vals = results[irr][key]
            v = vals[~np.isnan(vals)]
            if len(v) > 0:
                means.append(np.mean(v))
                stds.append(np.std(v))
                irr_labels.append(irr.capitalize())

        if means:
            bars = ax.bar(range(len(means)), means, yerr=stds, capsize=5,
                          color=[COLORS[l.lower()] for l in irr_labels], alpha=0.7,
                          edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(irr_labels)
            for i, (m, s) in enumerate(zip(means, stds)):
                ax.text(i, m + s + 0.01, f'{m:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel(unit)
        ax.set_title(label)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(out_dir, f'{crop}_cdr_comparison_bars.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out1}")
    plt.close()

    # Plot 2: CDR spatial maps
    fig, axes = plt.subplots(len(SCENARIOS_IRR), 2, figsize=(12, 5 * len(SCENARIOS_IRR)))
    fig.suptitle(f'{crop.capitalize()} — CDR Spatial Maps', fontsize=13, fontweight='bold')

    for i, irr in enumerate(SCENARIOS_IRR):
        if irr not in results or results[irr] is None:
            continue
        r = results[irr]

        # CDR total
        cdr = r['cdr_total'].copy()
        cdr[np.isnan(cdr)] = np.nan
        vmax = np.nanpercentile(cdr[~np.isnan(cdr)], 95) if np.any(~np.isnan(cdr)) else 1

        im = axes[i, 0].imshow(cdr, cmap='YlOrRd', vmin=0, vmax=vmax,
                                origin='upper', aspect='auto')
        fig.colorbar(im, ax=axes[i, 0], shrink=0.7, label='t CO2/ha/yr')
        axes[i, 0].set_title(f'{irr.capitalize()} — Total CDR')

        # Rock dissolved %
        rock = r['rock_dissolved_pct'].copy()
        im2 = axes[i, 1].imshow(rock, cmap='Blues', vmin=0, vmax=100,
                                 origin='upper', aspect='auto')
        fig.colorbar(im2, ax=axes[i, 1], shrink=0.7, label='% dissolved')
        axes[i, 1].set_title(f'{irr.capitalize()} — Rock Dissolution (30y)')

    plt.tight_layout()
    out2 = os.path.join(out_dir, f'{crop}_cdr_spatial_maps.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")
    plt.close()

    print(f"\nDone.")


if __name__ == '__main__':
    main()

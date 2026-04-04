"""
Check EW simulation results.

Loads .npy outputs from Results/{crop}_{irrigation}/, validates coverage,
reports statistics, and computes CDR estimates.

Usage:
    python check_ew_results.py                           # uses defaults (vite_drip)
    python check_ew_results.py --crop olivo --irr traditional --base-dir /scratch/user/lorenzo32/WATNEEDS+SMEW
    python check_ew_results.py --all                     # check all available scenarios
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
VARS = ['pH', 'Ca', 'Mg', 'Na', 'K', 'DIC', 'HCO3', 'CO3', 'Alk', 'CaCO3', 'MgCO3']
SCENARIOS = ['noEW', 'basalt']
SIM_YEARS = 30
N_DAYS = SIM_YEARS * 365  # 10950

# For CDR: conversion constants
Zr = 0.3         # m
conv_mol = 1e6   # internal units = umol/L
MM_CO2 = 44.01   # g/mol


def load_npy(path):
    """Load .npy, return array or None."""
    if os.path.exists(path):
        return np.load(path)
    return None


def check_scenario(results_dir, crop, irrigation):
    """Check all output files for one crop/irrigation scenario."""

    print(f"\n{'='*70}")
    print(f"  {crop.upper()} / {irrigation.upper()}")
    print(f"  Directory: {results_dir}")
    print(f"{'='*70}")

    if not os.path.isdir(results_dir):
        print(f"  ERROR: directory not found!")
        return None

    # --- 1. File inventory ---
    print(f"\n--- File Inventory ---")
    files_found = {}
    files_missing = []

    for scenario in SCENARIOS:
        for var in VARS:
            fname = f"{var}_sic_{scenario}_daily.npy"
            fpath = os.path.join(results_dir, fname)
            if os.path.exists(fpath):
                size_mb = os.path.getsize(fpath) / 1e6
                files_found[(scenario, var)] = fpath
                print(f"  [OK]  {fname:45s}  {size_mb:7.1f} MB")
            else:
                files_missing.append(fname)
                print(f"  [--]  {fname:45s}  MISSING")

        # M_rock for EW scenarios
        if scenario != 'noEW':
            fname = f"M_rock_{scenario}_daily.npy"
            fpath = os.path.join(results_dir, fname)
            if os.path.exists(fpath):
                size_mb = os.path.getsize(fpath) / 1e6
                files_found[(scenario, 'M_rock')] = fpath
                print(f"  [OK]  {fname:45s}  {size_mb:7.1f} MB")
            else:
                files_missing.append(fname)
                print(f"  [--]  {fname:45s}  MISSING")

    total_expected = len(SCENARIOS) * len(VARS) + len([s for s in SCENARIOS if s != 'noEW'])
    print(f"\n  Found: {len(files_found)}/{total_expected} files")
    if files_missing:
        print(f"  Missing: {files_missing}")

    if not files_found:
        print("  No files to analyze!")
        return None

    # --- 2. Coverage & shape ---
    print(f"\n--- Pixel Coverage ---")
    coverage = {}

    for scenario in SCENARIOS:
        key = (scenario, 'pH')
        if key not in files_found:
            continue

        arr = np.load(files_found[key])
        rows, cols, n_t = arr.shape
        print(f"\n  Scenario: {scenario}")
        print(f"    Shape: {arr.shape}  (expected time dim: {N_DAYS})")

        if n_t != N_DAYS:
            print(f"    WARNING: unexpected time dimension! Got {n_t}, expected {N_DAYS}")

        # Valid = not all NaN along time axis
        valid_mask = ~np.all(np.isnan(arr), axis=2)
        n_valid = np.sum(valid_mask)
        n_total = valid_mask.size
        coverage[scenario] = valid_mask

        print(f"    Valid pixels: {n_valid} / {n_total} ({100*n_valid/n_total:.1f}%)")

        # Check for partial NaN (some days NaN, some not)
        any_nan = np.any(np.isnan(arr), axis=2)
        all_nan = np.all(np.isnan(arr), axis=2)
        partial_nan = any_nan & ~all_nan
        n_partial = np.sum(partial_nan)
        if n_partial > 0:
            print(f"    Partial NaN pixels: {n_partial} (some days missing)")

    # Coverage consistency between scenarios
    if len(coverage) > 1:
        keys = list(coverage.keys())
        consistent = np.all(coverage[keys[0]] == coverage[keys[1]])
        print(f"\n  Coverage consistent across scenarios: {consistent}")
        if not consistent:
            diff = coverage[keys[0]] != coverage[keys[1]]
            print(f"    Differ at {np.sum(diff)} pixels")

    # --- 3. Variable statistics ---
    print(f"\n--- Variable Statistics (last 5 years mean per pixel, then spatial stats) ---")

    last_5y = slice(-5*365, None)  # last 5 years

    for scenario in SCENARIOS:
        print(f"\n  Scenario: {scenario}")
        print(f"  {'Variable':>10s}  {'Mean':>10s}  {'Std':>10s}  {'Min':>10s}  {'Max':>10s}  {'Median':>10s}  {'Unit':>10s}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        units = {
            'pH': 'pH', 'Ca': 'umol/L', 'Mg': 'umol/L', 'Na': 'umol/L',
            'K': 'umol/L', 'DIC': 'umol/L', 'HCO3': 'umol/L', 'CO3': 'umol/L',
            'Alk': 'umol/L', 'CaCO3': 'mol/m2', 'MgCO3': 'mol/m2', 'M_rock': 'kg/m2',
        }

        check_vars = VARS + (['M_rock'] if scenario != 'noEW' else [])
        for var in check_vars:
            key = (scenario, var)
            if key not in files_found:
                continue

            arr = np.load(files_found[key])

            # Last 5 years mean per pixel
            px_mean = np.nanmean(arr[:, :, last_5y], axis=2)
            vals = px_mean[~np.isnan(px_mean)]

            if len(vals) == 0:
                print(f"  {var:>10s}  {'ALL NaN':>10s}")
                continue

            unit = units.get(var, '?')
            print(f"  {var:>10s}  {np.mean(vals):>10.2f}  {np.std(vals):>10.2f}  "
                  f"{np.min(vals):>10.2f}  {np.max(vals):>10.2f}  {np.median(vals):>10.2f}  {unit:>10s}")

            # Sanity checks
            if var == 'pH':
                if np.min(vals) < 3 or np.max(vals) > 11:
                    print(f"    WARNING: pH out of realistic range [{np.min(vals):.2f}, {np.max(vals):.2f}]")
                if np.any(vals < 0):
                    print(f"    WARNING: negative pH values!")
            elif var in ['Ca', 'Mg', 'Na', 'K', 'DIC', 'HCO3']:
                if np.any(vals < 0):
                    n_neg = np.sum(vals < 0)
                    print(f"    WARNING: {n_neg} pixels with negative concentrations")

    # --- 4. EW Effect (basalt - noEW) ---
    if all(s in coverage for s in ['noEW', 'basalt']):
        print(f"\n--- EW Effect: basalt - noEW (last 5 years) ---")

        key_vars = ['pH', 'DIC', 'Ca', 'Mg', 'CaCO3', 'MgCO3']
        print(f"  {'Variable':>10s}  {'Mean_diff':>10s}  {'Std_diff':>10s}  {'Min_diff':>10s}  {'Max_diff':>10s}  {'Unit':>10s}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        for var in key_vars:
            k_noew = (SCENARIOS[0], var)
            k_ew   = ('basalt', var)
            if k_noew not in files_found or k_ew not in files_found:
                continue

            arr_noew = np.load(files_found[k_noew])
            arr_ew   = np.load(files_found[k_ew])

            # Last 5y mean per pixel
            mean_noew = np.nanmean(arr_noew[:, :, last_5y], axis=2)
            mean_ew   = np.nanmean(arr_ew[:, :, last_5y], axis=2)
            diff = mean_ew - mean_noew

            vals = diff[~np.isnan(diff)]
            if len(vals) == 0:
                continue

            unit = units.get(var, '?')
            print(f"  {var:>10s}  {np.mean(vals):>+10.2f}  {np.std(vals):>10.2f}  "
                  f"{np.min(vals):>+10.2f}  {np.max(vals):>+10.2f}  {unit:>10s}")

    # --- 5. CDR Estimate ---
    if all(s in coverage for s in ['noEW', 'basalt']):
        print(f"\n--- CDR Estimate (basalt vs noEW, 30 years) ---")
        compute_cdr(files_found, results_dir)

    # --- 6. Mineral weathering seasonal plot ---
    if 'basalt' in coverage:
        print(f"\n--- Mineral Weathering Seasonal Plot ---")
        try:
            plot_mineral_cdr_seasonal(results_dir, crop, irrigation)
        except Exception as e:
            print(f"  Plot failed: {e}")

    return coverage


def compute_cdr(files_found, results_dir):
    """
    CDR = integral{ (DIC_EW - DIC_noEW) * L * n * Zr * 1000 } dt
        + (CaCO3_EW_final - CaCO3_noEW_final) * MM_CO2 / conv_mol
        + (MgCO3_EW_final - MgCO3_noEW_final) * MM_CO2 / conv_mol

    Note: We need hydro data (L) for the DIC leaching integral.
    Without it, we report pedogenic carbonate CDR and DIC stats.
    """
    # Load DIC
    dic_noew = load_npy(files_found.get(('noEW', 'DIC')))
    dic_ew   = load_npy(files_found.get(('basalt', 'DIC')))

    # Load carbonates
    caco3_noew = load_npy(files_found.get(('noEW', 'CaCO3')))
    caco3_ew   = load_npy(files_found.get(('basalt', 'CaCO3')))
    mgo3_noew  = load_npy(files_found.get(('noEW', 'MgCO3')))
    mgo3_ew    = load_npy(files_found.get(('basalt', 'MgCO3')))

    # Rock remaining
    m_rock = load_npy(files_found.get(('basalt', 'M_rock')))

    # --- Pedogenic carbonate CDR ---
    if caco3_ew is not None and caco3_noew is not None:
        # Final minus initial for each
        delta_caco3 = (np.nanmean(caco3_ew[:, :, -365:], axis=2) -
                       np.nanmean(caco3_ew[:, :, :365], axis=2)) - \
                      (np.nanmean(caco3_noew[:, :, -365:], axis=2) -
                       np.nanmean(caco3_noew[:, :, :365], axis=2))

        valid = ~np.isnan(delta_caco3)
        if valid.any():
            print(f"\n  Pedogenic CaCO3 (EW-noEW accumulation):")
            print(f"    Mean delta CaCO3: {np.mean(delta_caco3[valid]):+.4f} mol/m2")
            print(f"    Range: [{np.min(delta_caco3[valid]):+.4f}, {np.max(delta_caco3[valid]):+.4f}]")
            print(f"    CDR equiv: {np.mean(delta_caco3[valid]) * MM_CO2 / 1e6:+.4e} t CO2/m2")
        else:
            print(f"\n  Pedogenic CaCO3: ALL NaN — no valid pixels")

    if mgo3_ew is not None and mgo3_noew is not None:
        delta_mgco3 = (np.nanmean(mgo3_ew[:, :, -365:], axis=2) -
                       np.nanmean(mgo3_ew[:, :, :365], axis=2)) - \
                      (np.nanmean(mgo3_noew[:, :, -365:], axis=2) -
                       np.nanmean(mgo3_noew[:, :, :365], axis=2))

        valid = ~np.isnan(delta_mgco3)
        if valid.any():
            print(f"\n  Pedogenic MgCO3 (EW-noEW accumulation):")
            print(f"    Mean delta MgCO3: {np.mean(delta_mgco3[valid]):+.4f} mol/m2")
            print(f"    Range: [{np.min(delta_mgco3[valid]):+.4f}, {np.max(delta_mgco3[valid]):+.4f}]")
            print(f"    CDR equiv: {np.mean(delta_mgco3[valid]) * MM_CO2 / 1e6:+.4e} t CO2/m2")
        else:
            print(f"\n  Pedogenic MgCO3: ALL NaN — no valid pixels")

    # --- DIC leaching CDR (requires L from hydro data) ---
    if dic_ew is not None and dic_noew is not None:
        delta_dic = np.nanmean(dic_ew[:, :, -365*5:], axis=2) - \
                    np.nanmean(dic_noew[:, :, -365*5:], axis=2)
        valid = ~np.isnan(delta_dic)
        print(f"\n  DIC concentration increase (last 5y avg, EW-noEW):")
        print(f"    Mean delta DIC: {np.mean(delta_dic[valid]):+.2f} umol/L")
        print(f"    Range: [{np.min(delta_dic[valid]):+.2f}, {np.max(delta_dic[valid]):+.2f}]")
        print(f"    (Need hydro L data for flux-weighted CDR integral)")

    # --- Rock dissolution ---
    if m_rock is not None:
        rock_init = np.nanmean(m_rock[:, :, :365], axis=2)
        rock_final = np.nanmean(m_rock[:, :, -365:], axis=2)
        valid = ~np.isnan(rock_init) & ~np.isnan(rock_final)

        # Total applied: 3 x 4 kg/m2 = 12 kg/m2
        total_applied = 3 * 4.0
        dissolved = total_applied - rock_final[valid]

        print(f"\n  Rock dissolution (basalt):")
        print(f"    Applied total: {total_applied:.1f} kg/m2 (3 x 40 t/ha)")
        print(f"    Remaining (last year avg): {np.mean(rock_final[valid]):.3f} kg/m2")
        print(f"    Dissolved: {np.mean(dissolved):.3f} kg/m2 ({100*np.mean(dissolved)/total_applied:.1f}%)")
        print(f"    Range dissolved: [{np.min(dissolved):.3f}, {np.max(dissolved):.3f}]")


def plot_mineral_cdr_seasonal(results_dir, crop, irrigation):
    """
    Plot seasonal evolution of per-mineral weathering flux and total CDR rate.
    Uses EW_{mineral}_basalt_daily.npy files.
    Shows DOY climatology (mean + IQR across pixels and years).
    """
    minerals = ['labradorite', 'albite', 'diopside', 'anorthite']
    colors = {'labradorite': '#d62728', 'albite': '#9467bd',
              'diopside': '#2ca02c', 'anorthite': '#ff7f0e'}

    # Stoichiometric CO2 moles per mole of mineral dissolved
    # Based on cation release → carbonate equilibrium → DIC export
    # Labradorite (Ca0.6Na0.4)(Al1.6Si2.4)O8: releases 0.6 Ca + 0.4 Na → ~1.6 mol CO2/mol
    # Albite NaAlSi3O8: 1 Na → ~1 mol CO2/mol
    # Diopside CaMgSi2O6: 1 Ca + 1 Mg → ~4 mol CO2/mol
    # Anorthite CaAl2Si2O8: 1 Ca → ~2 mol CO2/mol
    # (Simplified — actual CDR depends on DIC speciation and leaching)
    # For this plot we show RAW weathering flux (EW), not CDR conversion
    # The total effective CDR comes from DIC(EW) - DIC(noEW) * L

    # Load DIC for total CDR rate
    dic_ew_path = os.path.join(results_dir, 'DIC_sic_basalt_daily.npy')
    dic_noew_path = os.path.join(results_dir, 'DIC_sic_noEW_daily.npy')

    # Load per-mineral EW fluxes
    ew_data = {}
    for mname in minerals:
        fpath = os.path.join(results_dir, f'EW_{mname}_basalt_daily.npy')
        if os.path.exists(fpath):
            ew_data[mname] = np.load(fpath)  # (rows, cols, n_days)
        else:
            print(f"  WARNING: {fpath} not found — skipping mineral plot")

    if not ew_data:
        print("  No per-mineral EW files found. Skipping seasonal plot.")
        return

    # Build DOY climatology for each mineral (spatial mean per day → reshape to years × 365)
    n_days = list(ew_data.values())[0].shape[2]
    n_years = n_days // 365

    fig, ax = plt.subplots(figsize=(12, 6))

    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    for mname in minerals:
        if mname not in ew_data:
            continue
        arr = ew_data[mname]  # (rows, cols, n_days)

        # Spatial mean (across valid pixels) per day
        valid_mask = ~np.all(np.isnan(arr), axis=2)
        spatial_mean = np.nanmean(arr, axis=(0, 1))  # (n_days,)

        # Reshape to (n_years, 365) and compute DOY stats
        ts = spatial_mean[:n_years * 365].reshape(n_years, 365)
        doy_mean = np.nanmean(ts, axis=0)
        doy_q25 = np.nanpercentile(ts, 25, axis=0)
        doy_q75 = np.nanpercentile(ts, 75, axis=0)

        # Smooth with 7-day rolling mean for readability
        def smooth(x, w=7):
            return np.convolve(x, np.ones(w)/w, mode='same')

        doy_mean_s = smooth(doy_mean)
        doy_q25_s = smooth(doy_q25)
        doy_q75_s = smooth(doy_q75)

        doy = np.arange(365)
        ax.fill_between(doy, doy_q25_s, doy_q75_s, color=colors[mname], alpha=0.15)
        ax.plot(doy, doy_mean_s, color=colors[mname], lw=2,
                label=f'{mname.capitalize()}')

    # Total effective CDR rate from DIC difference (if available)
    if os.path.exists(dic_ew_path) and os.path.exists(dic_noew_path):
        dic_ew = np.load(dic_ew_path)
        dic_noew = np.load(dic_noew_path)
        delta_dic = np.nanmean(dic_ew, axis=(0, 1)) - np.nanmean(dic_noew, axis=(0, 1))
        ts_dic = delta_dic[:n_years * 365].reshape(n_years, 365)
        dic_doy_mean = smooth(np.nanmean(ts_dic, axis=0))
        dic_doy_q25 = smooth(np.nanpercentile(ts_dic, 25, axis=0))
        dic_doy_q75 = smooth(np.nanpercentile(ts_dic, 75, axis=0))

        ax.fill_between(np.arange(365), dic_doy_q25, dic_doy_q75,
                         color='steelblue', alpha=0.1)
        ax.plot(np.arange(365), dic_doy_mean, color='steelblue', lw=2,
                ls='--', label='Total DIC increase (EW-noEW)')

    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=11)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Weathering flux (spatial mean)', fontsize=12)
    ax.set_title(f'Seasonal Evolution of Mineral Weathering vs Total DIC Effect\n'
                 f'{crop.capitalize()} {irrigation.capitalize()} — EW Basalt (Mt. Etna)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlim(0, 364)

    outpath = os.path.join(results_dir, f'mineral_cdr_seasonal_{crop}_{irrigation}.png')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {outpath}")


def discover_scenarios(base_dir):
    """Find all Results/{crop}_{irrigation}/ directories."""
    results_base = os.path.join(base_dir, 'Results')
    if not os.path.isdir(results_base):
        return []

    scenarios = []
    for d in sorted(os.listdir(results_base)):
        full = os.path.join(results_base, d)
        if not os.path.isdir(full):
            continue
        parts = d.split('_', 1)
        if len(parts) == 2:
            scenarios.append({'crop': parts[0], 'irrigation': parts[1], 'dir': full})
        else:
            scenarios.append({'crop': d, 'irrigation': '?', 'dir': full})

    return scenarios


def main():
    parser = argparse.ArgumentParser(description='Check EW simulation results')
    parser.add_argument('--crop', type=str, default='vite')
    parser.add_argument('--irr', type=str, default='drip')
    parser.add_argument('--base-dir', type=str,
                        default='/scratch/user/lorenzo32/WATNEEDS+SMEW')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Explicit path to results directory (overrides --crop/--irr)')
    parser.add_argument('--all', action='store_true',
                        help='Check all available scenarios')
    parser.add_argument('--scenarios', nargs='+', default=['noEW', 'basalt'],
                        help='Scenarios to check (default: noEW basalt)')
    args = parser.parse_args()

    global SCENARIOS
    SCENARIOS = args.scenarios

    if args.all:
        found = discover_scenarios(args.base_dir)
        if not found:
            print(f"No Results/ subdirectories found in {args.base_dir}")
            sys.exit(1)
        print(f"Found {len(found)} scenario(s):")
        for s in found:
            print(f"  {s['crop']}/{s['irrigation']} -> {s['dir']}")
        for s in found:
            check_scenario(s['dir'], s['crop'], s['irrigation'])
    elif args.results_dir:
        crop = args.crop
        irr = args.irr
        check_scenario(args.results_dir, crop, irr)
    else:
        results_dir = os.path.join(args.base_dir, 'Results', f'{args.crop}_{args.irr}')
        check_scenario(results_dir, args.crop, args.irr)


if __name__ == '__main__':
    main()

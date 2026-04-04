"""
Plot Moisture, Leaching (DP orig + L shallow), ET, I for surface irrigation.
2 non-coastal pixels, single-pixel time series (not means).
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import rasterio
from rasterio.warp import reproject, Resampling
import scipy.io
import h5py

base_dir = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW"
main_output_dir = os.path.join(base_dir, "SMEW_Output_4Hour_MERIDA")
soil_maps_dir = os.path.join(base_dir, "soil_param (1)")
ref_map_path = os.path.join(base_dir, r"Aree_coltivate\sicilia_cellarea_10km.tif")
def _surf_path(fname):
    return os.path.join(base_dir, r"SURFACE\giornalieri", fname)
fc_file = os.path.join(soil_maps_dir, r"fc_sicilia (3).tif")
wp_file = os.path.join(soil_maps_dir, r"wp_sicilia (3).tif")
ths_file = os.path.join(soil_maps_dir, r"ths_sicilia (4).tif")
OUTPUT_DIR = os.path.join(base_dir, "_DIAGNOSIS_SURFACE_PIXELS")

# 2 pixels to plot
PIXELS = [(10, 22, "P1"), (10, 21, "P2")]
YEAR_START, YEAR_END = 2020, 2022  # 2–3 years for readability
tif_h, tif_w = 39, 43
N_SUBSTEPS = 6
CROP_MAP = {'vite': 1, 'olivo': 2, 'pesco': 3, 'agrumi': 4, 'grano': 5}
ZR_RULES = {1: 1.50, 2: 1.45, 3: 1.50, 4: 1.50, 5: 1.50}  # irr

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_mat(path, var):
    try:
        return scipy.io.loadmat(path)[var]
    except Exception:
        with h5py.File(path, 'r') as f:
            return np.array(f.get(var)).T


def crop_center(arr, h, w):
    if arr is None or arr.ndim < 2:
        return None
    H, W = arr.shape[:2]
    sr, sc = (H - h) // 2, (W - w) // 2
    if arr.ndim == 3:
        return arr[sr:sr+h, sc:sc+w, :]
    return arr[sr:sr+h, sc:sc+w]


def load_and_match(path, ref_profile):
    with rasterio.open(path) as src:
        dest = np.zeros((ref_profile['height'], ref_profile['width']), dtype=np.float64)
        dst_crs = ref_profile.get('crs') or src.crs
        if dst_crs is None:
            from rasterio.crs import CRS
            dst_crs = CRS.from_epsg(32633)
        reproject(rasterio.band(src, 1), dest,
                  src_transform=src.transform, src_crs=src.crs or dst_crs,
                  dst_transform=ref_profile['transform'], dst_crs=dst_crs,
                  resampling=Resampling.nearest)
        if src.nodata is not None:
            dest[dest == src.nodata] = np.nan
        dest[np.isinf(dest)] = np.nan
        return dest


def main():
    with rasterio.open(ref_map_path) as src:
        ref_profile = src.profile.copy()
        ref_data = src.read(1).astype(float)
    land_mask = np.isfinite(ref_data) & (ref_data > 0)
    n_map = load_and_match(ths_file, ref_profile)
    n_map[~land_mask] = np.nan
    raw_fc = load_and_match(fc_file, ref_profile)
    raw_fc[~land_mask] = np.nan
    raw_wp = load_and_match(wp_file, ref_profile)
    raw_wp[~land_mask] = np.nan

    # Surface folders
    surf_folders = [f for f in os.listdir(main_output_dir)
                    if f.endswith('_surface') and os.path.isdir(os.path.join(main_output_dir, f))]
    if not surf_folders:
        print("No surface folders found.")
        return

    for folder in surf_folders:
        crop_name = folder.replace('_surface', '')
        if crop_name not in CROP_MAP:
            continue
        crop_code = CROP_MAP[crop_name]
        folder_path = os.path.join(main_output_dir, folder)
        zr = ZR_RULES[crop_code]
        cap_deep = n_map * zr * 1000.0

        # Collect data per pixel
        data = {k: {'s_shal': [], 's_deep': [], 'L_shal': [], 'DP': [], 'ET': [], 'T': [], 'I_shal': [], 'I_orig': []}
                for k in PIXELS}
        dates_all = []

        for year in range(YEAR_START, YEAR_END + 1):
            for month in range(1, 13):
                f_s = os.path.join(folder_path, f"shallow_s_{year}_{month}.mat")
                f_pr = _surf_path(f"outputPR_{year}_{month}_{crop_code}.mat")
                if not os.path.exists(f_s) or not os.path.exists(f_pr):
                    continue
                try:
                    s_g = load_mat(f_s, 's_shallow')
                    L_g = load_mat(os.path.join(folder_path, f"shallow_L_{year}_{month}.mat"), 'L_shallow')
                    T_g = load_mat(os.path.join(folder_path, f"shallow_T_{year}_{month}.mat"), 'T_shallow')
                    f_i = os.path.join(folder_path, f"shallow_I_{year}_{month}.mat")
                    I_g = load_mat(f_i, 'I_shallow') if os.path.exists(f_i) else np.zeros_like(L_g)
                except Exception as e:
                    print(f"Skip {year}-{month}: {e}")
                    continue

                f_et = _surf_path(f"outputET_{year}_{month}_{crop_code}.mat")
                f_dp = _surf_path(f"outputDP_{year}_{month}_{crop_code}.mat")
                f_tb = _surf_path(f"outputTB_{year}_{month}_{crop_code}.mat")
                if not os.path.exists(f_et) or not os.path.exists(f_dp):
                    continue
                et_raw = load_mat(f_et, 'outputET')
                dp_raw = load_mat(f_dp, 'outputDP')
                tb_raw = load_mat(f_tb, 'outputTB') if os.path.exists(f_tb) else None
                pr_raw = load_mat(f_pr, 'outputPR')
                f_bw = _surf_path(f"outputBW_{year}_{month}_{crop_code}.mat")
                bw_raw = load_mat(f_bw, 'outputBW') if os.path.exists(f_bw) else np.zeros_like(pr_raw)

                et_c = crop_center(et_raw, tif_h, tif_w)
                dp_c = crop_center(dp_raw, tif_h, tif_w)
                tb_c = crop_center(tb_raw, tif_h, tif_w)
                pr_c = crop_center(pr_raw, tif_h, tif_w)
                bw_c = crop_center(bw_raw, tif_h, tif_w) if bw_raw is not None else np.zeros_like(pr_c)

                n_days = s_g.shape[2] // N_SUBSTEPS
                n_steps = n_days * N_SUBSTEPS

                for (r, c, name) in PIXELS:
                    s_sh = np.nanmean(s_g[r, c, :n_steps].reshape(n_days, N_SUBSTEPS), axis=1)
                    L_sh = np.nansum(L_g[r, c, :n_steps].reshape(n_days, N_SUBSTEPS), axis=1)
                    T_sh = np.nansum(T_g[r, c, :n_steps].reshape(n_days, N_SUBSTEPS), axis=1)
                    I_sh = np.nansum(I_g[r, c, :n_steps].reshape(n_days, N_SUBSTEPS), axis=1)
                    data[(r, c, name)]['s_shal'].extend(s_sh)
                    data[(r, c, name)]['L_shal'].extend(L_sh)
                    data[(r, c, name)]['T'].extend(T_sh)
                    data[(r, c, name)]['I_shal'].extend(I_sh)
                    data[(r, c, name)]['DP'].extend(dp_c[r, c, :n_days] if dp_c is not None else np.full(n_days, np.nan))
                    data[(r, c, name)]['ET'].extend(et_c[r, c, :n_days] if et_c is not None else np.full(n_days, np.nan))
                    s_deep = (tb_c[r, c, :n_days] / np.maximum(cap_deep[r, c], 1e-6)) if tb_c is not None else np.full(n_days, np.nan)
                    data[(r, c, name)]['s_deep'].extend(s_deep)
                    pr_d = pr_c[r, c, :n_days] if pr_c is not None else np.zeros(n_days)
                    bw_d = bw_c[r, c, :n_days] if bw_c is not None else np.zeros(n_days)
                    data[(r, c, name)]['I_orig'].extend(np.nan_to_num(pr_d, nan=0) + np.nan_to_num(bw_d, nan=0))

                for d in range(n_days):
                    dates_all.append(datetime(year, month, 1) + timedelta(days=d))

        if len(dates_all) == 0:
            print(f"No data for {folder}")
            continue

        dates = np.array(dates_all)
        # Only plot pixels with data
        pixels_to_plot = [k for k in PIXELS if len(data[k]['s_shal']) == len(dates)]
        if len(pixels_to_plot) < 2:
            print(f"Skip {folder}: not enough pixels with matching data")
            continue

        # Plot: 2 pixels x 4 panels (s, L, ET, I)
        fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharex=True)
        fig.suptitle(f"SURFACE — {crop_name.upper()} — 2 inland pixels", fontsize=12)

        for i, (r, c, name) in enumerate(pixels_to_plot[:2]):
            d = data[(r, c, name)]
            s_sh = np.array(d['s_shal'])
            L_sh = np.array(d['L_shal'])
            DP = np.array(d['DP'])
            ET = np.array(d['ET'])
            T = np.array(d['T'])
            I_sh = np.array(d['I_shal'])
            I_orig = np.array(d['I_orig'])

            # Moisture
            s_deep_arr = np.array(d['s_deep'])
            axes[i, 0].plot(dates, s_sh, color='#009E73', lw=0.8, label='s shallow')
            axes[i, 0].plot(dates, s_deep_arr, color='gray', ls=':', lw=0.8, label='s deep')
            axes[i, 0].set_ylabel('s [-]')
            axes[i, 0].set_ylim(0, 1.05)
            axes[i, 0].legend(fontsize=7)
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_title(f"{name} ({r},{c}) — Moisture")

            # Leaching: DP + L
            axes[i, 1].plot(dates, DP, color='#D55E00', lw=0.8, label='DP orig (mm/d)')
            axes[i, 1].plot(dates, L_sh, color='#0072B2', lw=0.8, label='L shallow (mm/d)')
            axes[i, 1].set_ylabel('mm/day')
            axes[i, 1].set_ylim(bottom=0)
            axes[i, 1].legend(fontsize=7)
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_title(f"{name} — Leaching")

            # ET
            axes[i, 2].plot(dates, ET, color='gray', ls=':', lw=0.8, label='ET deep')
            axes[i, 2].plot(dates, T, color='#0072B2', lw=0.8, label='T shallow')
            axes[i, 2].set_ylabel('mm/day')
            axes[i, 2].set_ylim(bottom=0)
            axes[i, 2].legend(fontsize=7)
            axes[i, 2].grid(True, alpha=0.3)
            axes[i, 2].set_title(f"{name} — ET")

            # Infiltration
            axes[i, 3].plot(dates, I_orig, color='gray', ls=':', lw=0.8, label='I orig (PR+BW)')
            axes[i, 3].plot(dates, I_sh, color='#56B4E9', lw=0.8, label='I shallow')
            axes[i, 3].set_ylabel('mm/day')
            axes[i, 3].set_ylim(bottom=0)
            axes[i, 3].legend(fontsize=7)
            axes[i, 3].grid(True, alpha=0.3)
            axes[i, 3].set_title(f"{name} — Infiltration")

        for j in range(4):
            axes[1, j].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            axes[1, j].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
            axes[1, j].tick_params(axis='x', rotation=25)
        axes[1, 0].set_xlabel('Date')
        axes[1, 1].set_xlabel('Date')
        axes[1, 2].set_xlabel('Date')
        axes[1, 3].set_xlabel('Date')

        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"surface_{crop_name}_2pixels.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()

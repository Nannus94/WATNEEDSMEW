"""
Plot veg_mature phenology vs mean transpiration for all 5 crops.
Uses CROP_PARAMS from calibration script + local SMEW_Output_4Hour_MERIDA data.
"""
import sys, os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Latitude 5511\EW Models\PRIN CHANCES\EW_Model_CHANCES")
from pyEW.vegetation import veg_mature

# ── CROP_PARAMS (from calibration_full_map_multi_robust_noFert.py) ─────────────
CROP_PARAMS = {
    'Vite':   {'K_max': 1450.0, 'v_min_ratio': 0.05},
    'Olivo':  {'K_max': 3530.0, 'v_min_ratio': 0.70},
    'Agrumi': {'K_max': 5790.0, 'v_min_ratio': 0.80},
    'Pesco':  {'K_max': 2960.0, 'v_min_ratio': 0.10},
    'Grano':  {'K_max': 1000.0, 'v_min_ratio': 0.00},
}

# irrigation to use for T (surface where available, rainfed for grano)
HYDRO_DIR = {
    'Vite':   r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\SMEW_Output_4Hour_MERIDA\vite_surface",
    'Olivo':  r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\SMEW_Output_4Hour_MERIDA\olivo_surface",
    'Agrumi': r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\SMEW_Output_4Hour_MERIDA\agrumi_surface",
    'Pesco':  r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\SMEW_Output_4Hour_MERIDA\pesco_surface",
    'Grano':  r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\SMEW_Output_4Hour_MERIDA\grano_rainfed",
}

STEPS_DAY = 6   # 4h resolution

# ── Load multi-year T, compute DOY climatology ────────────────────────────────
def load_T_climatology(hydro_dir, years=range(1993, 2013)):
    """
    Load T for multiple years, build daily climatology (mean ± std per DOY).
    Returns doy_mean [365], doy_q25 [365], doy_q75 [365]
    """
    daily_by_doy = [[] for _ in range(365)]
    doy_offset = 0
    for year in years:
        year_vals = []
        for month in range(1, 13):
            fpath = os.path.join(hydro_dir, f"shallow_T_{year}_{month}.mat")
            if not os.path.exists(fpath):
                continue
            mat  = scipy.io.loadmat(fpath)
            key  = [k for k in mat if not k.startswith('_')][0]
            data = mat[key].astype(np.float32)       # (rows, cols, timesteps)
            n_steps = data.shape[2]
            n_days  = n_steps // STEPS_DAY
            data    = data[:, :, :n_days * STEPS_DAY]
            # spatial mean over valid pixels (T > 0)
            valid = (data > 0) & np.isfinite(data)
            spatial = np.where(valid, data, np.nan)
            sp_mean = np.nanmean(spatial, axis=(0, 1))          # (n_steps,)
            daily   = sp_mean.reshape(n_days, STEPS_DAY).sum(axis=1)  # mm/d
            year_vals.extend(daily.tolist())
        # assign to DOY bins (truncate to 365)
        for d, val in enumerate(year_vals[:365]):
            daily_by_doy[d].append(val)

    doy_mean = np.array([np.nanmean(v) if v else np.nan for v in daily_by_doy])
    doy_q25  = np.array([np.nanpercentile(v, 25) if v else np.nan for v in daily_by_doy])
    doy_q75  = np.array([np.nanpercentile(v, 75) if v else np.nan for v in daily_by_doy])
    return doy_mean, doy_q25, doy_q75

# ── Plot ──────────────────────────────────────────────────────────────────────
crops = list(CROP_PARAMS.keys())
fig, axes = plt.subplots(len(crops), 1, figsize=(12, 16), sharex=True)
fig.suptitle("veg_mature phenology vs WATNEEDS Transpiration climatology\n(normalized, surface irrigation)",
             fontsize=13, fontweight='bold')

doy = np.arange(365)

for ax, crop in zip(axes, crops):
    p  = CROP_PARAMS[crop]
    v  = veg_mature(doy.astype(float), crop, p['K_max'], v_min_ratio=p['v_min_ratio'])
    v_norm = (v - v.min()) / (v.max() - v.min() + 1e-9)

    # transpiration climatology
    T_mean = T_q25 = T_q75 = None
    if os.path.exists(HYDRO_DIR[crop]):
        T_mean, T_q25, T_q75 = load_T_climatology(HYDRO_DIR[crop])
        # normalize
        T_max = np.nanmax(T_mean)
        if T_max > 0:
            T_mean /= T_max; T_q25 /= T_max; T_q75 /= T_max

    ax2 = ax.twinx()

    ax.plot(doy, v_norm, color='green', lw=2.5, label='v/K_max (veg_mature)')
    ax.set_ylabel('v  (norm.)', color='green', fontsize=9)
    ax.tick_params(axis='y', labelcolor='green')
    ax.set_ylim(-0.05, 1.2)

    if T_mean is not None:
        ax2.fill_between(doy, T_q25, T_q75, color='steelblue', alpha=0.25, label='T  25–75%')
        ax2.plot(doy, T_mean, color='steelblue', lw=1.8, label='T mean (norm.)')
    ax2.set_ylabel('T  (norm.)', color='steelblue', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.set_ylim(-0.05, 1.3)

    title = (f"{crop}   K_max={p['K_max']:.0f} gC/m²   "
             f"v_min_ratio={p['v_min_ratio']}   "
             f"irr={os.path.basename(HYDRO_DIR[crop]).split('_')[-1]}")
    ax.set_title(title, fontsize=9)

    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Day of year', fontsize=10)
plt.tight_layout()

out = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\veg_vs_T.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.show()

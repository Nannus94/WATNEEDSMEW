"""
Inspect time series of s, L, T for pixels that are:
  - valid in s (soil moisture)
  - BUT have ET=all-NaN and DP=all-zero

Data source: olivo_surface (shallow_s, shallow_L, shallow_T)
These are scipy .mat files, shape (39, 43, timesteps), no transpose needed.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import re
import datetime

DATA_DIR = Path(r"C:\Users\Latitude 5511\Downloads\olivo_surface")
OUT_DIR  = Path(r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW")

# ── 1. Load masks ─────────────────────────────────────────────────────────────
# s from olivo_surface (shallow_s) — the processed water balance output
s_mask = np.load(str(OUT_DIR / "valid_mask_s_olivo_surface.npy"))    # (39,43)
L_mask = np.load(str(OUT_DIR / "valid_mask_L_olivo_surface.npy"))

# ET and DP from giornalieri_SURF_5crops (raw WATNEEDS output, crop 2)
et_mask = np.load(str(OUT_DIR / "valid_mask_ET_crop2_SURF.npy"))
dp_mask = np.load(str(OUT_DIR / "valid_mask_DP_crop2_SURF.npy"))
bp_mask = np.load(str(OUT_DIR / "valid_mask_BP_crop2_SURF_v2.npy"))

print(f"s  (shallow) valid : {s_mask.sum()}")
print(f"L  (shallow) valid : {L_mask.sum()}")
print(f"BP (WATNEEDS) valid: {bp_mask.sum()}")
print(f"ET (WATNEEDS) valid: {et_mask.sum()}")
print(f"DP (WATNEEDS) valid: {dp_mask.sum()}")

# Anomalous: BP valid (has rain/water) but ET=NaN and DP=0 in WATNEEDS
anomalous = bp_mask & ~et_mask & ~dp_mask
print(f"\nAnomalous (BP ok, ET=NaN, DP=0): {anomalous.sum()}")

anom_idx = list(zip(*np.where(anomalous)))
print(f"Anomalous pixel indices: {anom_idx[:10]}")

# Pick a few representative pixels to plot — spread across the list
n_plot = min(4, len(anom_idx))
step = max(1, len(anom_idx) // n_plot)
selected = anom_idx[::step][:n_plot]
print(f"Selected anomalous pixels to plot: {selected}")

# ── 2. Load full time series for selected pixels ──────────────────────────────
def load_all_months(var_name, key_name):
    """Load and concatenate all monthly files for a variable."""
    pattern = re.compile(rf"^shallow_{var_name}_(\d{{4}})_(\d{{1,2}})\.mat$")
    files = []
    for f in DATA_DIR.iterdir():
        m = pattern.match(f.name)
        if m:
            files.append((int(m.group(1)), int(m.group(2)), f))
    files.sort(key=lambda x: (x[0], x[1]))

    arrays = []
    dates  = []
    for year, month, fpath in files:
        mat  = sio.loadmat(str(fpath))
        data = mat[key_name]          # (39, 43, steps_in_month)
        arrays.append(data)
        n_steps = data.shape[2]
        steps_per_day = n_steps // 30  # approx
        import calendar
        n_days = calendar.monthrange(year, month)[1]
        steps_per_day_exact = n_steps // n_days
        for d in range(n_days):
            for step in range(steps_per_day_exact):
                dates.append(datetime.date(year, month, d + 1))

    full = np.concatenate(arrays, axis=2)
    return full, dates

print("\nLoading s, L, T time series...")
s_full, dates = load_all_months("s", "s_shallow")
L_full, _     = load_all_months("L", "L_shallow")
T_full, _     = load_all_months("T", "T_shallow")
print(f"Loaded shape: {s_full.shape}  ({len(dates)} timesteps)")

# Convert dates to matplotlib format
import matplotlib.dates as mdates
date_nums = mdates.date2num([datetime.datetime(d.year, d.month, d.day) for d in dates])

# ── 3. Also pick 2 "normal" pixels (valid in s, L, ET, DP all) for comparison ─
normal = s_mask & L_mask & et_mask & dp_mask
normal_idx = list(zip(*np.where(normal)))
normal_selected = normal_idx[:2]
print(f"Normal pixels (all valid): {len(normal_idx)}")
print(f"Selected normal pixels: {normal_selected}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
n_anom   = len(selected)
n_normal = len(normal_selected)
n_total  = n_anom + n_normal

fig, axes = plt.subplots(n_total, 3, figsize=(18, 3.5 * n_total), sharex=True)
fig.suptitle("Time series: s, L, T from olivo_surface (shallow_* files)\n"
             "ANOMALOUS = BP valid in WATNEEDS but ET=NaN and DP=0  |  NORMAL = valid in all variables",
             fontsize=12, fontweight="bold")

VAR_STYLES = {
    "s": {"color": "steelblue",  "label": "Soil moisture s [-]",     "ylim": (0, 1)},
    "L": {"color": "seagreen",   "label": "Leaching L [mm/step]",    "ylim": None},
    "T": {"color": "darkorange", "label": "Transpiration T [mm/step]","ylim": None},
}

def plot_pixel(axes_row, ii, jj, tag):
    s_px = s_full[ii, jj, :]
    L_px = L_full[ii, jj, :]
    T_px = T_full[ii, jj, :]

    for ax, (var, arr) in zip(axes_row, [("s", s_px), ("L", L_px), ("T", T_px)]):
        st = VAR_STYLES[var]
        nan_frac = np.isnan(arr).mean() * 100
        pos_frac = np.nansum(arr > 0) / len(arr) * 100

        ax.plot(date_nums, arr, color=st["color"], linewidth=0.4, alpha=0.8)
        ax.set_ylabel(st["label"], fontsize=8)
        if st["ylim"]:
            ax.set_ylim(st["ylim"])
        ax.set_title(f"{tag}  pixel ({ii},{jj})  |  NaN={nan_frac:.1f}%  pos={pos_frac:.1f}%",
                     fontsize=8.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.4)

        # Annotate stats
        ax.text(0.01, 0.97,
                f"mean={np.nanmean(arr):.4f}\nmax={np.nanmax(arr):.4f}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

for k, (ii, jj) in enumerate(selected):
    plot_pixel(axes[k], ii, jj, "ANOMALOUS")

for k, (ii, jj) in enumerate(normal_selected):
    plot_pixel(axes[n_anom + k], ii, jj, "NORMAL")

plt.tight_layout()
out_png = OUT_DIR / "anomalous_pixel_timeseries.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_png}")

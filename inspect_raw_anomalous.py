"""
Plot raw time series of BP, ET, DP from giornalieri_SURF_5crops
for anomalous pixels: BP valid but ET=all-NaN or DP=all-zero.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import h5py
import re
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Latitude 5511\Downloads\giornalieri_SURF_5crops\giornalieri_SURF_5crops")
OUT_DIR  = Path(r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW")
CROP = 2
SR, SC, TH, TW = 1, 1, 39, 43   # crop offsets after .T

# ── 1. Identify anomalous pixels ──────────────────────────────────────────────
bp_mask = np.load(str(OUT_DIR / "valid_mask_BP_crop2_SURF_v2.npy"))
et_mask = np.load(str(OUT_DIR / "valid_mask_ET_crop2_SURF.npy"))
dp_mask = np.load(str(OUT_DIR / "valid_mask_DP_crop2_SURF.npy"))

anom_et = bp_mask & ~et_mask   # BP ok, ET=NaN
anom_dp = bp_mask & ~dp_mask   # BP ok, DP=0

# Pick one pixel of each type + one normal pixel
et_pix   = list(zip(*np.where(anom_et)))[0]    # first ET-NaN anomalous
dp_pix   = list(zip(*np.where(anom_dp & et_mask)))[0]  # DP=0 but ET valid
norm_pix = list(zip(*np.where(bp_mask & et_mask & dp_mask)))[0]  # fully normal

print(f"ET-NaN pixel : {et_pix}")
print(f"DP-zero pixel: {dp_pix}")
print(f"Normal pixel : {norm_pix}")

pixels = {
    f"ET=NaN  pixel {et_pix}":   et_pix,
    f"DP=0    pixel {dp_pix}":   dp_pix,
    f"Normal  pixel {norm_pix}": norm_pix,
}

# ── 2. Load time series for all three variables ───────────────────────────────
def load_var(var):
    key     = f"output{var}"
    pattern = re.compile(rf"^output{var}_(\d{{4}})_(\d{{1,2}})_{CROP}\.mat$")
    files   = sorted(
        [(int(m.group(1)), int(m.group(2)), f)
         for f in DATA_DIR.iterdir()
         for m in [pattern.match(f.name)] if m],
        key=lambda x: (x[0], x[1])
    )
    arrays, dates = [], []
    for year, month, fpath in files:
        with h5py.File(str(fpath), "r") as f:
            raw = f[key][:]                          # (days, 45, 41)
        cropped = raw.T[SR:SR+TH, SC:SC+TW, :]      # (39, 43, days)
        arrays.append(cropped)
        import calendar
        for d in range(calendar.monthrange(year, month)[1]):
            dates.append(datetime.date(year, month, d + 1))
    return np.concatenate(arrays, axis=2), dates

print("Loading BP, ET, DP...")
bp_full, dates = load_var("BP")
et_full, _     = load_var("ET")
dp_full, _     = load_var("DP")
print(f"Shape: {bp_full.shape}  ({len(dates)} days)")

date_nums = mdates.date2num(
    [datetime.datetime(d.year, d.month, d.day) for d in dates]
)

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(pixels), 3, figsize=(18, 4 * len(pixels)), sharex=True)
fig.suptitle(
    "Raw daily time series from giornalieri_SURF_5crops  |  crop 2 (olivo)\n"
    "BP = total precip+irrig  |  ET = evapotranspiration  |  DP = deep percolation",
    fontsize=12, fontweight="bold"
)

VAR_META = {
    "BP": ("steelblue",  "BP [mm/d]"),
    "ET": ("darkorange", "ET [mm/d]"),
    "DP": ("seagreen",   "DP [mm/d]"),
}

for row, (label, (ii, jj)) in enumerate(pixels.items()):
    for col, (var, full) in enumerate(
            [("BP", bp_full), ("ET", et_full), ("DP", dp_full)]):
        ax  = axes[row, col]
        arr = full[ii, jj, :]
        color, ylabel = VAR_META[var]

        nan_pct  = np.isnan(arr).mean() * 100
        zero_pct = (arr == 0).mean() * 100 if not np.all(np.isnan(arr)) else 0
        pos_pct  = (np.nansum(arr > 0) / len(arr)) * 100

        ax.plot(date_nums, arr, color=color, linewidth=0.5, alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.4)

        stats = (f"NaN={nan_pct:.0f}%  zero={zero_pct:.0f}%  pos={pos_pct:.0f}%\n"
                 f"mean={np.nanmean(arr):.3f}  max={np.nanmax(arr):.3f}")
        ax.text(0.01, 0.97, stats, transform=ax.transAxes,
                fontsize=7, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))

        if col == 0:
            ax.set_title(label, fontsize=9, fontweight="bold", loc="left")

plt.tight_layout()
out = OUT_DIR / "raw_anomalous_timeseries.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight")
print(f"Figure saved: {out}")

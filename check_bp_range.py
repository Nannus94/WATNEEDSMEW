"""
Check BP values across all valid pixels — should be in [0,1] if it's
soil moisture as fraction of max saturation.
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
SR, SC, TH, TW = 1, 1, 39, 43

bp_mask = np.load(str(OUT_DIR / "valid_mask_BP_crop2_SURF_v2.npy"))
valid_idx = list(zip(*np.where(bp_mask)))

# ── 1. Quick global stats: scan all files, track max/min per pixel ────────────
print("Scanning all BP files for global range...")
bp_max = np.full((TH, TW), -np.inf)
bp_min = np.full((TH, TW),  np.inf)

pattern = re.compile(rf"^outputBP_(\d{{4}})_(\d{{1,2}})_{CROP}\.mat$")
files = sorted(
    [(int(m.group(1)), int(m.group(2)), f)
     for f in DATA_DIR.iterdir()
     for m in [pattern.match(f.name)] if m],
    key=lambda x: (x[0], x[1])
)

for i, (year, month, fpath) in enumerate(files):
    with h5py.File(str(fpath), "r") as f:
        raw = f["outputBP"][:]
    c = raw.T[SR:SR+TH, SC:SC+TW, :]
    bp_max = np.fmax(bp_max, np.nanmax(c, axis=2))
    bp_min = np.fmin(bp_min, np.nanmin(c, axis=2))
    if (i+1) % 120 == 0:
        print(f"  {i+1}/{len(files)} files...", flush=True)

# Report global stats over valid pixels only
valid_max = bp_max[bp_mask]
valid_min = bp_min[bp_mask]
print(f"\nOver {bp_mask.sum()} valid pixels:")
print(f"  Global min  : {valid_min.min():.4f}")
print(f"  Global max  : {valid_max.max():.4f}")
print(f"  Mean of maxs: {valid_max.mean():.4f}")
print(f"  Pixels with max > 1.0: {(valid_max > 1.0).sum()}")
print(f"  Pixels with max > 1.5: {(valid_max > 1.5).sum()}")
print(f"  Pixels with max > 10 : {(valid_max > 10).sum()}")
print(f"  Pixels with max > 100: {(valid_max > 100).sum()}")

# ── 2. Load full time series for 6 spread-out normal pixels ──────────────────
print("\nLoading full BP time series for selected pixels...")
arrays, dates = [], []
import calendar
for year, month, fpath in files:
    with h5py.File(str(fpath), "r") as f:
        raw = f["outputBP"][:]
    arrays.append(raw.T[SR:SR+TH, SC:SC+TW, :])
    for d in range(calendar.monthrange(year, month)[1]):
        dates.append(datetime.date(year, month, d+1))

bp_full = np.concatenate(arrays, axis=2)
date_nums = mdates.date2num(
    [datetime.datetime(d.year, d.month, d.day) for d in dates]
)

# Pick 6 normal pixels spread across the valid set (exclude (6,42))
et_mask = np.load(str(OUT_DIR / "valid_mask_ET_crop2_SURF.npy"))
dp_mask = np.load(str(OUT_DIR / "valid_mask_DP_crop2_SURF.npy"))
normal  = bp_mask & et_mask & dp_mask
norm_idx = [px for px in list(zip(*np.where(normal))) if px != (6, 42)]
step = max(1, len(norm_idx) // 6)
selected = norm_idx[::step][:6]
print(f"Selected pixels: {selected}")

fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
fig.suptitle("BP (soil moisture %) — raw daily values, 6 normal pixels\n"
             "giornalieri_SURF_5crops  |  crop 2 (olivo)",
             fontsize=12, fontweight="bold")

for ax, (ii, jj) in zip(axes.flatten(), selected):
    arr = bp_full[ii, jj, :]
    ax.plot(date_nums, arr, color="steelblue", linewidth=0.5, alpha=0.85)
    ax.axhline(1.0, color="red", linewidth=0.8, linestyle="--", label="y=1")
    ax.set_title(f"pixel ({ii},{jj})", fontsize=9)
    ax.set_ylabel("BP [-]", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.text(0.01, 0.97,
            f"min={np.nanmin(arr):.3f}  max={np.nanmax(arr):.3f}\n"
            f"mean={np.nanmean(arr):.3f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))
    ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
out = OUT_DIR / "bp_range_check.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight")
print(f"Figure saved: {out}")

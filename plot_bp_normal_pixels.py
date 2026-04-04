"""
Plot BP time series for several "normal" pixels (valid in BP, ET, DP).
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import h5py
import re
import datetime
import calendar
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Latitude 5511\Downloads\giornalieri_SURF_5crops\giornalieri_SURF_5crops")
OUT_DIR  = Path(r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW")
CROP = 2
SR, SC, TH, TW = 1, 1, 39, 43

bp_mask = np.load(str(OUT_DIR / "valid_mask_BP_crop2_SURF_v2.npy"))
et_mask = np.load(str(OUT_DIR / "valid_mask_ET_crop2_SURF.npy"))
dp_mask = np.load(str(OUT_DIR / "valid_mask_DP_crop2_SURF.npy"))

# Normal = valid in all three
normal = bp_mask & et_mask & dp_mask
norm_idx = list(zip(*np.where(normal)))
print(f"Normal pixels (BP+ET+DP valid): {len(norm_idx)}")

# Pick 6 spread across the set
step = max(1, len(norm_idx) // 6)
selected = norm_idx[::step][:6]
print(f"Selected: {selected}")

# Load BP
pattern = re.compile(rf"^outputBP_(\d{{4}})_(\d{{1,2}})_{CROP}\.mat$")
files = sorted(
    [(int(m.group(1)), int(m.group(2)), f)
     for f in DATA_DIR.iterdir()
     for m in [pattern.match(f.name)] if m],
    key=lambda x: (x[0], x[1])
)
arrays, dates = [], []
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

fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
fig.suptitle("BP — normal pixels (valid in BP, ET, DP)\ngiornalieri_SURF_5crops  |  crop 2 (olivo)",
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
    ax.text(0.01, 0.97, f"min={np.nanmin(arr):.3f}  max={np.nanmax(arr):.3f}  mean={np.nanmean(arr):.3f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))
    ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
out = OUT_DIR / "bp_normal_pixels.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight")
print(f"Figure saved: {out}")

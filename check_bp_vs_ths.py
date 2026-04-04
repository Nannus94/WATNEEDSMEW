"""
Verify BP interpretation: compare BP values against ths (max saturation)
and fc (field capacity) maps, after applying the same crop+offset pipeline.

BP should be s/ths (relative saturation), so:
  BP = 1.0  -> fully saturated
  BP > 1.0  -> temporary ponding/oversaturation (physically possible but brief)
  BP ~ fc/ths -> typical field conditions
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import h5py
import rasterio
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

# ── 1. Load ths and fc maps (39x43 Sicily grid) ───────────────────────────────
with rasterio.open(r"C:\Users\Latitude 5511\Downloads\ths_sicilia (3).tif") as src:
    ths_raw = src.read(1).astype(float)
    nd = src.nodata
    if nd is not None: ths_raw[ths_raw == nd] = np.nan
with rasterio.open(r"C:\Users\Latitude 5511\Downloads\fc_sicilia (2).tif") as src:
    fc_raw  = src.read(1).astype(float)
    nd = src.nodata
    if nd is not None: fc_raw[fc_raw == nd] = np.nan

print(f"ths raw shape: {ths_raw.shape}  range: [{np.nanmin(ths_raw):.3f}, {np.nanmax(ths_raw):.3f}]")
print(f"fc  raw shape: {fc_raw.shape}   range: [{np.nanmin(fc_raw):.3f}, {np.nanmax(fc_raw):.3f}]")

# ths/fc are (41,45) — same raw WATNEEDS grid, need same .T + crop as BP
# After .T: (45, 41) -> crop center to (39, 43): sr=(45-39)//2=3, sc=(41-43)//2=-1
# But BP pipeline: h5py gives (days,45,41), .T gives (41,45,days), crop sr=1,sc=1
# ths/fc are 2D spatial: shape (41,45) already in (lon,lat) order -> crop sr=1,sc=1
ths_sr = (ths_raw.shape[0] - TH) // 2   # (41-39)//2 = 1
ths_sc = (ths_raw.shape[1] - TW) // 2   # (45-43)//2 = 1
ths = ths_raw[ths_sr:ths_sr+TH, ths_sc:ths_sc+TW]
fc  = fc_raw[ths_sr:ths_sr+TH, ths_sc:ths_sc+TW]

print(f"ths cropped: {ths.shape}  range: [{np.nanmin(ths):.3f}, {np.nanmax(ths):.3f}]")
print(f"fc  cropped: {fc.shape}   range: [{np.nanmin(fc):.3f}, {np.nanmax(fc):.3f}]")

# fc/ths ratio = relative field capacity (expected typical BP value)
fc_rel = fc / ths
print(f"fc/ths range: [{np.nanmin(fc_rel):.3f}, {np.nanmax(fc_rel):.3f}]  "
      f"mean={np.nanmean(fc_rel):.3f}")

# ── 2. Load BP full time series ───────────────────────────────────────────────
bp_mask = np.load(str(OUT_DIR / "valid_mask_BP_crop2_SURF_v2.npy"))
et_mask = np.load(str(OUT_DIR / "valid_mask_ET_crop2_SURF.npy"))
dp_mask = np.load(str(OUT_DIR / "valid_mask_DP_crop2_SURF.npy"))

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
print(f"\nBP loaded: {bp_full.shape}")

# ── 3. Select pixels: 3 normal + 1 high-max pixel ────────────────────────────
normal = bp_mask & et_mask & dp_mask
norm_idx = list(zip(*np.where(normal)))

# Find pixel with highest max BP (most likely to exceed 1)
bp_pixmax = np.array([np.nanmax(bp_full[ii, jj, :]) for ii, jj in norm_idx])
sorted_by_max = [norm_idx[i] for i in np.argsort(bp_pixmax)[::-1]]

# Pick: highest max, median max, lowest max, one random middle
selected = {
    "highest max BP": sorted_by_max[0],
    "upper-mid max BP": sorted_by_max[len(sorted_by_max)//4],
    "median max BP": sorted_by_max[len(sorted_by_max)//2],
    "lowest max BP": sorted_by_max[-1],
}

print("\nSelected pixels:")
for label, (ii, jj) in selected.items():
    bp_max_px = np.nanmax(bp_full[ii, jj, :])
    ths_px    = ths[ii, jj] if ths.shape == (TH, TW) else np.nan
    fc_px     = fc[ii, jj]  if fc.shape  == (TH, TW) else np.nan
    fc_rel_px = fc_px / ths_px if not np.isnan(ths_px) else np.nan
    print(f"  {label}: ({ii},{jj})  BP_max={bp_max_px:.3f}  "
          f"ths={ths_px:.3f}  fc={fc_px:.3f}  fc/ths={fc_rel_px:.3f}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(selected), 1, figsize=(16, 3.5 * len(selected)),
                          sharex=True)
fig.suptitle("BP (relative soil moisture) vs ths and fc thresholds\n"
             "giornalieri_SURF_5crops  |  crop 2 (olivo)",
             fontsize=12, fontweight="bold")

for ax, (label, (ii, jj)) in zip(axes, selected.items()):
    arr = bp_full[ii, jj, :]

    ths_px    = ths[ii, jj] if ths.shape == (TH, TW) else np.nan
    fc_px     = fc[ii, jj]  if fc.shape  == (TH, TW) else np.nan
    fc_rel_px = fc_px / ths_px if not np.isnan(ths_px) else np.nan

    ax.plot(date_nums, arr, color="steelblue", linewidth=0.5, alpha=0.85,
            label="BP (s/ths)")
    ax.axhline(1.0, color="red",    linewidth=1.0, linestyle="--",
               label="ths (saturation = 1.0)")
    if not np.isnan(fc_rel_px):
        ax.axhline(fc_rel_px, color="orange", linewidth=1.0, linestyle=":",
                   label=f"fc/ths = {fc_rel_px:.3f}")

    ax.set_ylabel("BP [-]", fontsize=8)
    ax.set_title(f"{label}  |  pixel ({ii},{jj})  "
                 f"ths={ths_px:.3f}  fc={fc_px:.3f}  "
                 f"BP_max={np.nanmax(arr):.3f}  BP_mean={np.nanmean(arr):.3f}",
                 fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.legend(fontsize=7, loc="upper right", ncol=3)

    # Shade oversaturation region
    ax.fill_between(date_nums, 1.0, arr,
                    where=(arr > 1.0),
                    color="red", alpha=0.15, label="_nolegend_")

plt.tight_layout()
out = OUT_DIR / "bp_vs_ths_fc.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out}")

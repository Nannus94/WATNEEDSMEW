"""
Recompute BP crop-2 valid pixel mask with the CORRECT pipeline from New_WB.py:
  1. h5py load  -> shape (days, 45, 41)
  2. .T          -> shape (41, 45, days)   [lon, lat, time]
  3. crop_center -> shape (39, 43)          [lat=tif_h, lon=tif_w]
  4. georeference with sicilia10km.tif

Valid pixel = no NaN at any timestep AND at least once positive.
RAM-safe: accumulates boolean masks only.
"""

import re
import matplotlib
matplotlib.use("Agg")
import h5py
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Latitude 5511\Downloads\giornalieri_SURF_5crops\giornalieri_SURF_5crops")
OUT_DIR  = Path(r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW")
REF_TIF  = r"C:\Users\Latitude 5511\Downloads\sicilia10km.tif"
PH_TIF   = r"C:\Users\Latitude 5511\Downloads\FULL_AGRUMI_DRIP_30Y_pH_map_MERGED.tif"

VAR  = "BP"
CROP = 2
KEY  = f"output{VAR}"

# ── 1. Reference grid ─────────────────────────────────────────────────────────
with rasterio.open(REF_TIF) as src:
    ref_transform = src.transform
    ref_crs       = src.crs
    TIF_H, TIF_W  = src.shape          # (39, 43)
    ref_data      = src.read(1).astype(float)

print(f"Reference grid: {TIF_H} x {TIF_W}, CRS={ref_crs}")
print(f"Transform: {ref_transform}")

# ── 2. Discover and sort files ────────────────────────────────────────────────
pattern = re.compile(rf"^output{VAR}_(\d{{4}})_(\d{{1,2}})_{CROP}\.mat$")
files = []
for f in DATA_DIR.iterdir():
    m = pattern.match(f.name)
    if m:
        files.append((int(m.group(1)), int(m.group(2)), f))
files.sort(key=lambda x: (x[0], x[1]))
print(f"Found {len(files)} files for {VAR} crop {CROP}")

# ── 3. Peek at raw shape and compute crop offsets ─────────────────────────────
with h5py.File(str(files[0][2]), "r") as f:
    raw_sample = f[KEY][:]          # (days, 45, 41)
# After .T -> (41, 45, days); spatial: (41, 45) = (lon_dim, lat_dim)
lon_dim, lat_dim = raw_sample.shape[2], raw_sample.shape[1]   # 41, 45
sr = (lon_dim - TIF_H) // 2    # (41-39)//2 = 1  — rows in transposed = lon
sc = (lat_dim - TIF_W) // 2    # (45-43)//2 = 1  — cols in transposed = lat
print(f"Raw (lon_dim={lon_dim}, lat_dim={lat_dim}) -> crop offsets sr={sr}, sc={sc}")

# ── 4. Incremental mask accumulation ─────────────────────────────────────────
has_nan = np.zeros((TIF_H, TIF_W), dtype=bool)
any_pos = np.zeros((TIF_H, TIF_W), dtype=bool)
total_steps = 0

for i, (year, month, fpath) in enumerate(files):
    with h5py.File(str(fpath), "r") as f:
        raw = f[KEY][:]                      # (days, 45, 41)
    data_T = raw.T                           # (41, 45, days)
    cropped = data_T[sr:sr+TIF_H, sc:sc+TIF_W, :]   # (39, 43, days)

    has_nan |= np.any(np.isnan(cropped), axis=2)
    any_pos |= np.any(cropped > 0,       axis=2)
    total_steps += cropped.shape[2]

    if (i + 1) % 60 == 0:
        print(f"  processed {i+1}/{len(files)} files ({year}-{month:02d})...", flush=True)

print(f"\nTotal time steps: {total_steps}")

valid_mask = (~has_nan) & any_pos
n_valid = int(valid_mask.sum())
n_total = valid_mask.size
print(f"Valid pixels: {n_valid} / {n_total}  ({100*n_valid/n_total:.1f} %)")

# ── 5. Save mask as npy ───────────────────────────────────────────────────────
np.save(str(OUT_DIR / f"valid_mask_{VAR}_crop{CROP}_SURF_v2.npy"), valid_mask)
print("Mask (npy) saved.")

# ── 6. Save as GeoTIFF ────────────────────────────────────────────────────────
out_tif = OUT_DIR / f"valid_mask_{VAR}_crop{CROP}_SURF_v2.tif"
with rasterio.open(
    str(out_tif), "w",
    driver="GTiff", height=TIF_H, width=TIF_W,
    count=1, dtype=rasterio.uint8,
    crs=ref_crs, transform=ref_transform,
    nodata=255, compress="lzw",
) as dst:
    dst.write(valid_mask.astype(np.uint8), 1)
print(f"GeoTIFF saved: {out_tif}")

# ── 7. Plot with cartopy ──────────────────────────────────────────────────────
with rasterio.open(PH_TIF) as src:
    ph_data = src.read(1).astype(float)
    ph_data[ph_data <= 0] = np.nan
    bounds = src.bounds

extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                          subplot_kw={"projection": ccrs.PlateCarree()})
fig.suptitle(
    f"Valid pixels  |  {VAR}  crop {CROP}  |  giornalieri SURF\n"
    "Criterion: no NaN at any timestep  AND  at least once positive",
    fontsize=12, fontweight="bold"
)

def add_geo_features(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"),   linewidth=0.5, linestyle=":")
    ax.add_feature(cfeature.LAND.with_scale("10m"),      facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"),     facecolor="#d0e8f5", zorder=0)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5)
    gl.top_labels = gl.right_labels = False
    return gl

# Panel 1: valid mask
ax = axes[0]
ax.set_extent(extent, crs=ccrs.PlateCarree())
add_geo_features(ax)

valid_color = plt.get_cmap("Blues")(0.65)
cmap2 = mcolors.ListedColormap(["#cccccc", valid_color])
ax.imshow(valid_mask.astype(float),
          origin="upper", extent=extent,
          cmap=cmap2, vmin=0, vmax=1,
          interpolation="nearest",
          transform=ccrs.PlateCarree(), zorder=2)

ax.set_title(f"{VAR} crop {CROP} — valid pixels\n"
             f"{n_valid}/{n_total} ({100*n_valid/n_total:.1f} %)", fontsize=10)
ax.legend(handles=[
    mpatches.Patch(facecolor="#cccccc", edgecolor="grey", label="Invalid"),
    mpatches.Patch(facecolor=valid_color, edgecolor="grey", label="Valid"),
], loc="lower right", fontsize=8, framealpha=0.85)

# Panel 2: pH reference
ax2 = axes[1]
ax2.set_extent(extent, crs=ccrs.PlateCarree())
add_geo_features(ax2)
im = ax2.imshow(ph_data,
                origin="upper", extent=extent,
                cmap="RdYlGn", vmin=5.5, vmax=8.5,
                interpolation="nearest",
                transform=ccrs.PlateCarree(), zorder=2)
plt.colorbar(im, ax=ax2, label="pH (CaCl2)", fraction=0.046, pad=0.04)
ax2.set_title("Reference: pH map (agrumi drip)", fontsize=10)

plt.tight_layout()
out_png = OUT_DIR / f"valid_pixels_{VAR}_crop{CROP}_SURF_v2.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
print(f"Figure saved: {out_png}")

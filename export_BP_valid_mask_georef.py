"""
Export the BP crop-2 valid pixel mask as a georeferenced GeoTIFF,
using the same projection as the pH maps (EPSG:4326, 10km grid).

The raw WATNEEDS output is (45, 41) in row-major (lat, lon) order.
It is center-cropped to (39, 43) to match the Sicily domain.
The final grid shares the same georeference as all other soil/pH maps.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

OUT_DIR  = Path(r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW")
REF_TIF  = r"C:\Users\Latitude 5511\Downloads\FULL_AGRUMI_DRIP_30Y_pH_map_MERGED.tif"

VAR   = "BP"
CROP  = 2

# ── 1. Load raw mask (45, 41) ─────────────────────────────────────────────────
mask_raw = np.load(str(OUT_DIR / f"valid_mask_{VAR}_crop{CROP}_SURF.npy"))
print(f"Raw mask shape: {mask_raw.shape}")   # expect (45, 41)

# ── 2. Center-crop to (39, 43) — same as build_infiltration_files.py ─────────
TARGET = (39, 43)
r, c   = mask_raw.shape
dr     = (r - TARGET[0]) // 2   # (45-39)//2 = 3
dc     = (c - TARGET[1]) // 2   # (41-43)//2 = -1  → dc=0, take all cols + pad
print(f"  dr={dr}, dc={dc}")

if dc < 0:
    # More target cols than source cols — pad with False
    pad = TARGET[1] - c
    pad_l = pad // 2
    pad_r = pad - pad_l
    cropped = mask_raw[dr:dr+TARGET[0], :]
    cropped = np.pad(cropped, ((0,0),(pad_l,pad_r)), constant_values=False)
    print(f"  Padded cols: left={pad_l}, right={pad_r}")
else:
    cropped = mask_raw[dr:dr+TARGET[0], dc:dc+TARGET[1]]

print(f"Cropped mask shape: {cropped.shape}")
assert cropped.shape == TARGET, f"Expected {TARGET}, got {cropped.shape}"

n_valid = int(cropped.sum())
n_total = cropped.size
print(f"Valid pixels after crop: {n_valid} / {n_total} ({100*n_valid/n_total:.1f} %)")

# ── 3. Read georeference from pH reference tif ────────────────────────────────
with rasterio.open(REF_TIF) as src:
    ref_transform = src.transform
    ref_crs       = src.crs
    ref_nodata    = 0
    print(f"Reference CRS: {ref_crs}")
    print(f"Reference transform: {ref_transform}")
    print(f"Reference shape: {src.shape}")

# ── 4. Save as GeoTIFF ────────────────────────────────────────────────────────
out_tif = OUT_DIR / f"valid_mask_{VAR}_crop{CROP}_SURF_georef.tif"
with rasterio.open(
    str(out_tif),
    "w",
    driver="GTiff",
    height=TARGET[0],
    width=TARGET[1],
    count=1,
    dtype=rasterio.uint8,
    crs=ref_crs,
    transform=ref_transform,
    nodata=255,
    compress="lzw",
) as dst:
    dst.write(cropped.astype(np.uint8), 1)

print(f"\nGeoTIFF saved: {out_tif}")

# ── 5. Plot with cartopy (same style as pH maps) ──────────────────────────────
# Read pH map for comparison overlay
with rasterio.open(REF_TIF) as src:
    ph_data = src.read(1).astype(float)
    ph_data[ph_data <= 0] = np.nan
    bounds = src.bounds   # left, bottom, right, top

extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                          subplot_kw={"projection": ccrs.PlateCarree()})
fig.suptitle(
    f"Valid pixels  |  {VAR}  crop {CROP}  |  giornalieri SURF\n"
    "Criterion: no NaN at any timestep  AND  at least one positive value",
    fontsize=12, fontweight="bold"
)

# ---- Panel 1: Valid mask ----
ax = axes[0]
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle=":")
ax.add_feature(cfeature.LAND,      facecolor="#f5f5f0", zorder=0)
ax.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5", zorder=0)

base_cmap  = plt.get_cmap("Blues")
valid_color = base_cmap(0.65)
cmap2 = mcolors.ListedColormap(["#cccccc", valid_color])

ax.imshow(cropped.astype(float),
          origin="upper", extent=extent,
          cmap=cmap2, vmin=0, vmax=1,
          interpolation="nearest",
          transform=ccrs.PlateCarree(), zorder=2)

gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5)
gl.top_labels = gl.right_labels = False
ax.set_title(f"{VAR} crop {CROP} — valid pixels\n"
             f"{n_valid}/{n_total} ({100*n_valid/n_total:.1f} %)", fontsize=10)

legend_elements = [
    mpatches.Patch(facecolor="#cccccc", edgecolor="grey", label="Invalid"),
    mpatches.Patch(facecolor=valid_color, edgecolor="grey", label="Valid"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.85)

# ---- Panel 2: pH reference map ----
ax2 = axes[1]
ax2.set_extent(extent, crs=ccrs.PlateCarree())
ax2.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax2.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle=":")
ax2.add_feature(cfeature.LAND,      facecolor="#f5f5f0", zorder=0)
ax2.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5", zorder=0)

im = ax2.imshow(ph_data,
                origin="upper", extent=extent,
                cmap="RdYlGn", vmin=5.5, vmax=8.5,
                interpolation="nearest",
                transform=ccrs.PlateCarree(), zorder=2)
plt.colorbar(im, ax=ax2, label="pH (CaCl2)", fraction=0.046, pad=0.04)
gl2 = ax2.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5)
gl2.top_labels = gl2.right_labels = False
ax2.set_title("Reference: pH map (agrumi drip)", fontsize=10)

plt.tight_layout()
out_png = OUT_DIR / f"valid_pixels_{VAR}_crop{CROP}_SURF_georef.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
print(f"Figure saved: {out_png}")

"""
Valid pixel masks for ET (transpiration) and DP (deep percolation / leaching),
crop 2, giornalieri_SURF_5crops.

Pipeline (from New_WB.py):
  h5py load  -> (days, 45, 41)
  .T          -> (41, 45, days)   [lon_dim=41, lat_dim=45]
  crop_center -> (39, 43, days)   sr=1, sc=1

Valid pixel = no NaN at any timestep AND at least once positive (> 0).
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

CROP = 2
VARS = {
    "ET": "Transpiration / ET",
    "DP": "Deep Percolation (Leaching)",
}

# ── Reference grid ────────────────────────────────────────────────────────────
with rasterio.open(REF_TIF) as src:
    ref_transform = src.transform
    ref_crs       = src.crs
    TIF_H, TIF_W  = src.shape          # (39, 43)

print(f"Reference grid: {TIF_H} x {TIF_W}, CRS={ref_crs}")

# Crop offsets (same for all vars — raw spatial is always 41x45 after .T)
LON_DIM, LAT_DIM = 41, 45
SR = (LON_DIM - TIF_H) // 2   # (41-39)//2 = 1
SC = (LAT_DIM - TIF_W) // 2   # (45-43)//2 = 1
print(f"Crop offsets: sr={SR}, sc={SC}\n")

# ── Process each variable ─────────────────────────────────────────────────────
masks = {}

for VAR, VAR_LABEL in VARS.items():
    KEY = f"output{VAR}"
    pattern = re.compile(rf"^output{VAR}_(\d{{4}})_(\d{{1,2}})_{CROP}\.mat$")

    files = []
    for f in DATA_DIR.iterdir():
        m = pattern.match(f.name)
        if m:
            files.append((int(m.group(1)), int(m.group(2)), f))
    files.sort(key=lambda x: (x[0], x[1]))
    print(f"[{VAR}] Found {len(files)} files")

    has_nan = np.zeros((TIF_H, TIF_W), dtype=bool)
    any_pos = np.zeros((TIF_H, TIF_W), dtype=bool)
    total_steps = 0

    for i, (year, month, fpath) in enumerate(files):
        with h5py.File(str(fpath), "r") as f:
            raw = f[KEY][:]                          # (days, 45, 41)
        data_T  = raw.T                              # (41, 45, days)
        cropped = data_T[SR:SR+TIF_H, SC:SC+TIF_W, :]  # (39, 43, days)

        has_nan |= np.any(np.isnan(cropped), axis=2)
        any_pos |= np.any(cropped > 0,       axis=2)
        total_steps += cropped.shape[2]

        if (i + 1) % 60 == 0:
            print(f"  [{VAR}] {i+1}/{len(files)} files ({year}-{month:02d})...", flush=True)

    valid = (~has_nan) & any_pos
    n_valid = int(valid.sum())
    n_total = valid.size
    print(f"  [{VAR}] Total steps: {total_steps}")
    print(f"  [{VAR}] Valid pixels: {n_valid} / {n_total} ({100*n_valid/n_total:.1f} %)\n")

    masks[VAR] = {"mask": valid, "label": VAR_LABEL,
                  "n_valid": n_valid, "n_total": n_total}

    # Save npy
    np.save(str(OUT_DIR / f"valid_mask_{VAR}_crop{CROP}_SURF.npy"), valid)

    # Save GeoTIFF
    out_tif = OUT_DIR / f"valid_mask_{VAR}_crop{CROP}_SURF.tif"
    with rasterio.open(
        str(out_tif), "w",
        driver="GTiff", height=TIF_H, width=TIF_W,
        count=1, dtype=rasterio.uint8,
        crs=ref_crs, transform=ref_transform,
        nodata=255, compress="lzw",
    ) as dst:
        dst.write(valid.astype(np.uint8), 1)
    print(f"  [{VAR}] GeoTIFF saved: {out_tif}")

# ── Plot: ET | DP | pH reference ──────────────────────────────────────────────
with rasterio.open(PH_TIF) as src:
    ph_data = src.read(1).astype(float)
    ph_data[ph_data <= 0] = np.nan
    bounds = src.bounds

extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

COLORS = {"ET": "Oranges", "DP": "Greens"}

fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                          subplot_kw={"projection": ccrs.PlateCarree()})
fig.suptitle(
    f"Valid pixels  |  crop {CROP}  |  giornalieri SURF\n"
    "Criterion: no NaN at any timestep  AND  at least once positive",
    fontsize=12, fontweight="bold"
)

def add_geo(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"),   linewidth=0.5, linestyle=":")
    ax.add_feature(cfeature.LAND.with_scale("10m"),      facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"),     facecolor="#d0e8f5", zorder=0)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5)
    gl.top_labels = gl.right_labels = False

for ax, (VAR, info) in zip(axes[:2], masks.items()):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    add_geo(ax)

    vc = plt.get_cmap(COLORS[VAR])(0.65)
    cmap2 = mcolors.ListedColormap(["#cccccc", vc])
    ax.imshow(info["mask"].astype(float),
              origin="upper", extent=extent,
              cmap=cmap2, vmin=0, vmax=1,
              interpolation="nearest",
              transform=ccrs.PlateCarree(), zorder=2)

    ax.set_title(f"{VAR} — {info['label']}\n"
                 f"{info['n_valid']}/{info['n_total']} "
                 f"({100*info['n_valid']/info['n_total']:.1f} %)", fontsize=10)
    ax.legend(handles=[
        mpatches.Patch(facecolor="#cccccc", edgecolor="grey", label="Invalid"),
        mpatches.Patch(facecolor=vc,        edgecolor="grey", label="Valid"),
    ], loc="lower right", fontsize=8, framealpha=0.85)

# pH reference panel
ax3 = axes[2]
ax3.set_extent(extent, crs=ccrs.PlateCarree())
add_geo(ax3)
im = ax3.imshow(ph_data,
                origin="upper", extent=extent,
                cmap="RdYlGn", vmin=5.5, vmax=8.5,
                interpolation="nearest",
                transform=ccrs.PlateCarree(), zorder=2)
plt.colorbar(im, ax=ax3, label="pH (CaCl2)", fraction=0.046, pad=0.04)
ax3.set_title("Reference: pH map", fontsize=10)

plt.tight_layout()
out_png = OUT_DIR / f"valid_pixels_ET_DP_crop{CROP}_SURF.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_png}")

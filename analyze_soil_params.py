"""
Valid pixel maps for all soil parameter GeoTIFFs in soil_param (1)/.

For static rasters (single timestep), valid = finite AND non-NaN AND non-zero.
Produces:
  - one GeoTIFF per layer  (valid_mask_<name>_soilparam.tif)
  - one summary PNG with all maps in a grid
"""

import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

SOIL_DIR = Path(r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\soil_param (1)")
OUT_DIR  = Path(r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW")
REF_TIF  = r"C:\Users\Latitude 5511\Downloads\sicilia10km.tif"

# Skip duplicates — keep the canonical name
SKIP = {"r_het_Sic_10km_resampled2 (1).tif",
        "cec_sicily_masked_10km_pH (1).tif",
        "soc_sicily_masked_10km_pH (1).tif"}

# Human-readable labels
LABELS = {
    "sicily_ph_cacl2_10km.tif":           ("pH",        "pH (CaCl2)",              "RdYlGn",  5.5, 8.0),
    "bdod_sicily_masked_10km_pH.tif":     ("BD",        "Bulk density [dg/cm3]",   "YlOrBr",  10,  16),
    "cec_sicily_masked_10km_pH (1).tif":  ("CEC",       "CEC [cmol+/kg]",          "YlOrRd",  15,  26),
    "soc_sicily_masked_10km_pH (1).tif":  ("SOC",       "SOC [dg/kg]",             "Greens",  10,  90),
    "Anions_interpolated_umolC_L.tif":    ("Anions",    "Anions [umolC/L]",        "Blues",   2000,4500),
    "r_het_Sic_10km_resampled2.tif":      ("R_het",     "R_het [gC/m2/yr]",        "Oranges", 0,   600),
    "K_s.tif":                            ("Ks",        "K_s [mm/d]",              "PuBu",    0,   2),
    "n.tif":                              ("n",         "Porosity n [-]",           "BuPu",    0.3, 0.55),
    "b.tif":                              ("b",         "Brooks-Corey b [-]",       "copper",  3,   9),
    "s_fc.tif":                           ("s_fc",      "s_fc [-]",                "Blues",   0.4, 1.0),
    "s_h.tif":                            ("s_h",       "s_h [-]",                 "Greens",  0,   0.4),
    "s_w.tif":                            ("s_w",       "s_w [-]",                 "Purples", 0.1, 0.7),
}

# ── Load reference georef ─────────────────────────────────────────────────────
with rasterio.open(REF_TIF) as src:
    ref_transform = src.transform
    ref_crs       = src.crs
    TIF_H, TIF_W  = src.shape
    bounds        = src.bounds

extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
print(f"Reference grid: {TIF_H}x{TIF_W}, {ref_crs}")

# ── Process each tif ──────────────────────────────────────────────────────────
results = {}   # fname -> {shortname, label, data, valid_mask, n_valid, ...}

tif_files = sorted([f for f in SOIL_DIR.iterdir()
                    if f.suffix == ".tif" and f.name not in SKIP])

for fpath in tif_files:
    with rasterio.open(str(fpath)) as src:
        data = src.read(1).astype(float)
        nd   = src.nodata
    if nd is not None:
        data[data == nd] = np.nan

    valid = np.isfinite(data) & (data != 0)
    n_valid = int(valid.sum())
    n_total = data.size

    info = LABELS.get(fpath.name, (fpath.stem, fpath.stem, "viridis",
                                   np.nanmin(data), np.nanmax(data)))
    shortname, label, cmap, vmin, vmax = info

    print(f"  {shortname:8s}: {n_valid:4d}/{n_total} valid  "
          f"({100*n_valid/n_total:.1f}%)  "
          f"range [{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]")

    # Save valid mask as GeoTIFF
    out_tif = OUT_DIR / f"valid_mask_{shortname}_soilparam.tif"
    with rasterio.open(
        str(out_tif), "w",
        driver="GTiff", height=TIF_H, width=TIF_W,
        count=1, dtype=rasterio.uint8,
        crs=ref_crs, transform=ref_transform,
        nodata=255, compress="lzw",
    ) as dst:
        dst.write(valid.astype(np.uint8), 1)

    results[fpath.name] = {
        "shortname": shortname, "label": label,
        "data": data, "valid": valid,
        "n_valid": n_valid, "n_total": n_total,
        "cmap": cmap, "vmin": vmin, "vmax": vmax,
    }

print(f"\nProcessed {len(results)} layers.")

# ── Plot: two rows — top = actual values, bottom = valid mask ─────────────────
n = len(results)
ncols = 6
nrows_data = (n + ncols - 1) // ncols   # rows for value maps
nrows_mask = (n + ncols - 1) // ncols   # rows for valid masks

# --- Figure 1: actual values projected over Sicily ---
fig1, axes1 = plt.subplots(
    nrows_data, ncols,
    figsize=(ncols * 3.5, nrows_data * 4),
    subplot_kw={"projection": ccrs.PlateCarree()}
)
fig1.suptitle("Soil parameter maps — Sicily 10 km grid", fontsize=14, fontweight="bold")
axes1_flat = axes1.flatten()

def add_geo(ax):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7)
    ax.add_feature(cfeature.LAND.with_scale("10m"),      facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"),     facecolor="#d0e8f5", zorder=0)
    gl = ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4)

for idx, (fname, info) in enumerate(results.items()):
    ax = axes1_flat[idx]
    add_geo(ax)
    data_plot = info["data"].copy()
    data_plot[~info["valid"]] = np.nan
    im = ax.imshow(data_plot,
                   origin="upper", extent=extent,
                   cmap=info["cmap"],
                   vmin=info["vmin"], vmax=info["vmax"],
                   interpolation="nearest",
                   transform=ccrs.PlateCarree(), zorder=2)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=info["label"])
    ax.set_title(f"{info['shortname']}\n{info['n_valid']}/{info['n_total']} "
                 f"({100*info['n_valid']/info['n_total']:.1f}%)", fontsize=9)

# Hide unused axes
for idx in range(len(results), len(axes1_flat)):
    axes1_flat[idx].set_visible(False)

plt.tight_layout()
out1 = OUT_DIR / "soil_params_values.png"
plt.savefig(str(out1), dpi=150, bbox_inches="tight")
print(f"Values figure saved: {out1}")
plt.close()

# --- Figure 2: valid pixel masks ---
fig2, axes2 = plt.subplots(
    nrows_mask, ncols,
    figsize=(ncols * 3.5, nrows_mask * 4),
    subplot_kw={"projection": ccrs.PlateCarree()}
)
fig2.suptitle("Valid pixels — soil parameter maps\n"
              "(finite, non-NaN, non-zero)", fontsize=14, fontweight="bold")
axes2_flat = axes2.flatten()

for idx, (fname, info) in enumerate(results.items()):
    ax = axes2_flat[idx]
    add_geo(ax)
    vc    = plt.get_cmap(info["cmap"])(0.65)
    cmap2 = mcolors.ListedColormap(["#cccccc", vc])
    ax.imshow(info["valid"].astype(float),
              origin="upper", extent=extent,
              cmap=cmap2, vmin=0, vmax=1,
              interpolation="nearest",
              transform=ccrs.PlateCarree(), zorder=2)
    ax.set_title(f"{info['shortname']}\n{info['n_valid']}/{info['n_total']} "
                 f"({100*info['n_valid']/info['n_total']:.1f}%)", fontsize=9)
    ax.legend(handles=[
        mpatches.Patch(facecolor="#cccccc", edgecolor="grey", label="Invalid", linewidth=0.5),
        mpatches.Patch(facecolor=vc,        edgecolor="grey", label="Valid",   linewidth=0.5),
    ], loc="lower right", fontsize=7, framealpha=0.8)

for idx in range(len(results), len(axes2_flat)):
    axes2_flat[idx].set_visible(False)

plt.tight_layout()
out2 = OUT_DIR / "soil_params_valid_masks.png"
plt.savefig(str(out2), dpi=150, bbox_inches="tight")
print(f"Valid masks figure saved: {out2}")
plt.close()

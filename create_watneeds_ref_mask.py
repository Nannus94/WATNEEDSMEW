"""
Create the WATNEEDS reference mask (426 pixels) and save as GeoTIFF.
This is THE common mask for all crops and scenarios.
Also shows overlap with all crop area maps.
"""
import numpy as np
import scipy.io
import os
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── PATHS ─────────────────────────────────────────────────────────────────────
HYDRO_DIR  = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\SMEW_Output_4Hour_MERIDA\olivo_surface"
AREA_DIR   = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\Aree_coltivate"
MAINLAND   = os.path.join(AREA_DIR, "sicily_mainland_mask.tif")
OUT_MASK   = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\Aree_coltivate\watneeds_ref_mask.tif"
OUT_PNG    = r"C:\Users\Latitude 5511\Downloads\watneeds_ref_mask_overview.png"

CROPS = {
    "Olivo":  "sicily10km_olives_total_ha.tif",
    "Vite":   "sicily10km_vineyard_total_ha.tif",
    "Agrumi": "sicily10km_citrus_total_ha.tif",
    "Pesco":  "sicily10km_fruits_total_ha.tif",
    "Grano":  "sicily10km_wheat_total_ha.tif",
}

# ── BUILD MASK FROM WATNEEDS HYDRO DATA ──────────────────────────────────────
mat = scipy.io.loadmat(os.path.join(HYDRO_DIR, "shallow_s_2000_6.mat"))
key = [k for k in mat if not k.startswith("_")][0]
s = mat[key].astype(float)
watneeds_mask = ~np.all(np.isnan(s), axis=2)  # True where WATNEEDS has data

# ── SAVE AS GEOTIFF ──────────────────────────────────────────────────────────
# Copy georeferencing from an existing crop tif
ref_tif = os.path.join(AREA_DIR, list(CROPS.values())[0])
# Try cluster version in Downloads first
ref_tif_dl = os.path.join(r"C:\Users\Latitude 5511\Downloads", list(CROPS.values())[0])
if os.path.exists(ref_tif_dl):
    ref_tif = ref_tif_dl

with rasterio.open(ref_tif) as src:
    profile = src.profile.copy()

profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
with rasterio.open(OUT_MASK, "w", **profile) as dst:
    dst.write(watneeds_mask.astype(np.uint8), 1)
print(f"Saved reference mask: {OUT_MASK}  ({watneeds_mask.sum()} pixels)")

# ── LOAD CROP MAPS ───────────────────────────────────────────────────────────
crop_masks = {}
for crop, fname in CROPS.items():
    # Try Downloads (cluster version) first, then local Aree_coltivate
    fpath = os.path.join(r"C:\Users\Latitude 5511\Downloads", fname)
    if not os.path.exists(fpath):
        fpath = os.path.join(AREA_DIR, fname)
    if os.path.exists(fpath):
        with rasterio.open(fpath) as src:
            d = src.read(1).astype(float)
        crop_masks[crop] = d > 0
    else:
        print(f"  WARNING: {fname} not found")

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("WATNEEDS Reference Mask (426 px) vs Crop Areas\nRed = cultivated but outside WATNEEDS domain",
             fontsize=13, fontweight="bold")

# Panel 0: the mask itself
ax = axes[0, 0]
display = np.full(watneeds_mask.shape, np.nan)
display[watneeds_mask] = 1
display[~watneeds_mask & np.ones_like(watneeds_mask, dtype=bool)] = 0
ax.imshow(display, cmap="Greens", vmin=0, vmax=1, origin="upper", aspect="auto")
ax.set_title(f"WATNEEDS ref mask ({watneeds_mask.sum()} px)\nGreen = valid domain")

# Panels 1-5: each crop
for i, (crop, cmask) in enumerate(crop_masks.items()):
    ax = axes[(i + 1) // 3, (i + 1) % 3]

    inside  = cmask & watneeds_mask
    outside = cmask & ~watneeds_mask

    # Background: WATNEEDS domain in light grey
    bg = np.full(watneeds_mask.shape, np.nan)
    bg[watneeds_mask] = 0.85
    ax.imshow(bg, cmap="Greys", vmin=0, vmax=1, origin="upper", aspect="auto")

    # Inside: green dots
    ri, ci = np.where(inside)
    ax.scatter(ci, ri, s=30, c="green", marker="o", edgecolors="none",
               label=f"Inside ({inside.sum()})")

    # Outside: red dots
    ro, co = np.where(outside)
    if len(ro) > 0:
        ax.scatter(co, ro, s=50, c="red", marker="x", linewidths=1.5,
                   label=f"Outside WATNEEDS ({outside.sum()})")

    ax.set_title(f"{crop}  ({int(cmask.sum())} cultivated px)")
    ax.legend(fontsize=8, loc="lower right")

for ax in axes.flat:
    ax.set_xlabel("Col")
    ax.set_ylabel("Row")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
plt.close()

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print(f"\n{'Crop':<10} {'Cultivated':>10} {'Inside':>8} {'Outside':>8} {'Coverage':>9}")
print("-" * 50)
for crop, cmask in crop_masks.items():
    inside  = int(np.sum(cmask & watneeds_mask))
    outside = int(np.sum(cmask & ~watneeds_mask))
    total   = int(cmask.sum())
    pct     = 100 * inside / total if total > 0 else 0
    print(f"{crop:<10} {total:>10} {inside:>8} {outside:>8} {pct:>8.1f}%")

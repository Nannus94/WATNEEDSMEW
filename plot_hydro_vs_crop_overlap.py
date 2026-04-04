"""
Show overlap between WATNEEDS hydro valid pixels and olive cultivated area.
Three panels: (1) mean s from hydro, (2) olive ha, (3) overlap diagnostic.
"""
import numpy as np
import rasterio
import scipy.io
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
HYDRO_DIR   = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\SMEW_Output_4Hour_MERIDA\olivo_surface"
OLIVO_FILE  = r"C:\Users\Latitude 5511\Downloads\sicily10km_olives_total_ha.tif"
MAINLAND    = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\Aree_coltivate\sicily_mainland_mask.tif"
OUTPATH     = r"C:\Users\Latitude 5511\Downloads\hydro_vs_crop_overlap.png"
YEAR        = 1994  # first year of calibration window


# ── LOAD HYDRO ────────────────────────────────────────────────────────────────
all_s = []
for month in range(1, 13):
    fpath = os.path.join(HYDRO_DIR, f"shallow_s_{YEAR}_{month}.mat")
    if not os.path.exists(fpath):
        continue
    mat = scipy.io.loadmat(fpath)
    key = [k for k in mat if not k.startswith("_")][0]
    all_s.append(mat[key].astype(float))

s_full = np.concatenate(all_s, axis=2)
s_mean = np.nanmean(s_full, axis=2)
s_min  = np.nanmin(s_full, axis=2)
s_first = s_full[:, :, 0]

# Reproduce the exact hydro_valid_mask from calibration
hydro_valid = (s_mean > 0.01) & (s_first > 0.001) & (s_min > 0)
# Also flag all-NaN pixels
all_nan = np.all(np.isnan(s_full), axis=2)


# ── LOAD CROP + MAINLAND ─────────────────────────────────────────────────────
with rasterio.open(OLIVO_FILE) as src:
    olivo = src.read(1).astype(float)
with rasterio.open(MAINLAND) as src:
    mainland = src.read(1).astype(bool)

olivo_mask = olivo > 0


# ── OVERLAP CATEGORIES ───────────────────────────────────────────────────────
# 0 = sea/outside mainland
# 1 = mainland, no olive, no hydro
# 2 = mainland, hydro valid only (no olive)
# 3 = mainland, olive only (no hydro) ← PROBLEM pixels
# 4 = mainland, olive + hydro valid ← GOOD
# 5 = mainland, olive + hydro but fails validity check

cat = np.zeros(s_mean.shape, dtype=int)
cat[mainland]                                   = 1  # mainland base
cat[mainland & hydro_valid & ~olivo_mask]        = 2  # hydro only
cat[mainland & olivo_mask & all_nan]             = 3  # olive, no hydro at all
cat[mainland & olivo_mask & ~all_nan & ~hydro_valid] = 5  # olive, hydro exists but fails check
cat[mainland & olivo_mask & hydro_valid]         = 4  # both valid

n_good     = int(np.sum(cat == 4))
n_no_hydro = int(np.sum(cat == 3))
n_bad_hydro = int(np.sum(cat == 5))
n_hydro_only = int(np.sum(cat == 2))


# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle(f"Olivo Surface — Hydro vs Crop Overlap (year {YEAR})",
             fontsize=13, fontweight="bold")

# Panel 1: mean s (hydro coverage)
s_display = s_mean.copy()
s_display[~mainland] = np.nan
im0 = axes[0].imshow(s_display, cmap="Blues", vmin=0, vmax=1, origin="upper", aspect="auto")
fig.colorbar(im0, ax=axes[0], shrink=0.75, label="mean s [-]")
# Mark olive pixels
or_, oc_ = np.where(olivo_mask)
axes[0].scatter(oc_, or_, s=8, c="red", marker="o", alpha=0.7, linewidths=0, label=f"Olive ({int(olivo_mask.sum())} px)")
axes[0].legend(fontsize=8, loc="lower right")
axes[0].set_title("WATNEEDS mean soil moisture\n+ olive pixels (red)")

# Panel 2: olive area [ha]
olivo_display = olivo.copy()
olivo_display[olivo_display <= 0] = np.nan
olivo_display[~mainland] = np.nan
im1 = axes[1].imshow(olivo_display, cmap="YlGn", origin="upper", aspect="auto")
fig.colorbar(im1, ax=axes[1], shrink=0.75, label="Olive area [ha]")
# Mark hydro-valid pixels
hr, hc = np.where(hydro_valid)
axes[1].scatter(hc, hr, s=6, c="steelblue", marker="s", alpha=0.3, linewidths=0, label=f"Hydro valid ({int(hydro_valid.sum())} px)")
axes[1].legend(fontsize=8, loc="lower right")
axes[1].set_title("Olive cultivated area [ha]\n+ hydro valid pixels (blue)")

# Panel 3: overlap diagnostic
cmap_cat = ListedColormap(["white", "#e0e0e0", "#6baed6", "#d62728", "#2ca02c", "#ff7f0e"])
im2 = axes[2].imshow(cat, cmap=cmap_cat, vmin=0, vmax=5, origin="upper", aspect="auto")
# Legend via patches
import matplotlib.patches as mpatches
patches = [
    mpatches.Patch(color="white", label="Sea"),
    mpatches.Patch(color="#e0e0e0", label=f"Mainland only"),
    mpatches.Patch(color="#6baed6", label=f"Hydro only, no olive ({n_hydro_only})"),
    mpatches.Patch(color="#d62728", label=f"Olive, NO hydro ({n_no_hydro})"),
    mpatches.Patch(color="#2ca02c", label=f"Olive + hydro OK ({n_good})"),
    mpatches.Patch(color="#ff7f0e", label=f"Olive + hydro bad ({n_bad_hydro})"),
]
axes[2].legend(handles=patches, fontsize=7, loc="lower right")
axes[2].set_title("Overlap diagnostic")

for ax in axes:
    ax.set_xlabel("Col (W→E)")
    ax.set_ylabel("Row (N→S)")

plt.tight_layout()
plt.savefig(OUTPATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPATH}")
plt.close()

# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────
print(f"\nOlive cultivated pixels:     {int(olivo_mask.sum())}")
print(f"Hydro valid pixels:          {int(hydro_valid.sum())}")
print(f"Olive + hydro valid (GOOD):  {n_good}")
print(f"Olive, NO hydro (all NaN):   {n_no_hydro}")
print(f"Olive, hydro bad (s=0/low):  {n_bad_hydro}")
print(f"Hydro only (no olive):       {n_hydro_only}")

if n_no_hydro > 0:
    print(f"\nPixels with olive but NO hydro data:")
    for r, c in zip(*np.where(cat == 3)):
        print(f"  [{r:2d},{c:2d}]  olive={olivo[r,c]:.1f} ha  s_mean={'NaN':>8s}")

if n_bad_hydro > 0:
    print(f"\nPixels with olive but BAD hydro (exists but fails validity):")
    for r, c in zip(*np.where(cat == 5)):
        print(f"  [{r:2d},{c:2d}]  olive={olivo[r,c]:.1f} ha  s_mean={s_mean[r,c]:.4f}  s_min={s_min[r,c]:.6f}  s_first={s_first[r,c]:.6f}")

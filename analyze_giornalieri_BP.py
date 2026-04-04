"""
Analyze BP variable, crop 2, from giornalieri_SURF_5crops dataset.
RAM-safe: accumulates valid-pixel mask incrementally (no full concat in memory).

Valid pixel = no NaN in ANY timestep AND at least once positive.
"""

import re
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no GUI window needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Latitude 5511\Downloads\giornalieri_SURF_5crops\giornalieri_SURF_5crops")
OUT_DIR  = Path(r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW")

VAR   = "BP"
CROP  = 2
KEY   = f"output{VAR}"   # HDF5 dataset key inside each file

# ── 1. Discover and sort files ────────────────────────────────────────────────
pattern = re.compile(rf"^output{VAR}_(\d{{4}})_(\d{{1,2}})_{CROP}\.mat$")

files = []
for f in DATA_DIR.iterdir():
    m = pattern.match(f.name)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        files.append((year, month, f))

files.sort(key=lambda x: (x[0], x[1]))
print(f"Found {len(files)} files for {VAR} crop {CROP}")

# ── 2. Peek at spatial shape ──────────────────────────────────────────────────
with h5py.File(str(files[0][2]), "r") as f:
    sample = f[KEY][:]          # (days, lat, lon)
    spatial_shape = sample.shape[1:]   # (lat, lon)
    print(f"Spatial shape: {spatial_shape}  |  sample time steps: {sample.shape[0]}")

# ── 3. Incremental mask accumulation (RAM-safe) ───────────────────────────────
# has_nan : True if ANY timestep in ANY file was NaN at that pixel
# any_pos : True if ANY timestep in ANY file was > 0 at that pixel
has_nan = np.zeros(spatial_shape, dtype=bool)
any_pos = np.zeros(spatial_shape, dtype=bool)

total_steps = 0
for i, (year, month, fpath) in enumerate(files):
    with h5py.File(str(fpath), "r") as f:
        data = f[KEY][:]          # (days, lat, lon)  float64
    # data axes: (time, lat, lon) — confirmed from inspection
    has_nan |= np.any(np.isnan(data), axis=0)
    any_pos |= np.any(data > 0,       axis=0)
    total_steps += data.shape[0]
    if (i + 1) % 60 == 0:
        print(f"  processed {i+1}/{len(files)} files  ({year}-{month:02d})...", flush=True)

print(f"\nTotal time steps processed: {total_steps}")

valid_mask = (~has_nan) & any_pos
n_valid = int(valid_mask.sum())
n_total = valid_mask.size
print(f"Valid pixels: {n_valid} / {n_total}  ({100*n_valid/n_total:.1f} %)")

# ── 4. Save mask ──────────────────────────────────────────────────────────────
mask_path = OUT_DIR / f"valid_mask_{VAR}_crop{CROP}_SURF.npy"
np.save(str(mask_path), valid_mask)
print(f"Mask saved: {mask_path}")

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
fig.suptitle(
    f"Valid pixels  |  {VAR}  crop {CROP}  |  giornalieri SURF\n"
    "Criterion: no NaN at any timestep  AND  at least one positive value",
    fontsize=11, fontweight="bold"
)

base_cmap = plt.get_cmap("Blues")
valid_color = base_cmap(0.65)
cmap2 = mcolors.ListedColormap(["#cccccc", valid_color])

ax.imshow(valid_mask.astype(float), origin="upper",
          cmap=cmap2, vmin=0, vmax=1, interpolation="nearest")
ax.set_title(
    f"{n_valid} / {n_total} valid pixels  ({100*n_valid/n_total:.1f} %)",
    fontsize=10
)
ax.set_xlabel("lon index", fontsize=9)
ax.set_ylabel("lat index", fontsize=9)

legend_elements = [
    mpatches.Patch(facecolor="#cccccc", edgecolor="grey", label="Invalid (NaN or all-zero)"),
    mpatches.Patch(facecolor=valid_color, edgecolor="grey", label="Valid"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.85)

plt.tight_layout()
out_path = OUT_DIR / f"valid_pixels_{VAR}_crop{CROP}_SURF.png"
plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
print(f"Figure saved: {out_path}")

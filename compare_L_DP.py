import numpy as np
import scipy.io as sio
import h5py
from pathlib import Path

OUT_DIR  = Path(r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW")
DATA_L   = Path(r"C:\Users\Latitude 5511\Downloads\olivo_surface")
DATA_DP  = Path(r"C:\Users\Latitude 5511\Downloads\giornalieri_SURF_5crops\giornalieri_SURF_5crops")

# Load pre-computed masks
L_mask  = np.load(str(OUT_DIR / "valid_mask_L_olivo_surface.npy"))
dp_mask = np.load(str(OUT_DIR / "valid_mask_DP_crop2_SURF.npy"))
bp_mask = np.load(str(OUT_DIR / "valid_mask_BP_crop2_SURF_v2.npy"))

print(f"shallow_L shape : {L_mask.shape}  valid: {L_mask.sum()}/{L_mask.size}")
print(f"DP        shape : {dp_mask.shape}  valid: {dp_mask.sum()}/{dp_mask.size}")
print(f"BP        shape : {bp_mask.shape}  valid: {bp_mask.sum()}/{bp_mask.size}")
print()
print(f"Overlap  L & DP : {(L_mask & dp_mask).sum()}")
print(f"L valid, DP not : {(L_mask & ~dp_mask).sum()}")
print(f"DP valid, L not : {(dp_mask & ~L_mask).sum()}")
print(f"L == DP exactly : {np.array_equal(L_mask, dp_mask)}")

# For pixels where L is valid but DP is not — check if DP is all-zero there
extra = L_mask & ~dp_mask
print(f"\nPixels where shallow_L valid but DP not: {extra.sum()}")

if extra.sum() > 0:
    # Sample one DP file and check those pixels
    dp_zero_count = np.zeros((39, 43), dtype=np.int32)
    dp_nan_count  = np.zeros((39, 43), dtype=np.int32)
    total_steps   = 0
    import re
    pat = re.compile(r"^outputDP_(\d{4})_(\d{1,2})_2\.mat$")
    files = sorted([(int(m.group(1)), int(m.group(2)), f)
                    for f in DATA_DP.iterdir()
                    for m in [pat.match(f.name)] if m],
                   key=lambda x: (x[0], x[1]))
    SR, SC = 1, 1
    for _, _, fpath in files:
        with h5py.File(str(fpath), "r") as f:
            raw = f["outputDP"][:]
        c = raw.T[SR:SR+39, SC:SC+43, :]
        dp_zero_count += np.sum(c == 0,       axis=2)
        dp_nan_count  += np.sum(np.isnan(c),  axis=2)
        total_steps   += c.shape[2]

    all_zero = (dp_zero_count[extra] == total_steps).sum()
    all_nan  = (dp_nan_count[extra]  == total_steps).sum()
    mixed    = extra.sum() - all_zero - all_nan
    print(f"  -> DP all-zero : {all_zero}")
    print(f"  -> DP all-NaN  : {all_nan}")
    print(f"  -> DP mixed    : {mixed}")

# For pixels where DP is valid but L is not — check shallow_L
extra2 = dp_mask & ~L_mask
print(f"\nPixels where DP valid but shallow_L not: {extra2.sum()}")

if extra2.sum() > 0:
    L_zero_count = np.zeros((39, 43), dtype=np.int32)
    L_nan_count  = np.zeros((39, 43), dtype=np.int32)
    total_steps_L = 0
    pat_L = re.compile(r"^shallow_L_(\d{4})_(\d{1,2})\.mat$")
    files_L = sorted([(int(m.group(1)), int(m.group(2)), f)
                      for f in DATA_L.iterdir()
                      for m in [pat_L.match(f.name)] if m],
                     key=lambda x: (x[0], x[1]))
    for _, _, fpath in files_L:
        mat = sio.loadmat(str(fpath))
        c = mat["L_shallow"]   # (39, 43, steps) — already in correct shape
        L_zero_count += np.sum(c == 0,       axis=2)
        L_nan_count  += np.sum(np.isnan(c),  axis=2)
        total_steps_L += c.shape[2]

    all_zero = (L_zero_count[extra2] == total_steps_L).sum()
    all_nan  = (L_nan_count[extra2]  == total_steps_L).sum()
    mixed    = extra2.sum() - all_zero - all_nan
    print(f"  -> shallow_L all-zero : {all_zero}")
    print(f"  -> shallow_L all-NaN  : {all_nan}")
    print(f"  -> shallow_L mixed    : {mixed}")

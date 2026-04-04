"""
Diagnose anomalous L and I for surface scenario in non-coastal pixels.
Compares: 1) raw input (PR, BW) dimensions and values, 2) output (L, I) surface vs drip.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io
import h5py

base_dir = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW"
main_output_dir = os.path.join(base_dir, "SMEW_Output_4Hour_Raw_Steps")
ref_map_path = os.path.join(base_dir, r"Aree_coltivate\sicilia_cellarea_10km.tif")
OUTPUT_DIR = os.path.join(base_dir, "_DIAGNOSIS_SURFACE_LI")

# Paths for ORIGINAL input data (before New_WB)
SURF_ORIG = os.path.join(base_dir, r"giornalieri_SURF_5crops\giornalieri_SURF_5crops")
DRIP_ORIG = os.path.join(base_dir, "giornalieri_DRIP_5crops")

# Test crop (exists in both surface and drip)
CROP_CODE = 1  # vite
tif_h, tif_w = 39, 43
N_SUBSTEPS = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_mat(filepath, var_name):
    try:
        return scipy.io.loadmat(filepath)[var_name]
    except Exception:
        with h5py.File(filepath, 'r') as f:
            return np.array(f.get(var_name)).T


def crop_center(arr, target_h, target_w):
    if arr is None or arr.ndim < 2:
        return None
    h, w = arr.shape[:2]
    sr, sc = (h - target_h) // 2, (w - target_w) // 2
    if arr.ndim == 3:
        return arr[sr:sr+target_h, sc:sc+target_w, :]
    return arr[sr:sr+target_h, sc:sc+target_w]


def main():
    import rasterio
    with rasterio.open(ref_map_path) as src:
        ref_data = src.read(1).astype(float)
    tif_h, tif_w = ref_data.shape
    land_mask = np.isfinite(ref_data) & (ref_data > 0)
    # coastal approx: edge pixels
    coastal_mask = np.zeros_like(land_mask, dtype=bool)
    coastal_mask[0, :] = True
    coastal_mask[-1, :] = True
    coastal_mask[:, 0] = True
    coastal_mask[:, -1] = True
    inland_mask = land_mask & ~coastal_mask

    print("=" * 60)
    print("DIAGNOSIS: Surface L & I anomaly in non-coastal pixels")
    print("=" * 60)

    # --- 1. Check raw input file dimensions (PR, BW) ---
    year_sample, month_sample = 2010, 6
    f_pr_surf = os.path.join(SURF_ORIG, f"outputPR_{year_sample}_{month_sample}_{CROP_CODE}.mat")
    f_bw_surf = os.path.join(SURF_ORIG, f"outputBW_{year_sample}_{month_sample}_{CROP_CODE}.mat")
    f_pr_drip = os.path.join(DRIP_ORIG, f"outputPR_{year_sample}_{month_sample}_{CROP_CODE}.mat")
    f_bw_drip = os.path.join(DRIP_ORIG, f"outputBW_{year_sample}_{month_sample}_{CROP_CODE}.mat")

    print("\n1. RAW INPUT FILE DIMENSIONS (PR, BW):")
    for name, path in [("Surface PR", f_pr_surf), ("Surface BW", f_bw_surf), ("Drip PR", f_pr_drip), ("Drip BW", f_bw_drip)]:
        if os.path.exists(path):
            arr = load_mat(path, "outputPR" if "PR" in name else "outputBW")
            print(f"   {name}: shape {arr.shape}")
        else:
            print(f"   {name}: FILE NOT FOUND")

    # --- 2. Compare PR and BW values (cropped) for a sample month ---
    print("\n2. SAMPLE VALUES (2010-06, cropped to 39x43):")
    if os.path.exists(f_pr_surf) and os.path.exists(f_pr_drip):
        pr_surf = crop_center(load_mat(f_pr_surf, 'outputPR'), tif_h, tif_w)
        pr_drip = crop_center(load_mat(f_pr_drip, 'outputPR'), tif_h, tif_w)
        bw_surf = crop_center(load_mat(f_bw_surf, 'outputBW'), tif_h, tif_w) if os.path.exists(f_bw_surf) else np.zeros_like(pr_surf)
        bw_drip = crop_center(load_mat(f_bw_drip, 'outputBW'), tif_h, tif_w) if os.path.exists(f_bw_drip) else np.zeros_like(pr_drip)

        for mask_name, mask in [("Inland", inland_mask), ("Coastal", coastal_mask)]:
            if np.sum(mask) == 0:
                continue
            pr_s_m = np.nanmean(pr_surf[mask, :])
            pr_d_m = np.nanmean(pr_drip[mask, :])
            bw_s_m = np.nanmean(bw_surf[mask, :])
            bw_d_m = np.nanmean(bw_drip[mask, :])
            print(f"   {mask_name}: PR surf={pr_s_m:.4f} drip={pr_d_m:.4f} | BW surf={bw_s_m:.4f} drip={bw_d_m:.4f}")

    # --- 3. Load shallow output and compare L, I surface vs drip ---
    surf_folder = os.path.join(main_output_dir, "vite_surface")
    drip_folder = os.path.join(main_output_dir, "vite_drip")

    if not os.path.exists(surf_folder) or not os.path.exists(drip_folder):
        print("\n3. Shallow output folders not found. Run New_WB first.")
        return

    print("\n3. SHALLOW OUTPUT (L, I) - temporal mean per pixel:")
    list_L_surf, list_I_surf = [], []
    list_L_drip, list_I_drip = [], []
    for year in range(2010, 2011):  # 1 year
        for month in range(1, 13):
            for folder in [surf_folder, drip_folder]:
                list_L = list_L_surf if folder == surf_folder else list_L_drip
                list_I = list_I_surf if folder == surf_folder else list_I_drip
                f_s = os.path.join(folder, f"shallow_s_{year}_{month}.mat")
                if not os.path.exists(f_s):
                    continue
                L = load_mat(os.path.join(folder, f"shallow_L_{year}_{month}.mat"), 'L_shallow')
                f_i = os.path.join(folder, f"shallow_I_{year}_{month}.mat")
                I = load_mat(f_i, 'I_shallow') if os.path.exists(f_i) else np.zeros_like(L)
                n_days = L.shape[2] // N_SUBSTEPS
                n_steps = n_days * N_SUBSTEPS
                L_daily = np.nansum(L[:, :, :n_steps].reshape(tif_h, tif_w, n_days, N_SUBSTEPS), axis=3)
                I_daily = np.nansum(I[:, :, :n_steps].reshape(tif_h, tif_w, n_days, N_SUBSTEPS), axis=3)
                list_L.append(L_daily)
                list_I.append(I_daily)

    if list_L_surf and list_L_drip:
        L_surf = np.nanmean(np.concatenate(list_L_surf, axis=2), axis=2)
        I_surf = np.nanmean(np.concatenate(list_I_surf, axis=2), axis=2)
        L_drip = np.nanmean(np.concatenate(list_L_drip, axis=2), axis=2)
        I_drip = np.nanmean(np.concatenate(list_I_drip, axis=2), axis=2)

        L_surf_inland = np.nanmean(L_surf[inland_mask])
        I_surf_inland = np.nanmean(I_surf[inland_mask])
        L_drip_inland = np.nanmean(L_drip[inland_mask])
        I_drip_inland = np.nanmean(I_drip[inland_mask])
        print(f"   Inland mean:  L surf={L_surf_inland:.4f} drip={L_drip_inland:.4f} | I surf={I_surf_inland:.4f} drip={I_drip_inland:.4f}")
        print(f"   Ratio surf/drip: L={L_surf_inland/max(L_drip_inland,1e-6):.2f}x  I={I_surf_inland/max(I_drip_inland,1e-6):.2f}x")

    # --- 4. Spatial maps: surface L, I vs drip; anomaly (surface - drip) ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes[0, 0].imshow(np.where(land_mask, L_surf, np.nan), cmap='Oranges', origin='upper')
    axes[0, 0].set_title("L Surface (mm/day)")
    axes[0, 1].imshow(np.where(land_mask, L_drip, np.nan), cmap='Oranges', origin='upper')
    axes[0, 1].set_title("L Drip (mm/day)")
    diff_L = L_surf - L_drip
    v = np.nanpercentile(np.abs(diff_L[land_mask]), 99)
    axes[0, 2].imshow(np.where(land_mask, diff_L, np.nan), cmap='RdBu_r', vmin=-v, vmax=v, origin='upper')
    axes[0, 2].set_title("L Surface - Drip")

    axes[1, 0].imshow(np.where(land_mask, I_surf, np.nan), cmap='Blues', origin='upper')
    axes[1, 0].set_title("I Surface (mm/day)")
    axes[1, 1].imshow(np.where(land_mask, I_drip, np.nan), cmap='Blues', origin='upper')
    axes[1, 1].set_title("I Drip (mm/day)")
    diff_I = I_surf - I_drip
    vI = np.nanpercentile(np.abs(diff_I[land_mask]), 99)
    axes[1, 2].imshow(np.where(land_mask, diff_I, np.nan), cmap='RdBu_r', vmin=-vI, vmax=vI, origin='upper')
    axes[1, 2].set_title("I Surface - Drip")

    for ax in axes.flat:
        ax.set_axis_off()
    plt.suptitle("Surface vs Drip — L & I anomaly diagnosis (vite, 2010)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "surface_vs_drip_LI_maps.png"), dpi=150)
    plt.close()
    print(f"\n   Saved: {OUTPUT_DIR}/surface_vs_drip_LI_maps.png")

    # --- 5. Scatter: BW input vs I output (surface) - is high I driven by high BW? ---
    if os.path.exists(f_pr_surf) and os.path.exists(f_bw_surf):
        bw_surf_flat = bw_surf.reshape(-1, bw_surf.shape[2])
        bw_mean = np.nanmean(bw_surf_flat, axis=1)
    else:
        bw_mean = np.zeros(tif_h * tif_w)
    I_surf_flat = I_surf.flatten()
    inland_flat = inland_mask.flatten()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(bw_mean[inland_flat], I_surf_flat[inland_flat], alpha=0.5, s=10)
    ax.set_xlabel("Mean BW input (mm/day)")
    ax.set_ylabel("Mean I output shallow (mm/day)")
    ax.set_title("Surface: BW input vs I output (inland pixels)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "surface_BW_vs_I_scatter.png"), dpi=150)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/surface_BW_vs_I_scatter.png")

    print("\n" + "=" * 60)
    print("Check if: 1) Surface PR/BW dimensions differ from drip")
    print("          2) Surface BW >> Drip BW in inland pixels")
    print("          3) Surface uses different variable names/units")
    print("=" * 60)


if __name__ == "__main__":
    main()

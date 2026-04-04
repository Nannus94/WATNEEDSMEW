"""
Compare olivo calibration results across 3 irrigation scenarios:
drip, traditional, rainfed.
"""
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr

# ── PATHS ─────────────────────────────────────────────────────────────────────
DL = r"C:\Users\Latitude 5511\Downloads"

SCENARIOS = {
    'Drip': {
        'alpha': f"{DL}\\FULL_OLIVO_DRIP_30Y_alpha_map_MERGED (5).tif",
        'error': f"{DL}\\FULL_OLIVO_DRIP_30Y_error_map_MERGED (1).tif",
        'color': '#2ca02c',
    },
    'Traditional': {
        'alpha': f"{DL}\\FULL_OLIVO_TRADITIONAL_30Y_alpha_map_MERGED (3).tif",
        'error': f"{DL}\\FULL_OLIVO_TRADITIONAL_30Y_error_map_MERGED (2).tif",
        'color': '#d62728',
    },
    'Rainfed': {
        'alpha': f"{DL}\\FULL_OLIVO_RAINFED_30Y_alpha_map_MERGED.tif",
        'error': f"{DL}\\FULL_OLIVO_RAINFED_30Y_error_map_MERGED.tif",
        'color': '#1f77b4',
    },
}

PH_FILE    = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\soil_param (1)\sicily_ph_cacl2_10km.tif"
OLIVO_FILE = f"{DL}\\sicily10km_olives_total_ha.tif"
WN_FILE    = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\Aree_coltivate\watneeds_ref_mask.tif"
OUTDIR     = DL


# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_tif(path):
    with rasterio.open(path) as src:
        d = src.read(1).astype(float)
        if src.nodata is not None:
            d[d == src.nodata] = np.nan
    return d

def nn_fill(arr):
    arr = arr.copy()
    m = np.isnan(arr)
    if m.any() and (~m).any():
        _, idx = distance_transform_edt(m, return_distances=True, return_indices=True)
        arr[m] = arr[idx[0][m], idx[1][m]]
    return arr


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
ph_raw = load_tif(PH_FILE)
ph_raw[(ph_raw <= 0) | (ph_raw >= 14)] = np.nan
ph_filled = nn_fill(ph_raw)

olivo = load_tif(OLIVO_FILE)
wn = load_tif(WN_FILE).astype(bool)
crop_mask = (olivo > 0) & wn

# Load alpha maps
alphas = {}
errors = {}
for name, info in SCENARIOS.items():
    a = load_tif(info['alpha'])
    a[a <= 0] = np.nan
    alphas[name] = a
    errors[name] = load_tif(info['error'])

# Valid masks per scenario
valid = {name: crop_mask & ~np.isnan(a) for name, a in alphas.items()}

# ── PIXEL OVERLAP ─────────────────────────────────────────────────────────────
all_valid = valid['Drip'] & valid['Traditional'] & valid['Rainfed']
any_valid = valid['Drip'] | valid['Traditional'] | valid['Rainfed']

print("=" * 60)
print("OLIVO CALIBRATION — 3 Scenario Comparison")
print("=" * 60)
print(f"Olivo in WATNEEDS mask:  {int(crop_mask.sum())}")
print()
for name in SCENARIOS:
    n = int(valid[name].sum())
    print(f"  {name:15s}  calibrated: {n}  failed: {int(crop_mask.sum()) - n}")
print()
print(f"All 3 valid (overlap):   {int(all_valid.sum())}")
print(f"Any valid:               {int(any_valid.sum())}")
print()

# Alpha stats per scenario (on overlap pixels only)
print(f"{'Scenario':>15s}  {'mean':>7s}  {'median':>7s}  {'std':>7s}  {'min':>7s}  {'max':>7s}  {'RMSE_pH':>8s}")
print("-" * 70)
for name in SCENARIOS:
    a = alphas[name][all_valid]
    e = errors[name][all_valid]
    rmse = np.sqrt(np.nanmean(e**2))
    print(f"{name:>15s}  {np.mean(a):7.3f}  {np.median(a):7.3f}  {np.std(a):7.3f}  {np.min(a):7.3f}  {np.max(a):7.3f}  {rmse:8.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Pixel overlap map
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(10, 8))
fig1.suptitle("Olivo — Calibration Coverage Across 3 Irrigation Scenarios", fontsize=13, fontweight='bold')

# Category map
cat = np.zeros(crop_mask.shape, dtype=int)
cat[crop_mask & ~any_valid] = 1  # crop but none calibrated
cat[valid['Drip'] & ~valid['Traditional'] & ~valid['Rainfed']] = 2
cat[valid['Traditional'] & ~valid['Drip'] & ~valid['Rainfed']] = 3
cat[valid['Rainfed'] & ~valid['Drip'] & ~valid['Traditional']] = 4
# 2 of 3
two_of_three = (valid['Drip'].astype(int) + valid['Traditional'].astype(int) + valid['Rainfed'].astype(int)) == 2
cat[two_of_three & crop_mask] = 5
cat[all_valid] = 6

cmap = ListedColormap(['white', '#999999', '#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd'])
ax1.imshow(cat, cmap=cmap, vmin=0, vmax=6, origin='upper', aspect='auto')
patches = [
    mpatches.Patch(color='white', label='Sea/outside'),
    mpatches.Patch(color='#999999', label=f'Crop, none calibrated ({int((crop_mask & ~any_valid).sum())})'),
    mpatches.Patch(color='#2ca02c', label=f'Drip only ({int((cat==2).sum())})'),
    mpatches.Patch(color='#d62728', label=f'Traditional only ({int((cat==3).sum())})'),
    mpatches.Patch(color='#1f77b4', label=f'Rainfed only ({int((cat==4).sum())})'),
    mpatches.Patch(color='#ff7f0e', label=f'2 of 3 ({int((cat==5).sum())})'),
    mpatches.Patch(color='#9467bd', label=f'All 3 ({int(all_valid.sum())})'),
]
ax1.legend(handles=patches, fontsize=8, loc='lower right')
ax1.set_xlabel('Col (W→E)')
ax1.set_ylabel('Row (N→S)')
plt.tight_layout()
fig1.savefig(f"{OUTDIR}\\olivo_3scenarios_overlap.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTDIR}\\olivo_3scenarios_overlap.png")
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Alpha vs pH for all 3 scenarios (scatter)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle("Olivo — Alpha vs pH (NN-interpolated) per Irrigation Scenario", fontsize=13, fontweight='bold')

for ax, name in zip(axes2, SCENARIOS):
    mask = valid[name] & crop_mask
    a = alphas[name][mask]
    p = ph_filled[mask]
    ax.scatter(p, a, s=15, c=SCENARIOS[name]['color'], alpha=0.6, edgecolors='none')
    if len(a) > 5:
        r, _ = pearsonr(p, a)
        ax.set_title(f"{name}  (n={len(a)}, r={r:.2f})")
    else:
        ax.set_title(f"{name}  (n={len(a)})")
    ax.set_xlabel("Soil pH (CaCl₂)")
    ax.set_ylabel("Alpha")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(5, np.nanpercentile(a, 99) + 0.5))

plt.tight_layout()
fig2.savefig(f"{OUTDIR}\\olivo_3scenarios_alpha_vs_ph.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTDIR}\\olivo_3scenarios_alpha_vs_ph.png")
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Alpha difference maps (Traditional-Drip, Rainfed-Drip, Trad-Rainfed)
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle("Olivo — Alpha Differences Between Irrigation Scenarios\n(on overlapping pixels only)",
              fontsize=13, fontweight='bold')

pairs = [
    ('Traditional − Drip', 'Traditional', 'Drip'),
    ('Rainfed − Drip', 'Rainfed', 'Drip'),
    ('Traditional − Rainfed', 'Traditional', 'Rainfed'),
]

for ax, (label, a_name, b_name) in zip(axes3, pairs):
    overlap = valid[a_name] & valid[b_name]
    diff_map = np.full(crop_mask.shape, np.nan)
    diff_map[overlap] = alphas[a_name][overlap] - alphas[b_name][overlap]

    vabs = np.nanpercentile(np.abs(diff_map[overlap]), 95) if overlap.any() else 1
    im = ax.imshow(diff_map, cmap='RdBu_r', vmin=-vabs, vmax=vabs, origin='upper', aspect='auto')
    fig3.colorbar(im, ax=ax, shrink=0.75, label='Δα')

    d = diff_map[overlap]
    ax.set_title(f"{label}\nmean={np.nanmean(d):.3f}  std={np.nanstd(d):.3f}")
    ax.set_xlabel('Col')
    ax.set_ylabel('Row')

plt.tight_layout()
fig3.savefig(f"{OUTDIR}\\olivo_3scenarios_alpha_diff.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTDIR}\\olivo_3scenarios_alpha_diff.png")
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Alpha spatial maps side by side
# ══════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))
fig4.suptitle("Olivo — Calibrated Alpha Maps per Scenario", fontsize=13, fontweight='bold')

# Shared color range across all 3
all_alpha_vals = np.concatenate([alphas[n][valid[n]] for n in SCENARIOS])
vmin_a = np.nanpercentile(all_alpha_vals, 2)
vmax_a = np.nanpercentile(all_alpha_vals, 98)

for ax, name in zip(axes4, SCENARIOS):
    a_display = alphas[name].copy()
    a_display[~wn] = np.nan
    im = ax.imshow(a_display, cmap='plasma', vmin=vmin_a, vmax=vmax_a, origin='upper', aspect='auto')
    fig4.colorbar(im, ax=ax, shrink=0.75, label='Alpha')
    ax.set_title(f"{name} (n={int(valid[name].sum())})")
    ax.set_xlabel('Col')
    ax.set_ylabel('Row')

plt.tight_layout()
fig4.savefig(f"{OUTDIR}\\olivo_3scenarios_alpha_maps.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTDIR}\\olivo_3scenarios_alpha_maps.png")
plt.close(fig4)

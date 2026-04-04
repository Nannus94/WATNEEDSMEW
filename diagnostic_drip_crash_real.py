"""
Diagnostic: Real pixel (18,28) — fails in drip, works in trad+rainfed.
Uses actual hydro data from SMEW_Output_4Hour_MERIDA.
"""
import sys, os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, r"C:\Users\Latitude 5511\EW Models\PRIN CHANCES\EW_Model_CHANCES")
import pyEW

conv_mol = 1e6
D_0_free = pyEW.D_0()
Zr = 0.3

# ── TARGET PIXEL ──────────────────────────────────────────────────────────────
PIX = (18, 28)
r, c = PIX

# ── LOAD REAL HYDRO DATA ─────────────────────────────────────────────────────
HYDRO_BASE = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\SMEW_Output_4Hour_MERIDA"

def load_pixel_hydro(hydro_dir, row, col, years=range(1994, 2024)):
    """Load s, L, T for one pixel across all years."""
    all_s, all_L, all_T = [], [], []
    for year in years:
        for month in range(1, 13):
            for var, lst in [('s', all_s), ('L', all_L), ('T', all_T)]:
                fpath = os.path.join(hydro_dir, f"shallow_{var}_{year}_{month}.mat")
                if not os.path.exists(fpath):
                    continue
                mat = scipy.io.loadmat(fpath)
                key = [k for k in mat if not k.startswith('_')][0]
                data = mat[key].astype(float)
                lst.append(data[row, col, :])
    return np.concatenate(all_s), np.concatenate(all_L), np.concatenate(all_T)

scenarios_hydro = {}
for name, folder in [('Drip', 'olivo_drip'), ('Traditional', 'olivo_surface'), ('Rainfed', 'olivo_rainfed')]:
    hdir = os.path.join(HYDRO_BASE, folder)
    if os.path.exists(hdir):
        s, L, T = load_pixel_hydro(hdir, r, c)
        scenarios_hydro[name] = {'s': s, 'L': L, 'T': T}
        print(f"{name:15s}  s: n={len(s)}  mean={np.nanmean(s):.4f}  min={np.nanmin(s):.4f}  max={np.nanmax(s):.4f}")

# ── LOAD SOIL DATA ────────────────────────────────────────────────────────────
soil_dir = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\soil_param (1)"

def load_fill_tif(fname):
    with rasterio.open(os.path.join(soil_dir, fname)) as src:
        d = src.read(1).astype(float)
        if src.nodata is not None: d[d == src.nodata] = np.nan
    m = np.isnan(d)
    if m.any() and (~m).any():
        _, idx = distance_transform_edt(m, return_distances=True, return_indices=True)
        d[m] = d[idx[0][m], idx[1][m]]
    return d

ph_map = load_fill_tif("sicily_ph_cacl2_10km.tif")
pH_target = float(ph_map[r, c])
print(f"\nPixel {PIX}: pH_target = {pH_target:.3f}")

# ── LOAD ALPHA RESULTS ────────────────────────────────────────────────────────
DL = r"C:\Users\Latitude 5511\Downloads"
alpha_files = {
    'Drip':        f"{DL}\\FULL_OLIVO_DRIP_30Y_alpha_map_MERGED (5).tif",
    'Traditional': f"{DL}\\FULL_OLIVO_TRADITIONAL_30Y_alpha_map_MERGED (3).tif",
    'Rainfed':     f"{DL}\\FULL_OLIVO_RAINFED_30Y_alpha_map_MERGED.tif",
}
for name, path in alpha_files.items():
    with rasterio.open(path) as src:
        a = src.read(1).astype(float)
        if src.nodata is not None: a[a == src.nodata] = np.nan
    val = a[r, c]
    status = f"alpha = {val:.3f}" if not np.isnan(val) else "FAILED"
    print(f"  {name:15s}: {status}")

# ── COMPUTE D AND IC FOR EACH SCENARIO ────────────────────────────────────────
colors = {'Drip': '#2ca02c', 'Traditional': '#d62728', 'Rainfed': '#1f77b4'}
n_soil = 0.43  # typical porosity for this pixel

ic_results = {}
for name, hdata in scenarios_hydro.items():
    s = hdata['s']
    # Compute D for every timestep
    D_ts = D_0_free * (1 - s)**(10/3) * n_soil**(4/3)
    D_mean = float(np.mean(D_ts[D_ts > 0]))

    # Simulate what carbon_respiration_dynamic would give
    # f_d depends on s_h, s_w, s_fc — use typical olivo values
    s_h, s_w, s_fc = 0.17, 0.35, 0.70
    f_d = np.zeros_like(s)
    f_d[s <= s_h] = 0.001
    mask1 = (s > s_h) & (s <= s_w)
    f_d[mask1] = (s[mask1] - s_h) / (s_w - s_h)
    mask2 = (s > s_w) & (s <= s_fc)
    f_d[mask2] = 1.0
    mask3 = (s > s_fc) & (s <= 1.0)
    f_d[mask3] = (1 - s[mask3]) / (1 - s_fc)

    f_d_mean = float(np.mean(f_d))

    # r_het_mean ~ r_het_in * (unit conversion). Use typical value ~70000
    r_het_mean = 70000.0
    r_aut_mean = 50000.0

    # pH_to_conc with mean D
    T_K = np.array([288.15])
    k1, k2, k_w, k_H = pyEW.K_C(T_K, conv_mol)
    CO2_atm = pyEW.CO2_atm(conv_mol)
    Z_CO2 = 0.15
    CO2_air = (r_het_mean + r_aut_mean) / (D_mean * 1000 / Z_CO2) + CO2_atm
    CO2_w = k_H[0] * CO2_air
    H = 10**(-pH_target) * conv_mol
    HCO3 = k1[0] * CO2_w / H
    Alk = HCO3 + 2 * k2[0] * k1[0] * CO2_w / H**2 - H + k_w[0] / H
    Ca = 0.60 * (Alk + 2800) / 2

    ic_results[name] = {
        's': s, 'D_ts': D_ts, 'D_mean': D_mean, 'f_d': f_d, 'f_d_mean': f_d_mean,
        'CO2_air': CO2_air, 'Ca': Ca, 'Alk': Alk,
    }
    print(f"  {name:15s}  s_mean={np.mean(s):.4f}  D_mean={D_mean:.6f}  CO2_air={CO2_air:.0f}  Ca_init={Ca:.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)
fig.suptitle(f"Pixel ({r},{c}) — Why drip crashes, traditional+rainfed work\n"
             f"pH_target = {pH_target:.3f}   |   Drip: FAILED   |   Traditional: α=1.920   |   Rainfed: α=2.500",
             fontsize=13, fontweight='bold', y=0.98)

# ── (a) Soil moisture time series (first 3 years) ────────────────────────────
ax_s = fig.add_subplot(gs[0, 0])
days_show = 365 * 3  # 3 years
steps_per_day = 6  # 4h resolution
for name in ['Drip', 'Traditional', 'Rainfed']:
    s = ic_results[name]['s']
    n_show = min(days_show * steps_per_day, len(s))
    t_days = np.arange(n_show) / steps_per_day
    ax_s.plot(t_days, s[:n_show], color=colors[name], lw=0.5, alpha=0.8, label=name)

ax_s.set_xlabel('Day', fontsize=11)
ax_s.set_ylabel('Soil moisture s [-]', fontsize=11)
ax_s.set_title('(a) Soil moisture — drip stays wet year-round', fontsize=11)
ax_s.legend(fontsize=10)
ax_s.grid(True, alpha=0.3)
ax_s.set_ylim(0.2, 1.05)

# ── (b) Diffusivity D time series (first 3 years) ────────────────────────────
ax_d = fig.add_subplot(gs[0, 1])
for name in ['Drip', 'Traditional', 'Rainfed']:
    D_ts = ic_results[name]['D_ts']
    n_show = min(days_show * steps_per_day, len(D_ts))
    t_days = np.arange(n_show) / steps_per_day
    ax_d.plot(t_days, D_ts[:n_show], color=colors[name], lw=0.5, alpha=0.8, label=name)

# Mark mean D for each
for name in ['Drip', 'Traditional', 'Rainfed']:
    ax_d.axhline(ic_results[name]['D_mean'], color=colors[name], ls='--', lw=2, alpha=0.7)

ax_d.set_xlabel('Day', fontsize=11)
ax_d.set_ylabel('D [m²/d]', fontsize=11)
ax_d.set_title('(b) CO₂ diffusivity — drip D_mean is 4× lower', fontsize=11)
ax_d.legend(fontsize=10)
ax_d.grid(True, alpha=0.3)
ax_d.set_yscale('log')
ax_d.set_ylim(1e-4, 0.1)

# ── (c) Moisture decomposition factor f_d ─────────────────────────────────────
ax_fd = fig.add_subplot(gs[1, 0])
for name in ['Drip', 'Traditional', 'Rainfed']:
    f_d = ic_results[name]['f_d']
    n_show = min(days_show * steps_per_day, len(f_d))
    t_days = np.arange(n_show) / steps_per_day
    ax_fd.plot(t_days, f_d[:n_show], color=colors[name], lw=0.5, alpha=0.8, label=name)
    ax_fd.axhline(ic_results[name]['f_d_mean'], color=colors[name], ls='--', lw=2, alpha=0.7)

ax_fd.set_xlabel('Day', fontsize=11)
ax_fd.set_ylabel('f_d [-]', fontsize=11)
ax_fd.set_title('(c) Moisture limitation f_d — drip suppresses decomposition (s > s_fc)',
                fontsize=11)
ax_fd.legend(fontsize=10)
ax_fd.grid(True, alpha=0.3)
ax_fd.set_ylim(-0.05, 1.1)

# ── (d) Bar chart: D_mean, CO2_air, Ca_init ──────────────────────────────────
ax_bar = fig.add_subplot(gs[1, 1])
x = np.arange(3)
width = 0.25
names = ['Drip', 'Traditional', 'Rainfed']

# Normalize to rainfed for comparison
D_vals = [ic_results[n]['D_mean'] for n in names]
CO2_vals = [ic_results[n]['CO2_air'] for n in names]
Ca_vals = [ic_results[n]['Ca'] for n in names]

ax_bar2 = ax_bar.twinx()

bars1 = ax_bar.bar(x - width, [d * 1000 for d in D_vals], width, color=[colors[n] for n in names],
                    alpha=0.7, edgecolor='black', linewidth=0.5)
ax_bar.set_ylabel('D_mean × 1000', fontsize=10)

bars2 = ax_bar2.bar(x, Ca_vals, width, color=[colors[n] for n in names],
                     alpha=0.4, edgecolor='black', linewidth=0.5, hatch='//')
ax_bar2.set_ylabel('Ca_init [µmol/L]', fontsize=10, color='darkred')
ax_bar2.tick_params(axis='y', labelcolor='darkred')

# Add value labels
for i, (d, ca) in enumerate(zip(D_vals, Ca_vals)):
    ax_bar.text(i - width, d * 1000 + 0.3, f'D={d:.4f}', ha='center', fontsize=8, fontweight='bold')
    ax_bar2.text(i, ca + 200, f'Ca={ca:.0f}', ha='center', fontsize=8, color='darkred', fontweight='bold')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(names, fontsize=11)
ax_bar.set_title('(d) Mean D and initial Ca per scenario\n(hatched = Ca from pH_to_conc)', fontsize=11)

# ── (e) The crash mechanism ───────────────────────────────────────────────────
ax_mech = fig.add_subplot(gs[2, 0])
ax_mech.axis('off')

# Show the chain for each scenario
y_start = 0.95
for name in names:
    ic = ic_results[name]
    color = colors[name]
    alpha_str = "FAILED" if name == "Drip" else f"α = {1.920 if name=='Traditional' else 2.500:.3f}"

    text = (f"{name} (s̄={np.mean(ic['s']):.3f})\n"
            f"  D_mean = {ic['D_mean']:.5f} m²/d\n"
            f"  → CO₂_air = {ic['CO2_air']:.0f} µmol/L\n"
            f"  → Ca_init = {ic['Ca']:.0f} µmol/L\n"
            f"  → Calibration: {alpha_str}")

    ax_mech.text(0.05, y_start, text, transform=ax_mech.transAxes,
                 fontsize=11, fontfamily='monospace', color=color, fontweight='bold',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.1))
    y_start -= 0.35

# ── (f) The fix demonstration ─────────────────────────────────────────────────
ax_fix = fig.add_subplot(gs[2, 1])

D_min_vals = np.linspace(0.001, 0.02, 50)
Ca_drip_clamped = []
Ca_rain_clamped = []

for D_min in D_min_vals:
    D_drip = max(ic_results['Drip']['D_mean'], D_min)
    D_rain = max(ic_results['Rainfed']['D_mean'], D_min)

    CO2_d = (70000 + 50000) / (D_drip * 1000 / 0.15) + CO2_atm
    CO2_r = (70000 + 50000) / (D_rain * 1000 / 0.15) + CO2_atm

    H = 10**(-pH_target) * conv_mol
    HCO3_d = k1[0] * k_H[0] * CO2_d / H
    Alk_d = HCO3_d + 2800
    Ca_drip_clamped.append(0.60 * Alk_d / 2)

    HCO3_r = k1[0] * k_H[0] * CO2_r / H
    Alk_r = HCO3_r + 2800
    Ca_rain_clamped.append(0.60 * Alk_r / 2)

ax_fix.plot(D_min_vals * 1000, Ca_drip_clamped, color=colors['Drip'], lw=2.5, label='Drip')
ax_fix.plot(D_min_vals * 1000, Ca_rain_clamped, color=colors['Rainfed'], lw=2.5, label='Rainfed')
ax_fix.axhline(5000, color='red', ls='--', alpha=0.5, label='Crash threshold (~5000)')
ax_fix.axhline(2000, color='gray', ls=':', alpha=0.5, label='Typical Ca')
ax_fix.axvline(5, color='black', ls='--', alpha=0.7, label='Proposed D_min = 0.005')

# Mark current D_mean for each
ax_fix.axvline(ic_results['Drip']['D_mean'] * 1000, color=colors['Drip'], ls=':', alpha=0.5)
ax_fix.axvline(ic_results['Rainfed']['D_mean'] * 1000, color=colors['Rainfed'], ls=':', alpha=0.5)

ax_fix.set_xlabel('D_min × 1000 [m²/d]', fontsize=11)
ax_fix.set_ylabel('Ca_init [µmol/L]', fontsize=11)
ax_fix.set_title('(f) Effect of D_min clamp on initial Ca\nD_min=0.005 keeps Ca < 3500 for all scenarios',
                  fontsize=11)
ax_fix.legend(fontsize=9)
ax_fix.grid(True, alpha=0.3)
ax_fix.set_ylim(0, 15000)

plt.savefig(r"C:\Users\Latitude 5511\Downloads\diagnostic_drip_crash_real.png",
            dpi=150, bbox_inches='tight')
print("Saved: diagnostic_drip_crash_real.png")
plt.close()

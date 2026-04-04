"""
Diagnostic: Why drip calibration crashes at wet pixels.
Shows the chain: low D → high CO2 → extreme Ca → solver crash.
Compares drip vs rainfed vs traditional for the same pixel.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, r"C:\Users\Latitude 5511\EW Models\PRIN CHANCES\EW_Model_CHANCES")
import pyEW

conv_mol = 1e6
D_0_free = pyEW.D_0()  # free-air diffusion ~1.38 m²/d

# ── pH_to_conc wrapper ───────────────────────────────────────────────────────
def compute_IC(pH, r_het, r_aut, D_val, temp=15.0, An_0=2800):
    """Run pH_to_conc and return all intermediate variables."""
    T_K = np.array([temp + 273.15])
    k1, k2, k_w, k_H = pyEW.K_C(T_K, conv_mol)
    CO2_atm = pyEW.CO2_atm(conv_mol)
    Z_CO2 = 0.15

    CO2_air = (r_het + r_aut) / (D_val * 1000 / Z_CO2) + CO2_atm
    CO2_w = k_H[0] * CO2_air
    H = 10**(-pH) * conv_mol
    HCO3 = k1[0] * CO2_w / H
    CO3 = k2[0] * k1[0] * CO2_w / H**2
    Alk = HCO3 + 2 * CO3 - H + k_w[0] / H
    charge = Alk + An_0
    Ca = 0.60 * charge / 2
    Mg = 0.30 * charge / 2

    return {
        'CO2_air': CO2_air, 'CO2_w': CO2_w, 'HCO3': HCO3,
        'Alk': Alk, 'charge': charge, 'Ca': Ca, 'Mg': Mg, 'H': H,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: D and CO2_air as function of soil moisture
# ══════════════════════════════════════════════════════════════════════════════
s_range = np.linspace(0.3, 0.95, 200)
n_soil = 0.43  # typical porosity

D_range = D_0_free * (1 - s_range)**(10/3) * n_soil**(4/3)

# Typical r_het and r_aut (annual mean from carbon_respiration_dynamic)
r_het_typ = 70000.0  # µmol/m²/d
r_aut_typ = 50000.0
Z_CO2 = 0.15
CO2_atm = pyEW.CO2_atm(conv_mol)
CO2_air_range = (r_het_typ + r_aut_typ) / (D_range * 1000 / Z_CO2) + CO2_atm

# Ca from pH_to_conc at pH=7.2 for each s
Ca_range = np.zeros_like(s_range)
for i, s_val in enumerate(s_range):
    ic = compute_IC(7.2, r_het_typ, r_aut_typ, D_range[i])
    Ca_range[i] = ic['Ca']

# Mark real pixel values
# Pixel (18,28): drip s_mean=0.749, trad s_mean=0.683, rainfed s_mean=0.642
scenarios = {
    'Drip':        {'s_mean': 0.749, 'color': '#2ca02c', 'marker': 'o'},
    'Traditional': {'s_mean': 0.683, 'color': '#d62728', 'marker': 's'},
    'Rainfed':     {'s_mean': 0.642, 'color': '#1f77b4', 'marker': '^'},
}

fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)
fig.suptitle("Why drip calibration crashes at wet pixels\nExample: Pixel (18,28) — Olivo, Sicily",
             fontsize=14, fontweight='bold', y=0.98)

# Panel 1: D vs s
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(s_range, D_range, 'k-', lw=2)
for name, info in scenarios.items():
    D_val = D_0_free * (1 - info['s_mean'])**(10/3) * n_soil**(4/3)
    ax1.plot(info['s_mean'], D_val, info['marker'], color=info['color'], ms=12, zorder=5,
             label=f"{name} (s={info['s_mean']:.3f}, D={D_val:.4f})")
ax1.set_xlabel('Mean soil moisture s [-]', fontsize=11)
ax1.set_ylabel('Diffusivity D [m²/d]', fontsize=11)
ax1.set_title('(a) CO₂ diffusivity drops exponentially with s', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhspan(0, 0.003, color='red', alpha=0.08, label='Crash zone')
ax1.set_ylim(1e-4, 0.1)

# Panel 2: CO2_air vs s
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(s_range, CO2_air_range, 'k-', lw=2)
for name, info in scenarios.items():
    D_val = D_0_free * (1 - info['s_mean'])**(10/3) * n_soil**(4/3)
    CO2 = (r_het_typ + r_aut_typ) / (D_val * 1000 / Z_CO2) + CO2_atm
    ax2.plot(info['s_mean'], CO2, info['marker'], color=info['color'], ms=12, zorder=5,
             label=f"{name} (CO₂={CO2:.0f})")
ax2.axhline(100, color='gray', ls='--', alpha=0.5, label='Typical soil CO₂ (~100)')
ax2.set_xlabel('Mean soil moisture s [-]', fontsize=11)
ax2.set_ylabel('CO₂_air [µmol/L]', fontsize=11)
ax2.set_title('(b) Soil CO₂ explodes at high s (low D)', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: Ca from pH_to_conc vs s
ax3 = fig.add_subplot(gs[1, 0])
ax3.semilogy(s_range, Ca_range, 'k-', lw=2)
for name, info in scenarios.items():
    D_val = D_0_free * (1 - info['s_mean'])**(10/3) * n_soil**(4/3)
    ic = compute_IC(7.2, r_het_typ, r_aut_typ, D_val)
    ax3.plot(info['s_mean'], ic['Ca'], info['marker'], color=info['color'], ms=12, zorder=5,
             label=f"{name} (Ca={ic['Ca']:.0f})")
ax3.axhline(2000, color='gray', ls='--', alpha=0.5, label='Typical soil Ca (~2000)')
ax3.axhspan(5000, 1e6, color='red', alpha=0.08)
ax3.text(0.35, 8000, 'Solver crash zone\n(extreme initial Ca)', fontsize=9, color='red', alpha=0.7)
ax3.set_xlabel('Mean soil moisture s [-]', fontsize=11)
ax3.set_ylabel('Initial Ca [µmol/L]', fontsize=11)
ax3.set_title('(c) pH_to_conc derives extreme Ca to maintain pH=7.2 with high CO₂', fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(100, 50000)

# Panel 4: The inconsistency — what pH WOULD you get with that CO2?
ax4 = fig.add_subplot(gs[1, 1])
# For a given CO2_air, what pH would you get with typical Ca=1000?
T_K = np.array([15.0 + 273.15])
k1, k2, k_w, k_H = pyEW.K_C(T_K, conv_mol)
CO2_w_range = k_H[0] * CO2_air_range
# From carbonate equilibrium: pH ≈ -log10(k1 * CO2_w / HCO3)
# With typical Alk~2000: HCO3 ≈ Alk, so H ≈ k1 * CO2_w / Alk
Alk_typ = 2000.0
H_equil = k1[0] * CO2_w_range / Alk_typ
pH_equil = -np.log10(H_equil / conv_mol)
pH_equil = np.clip(pH_equil, 4, 9)

ax4.plot(s_range, pH_equil, 'k-', lw=2, label='Equilibrium pH (with typical Alk)')
ax4.axhline(7.2, color='orange', ls='--', lw=2, label='ESDAC target pH = 7.2')
for name, info in scenarios.items():
    D_val = D_0_free * (1 - info['s_mean'])**(10/3) * n_soil**(4/3)
    CO2 = (r_het_typ + r_aut_typ) / (D_val * 1000 / Z_CO2) + CO2_atm
    CO2_w = k_H[0] * CO2
    H_eq = k1[0] * CO2_w / Alk_typ
    pH_eq = -np.log10(H_eq / conv_mol)
    ax4.plot(info['s_mean'], pH_eq, info['marker'], color=info['color'], ms=12, zorder=5,
             label=f"{name} (pH_eq={pH_eq:.1f})")

ax4.fill_between(s_range, pH_equil, 7.2, where=pH_equil < 7.2,
                  color='red', alpha=0.15, label='pH gap → Ca inflated to compensate')
ax4.set_xlabel('Mean soil moisture s [-]', fontsize=11)
ax4.set_ylabel('pH', fontsize=11)
ax4.set_title('(d) Equilibrium pH drops below target at high s\n→ pH_to_conc inflates Ca to force pH=7.2',
              fontsize=11)
ax4.legend(fontsize=8, loc='lower left')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(4.5, 8.5)

# Panel 5: The fix — D_init = max(D_mean, D_min)
ax5 = fig.add_subplot(gs[2, 0])
D_min_values = [0.003, 0.005, 0.01]
for D_min in D_min_values:
    D_clamped = np.maximum(D_range, D_min)
    Ca_clamped = np.zeros_like(s_range)
    for i in range(len(s_range)):
        ic = compute_IC(7.2, r_het_typ, r_aut_typ, D_clamped[i])
        Ca_clamped[i] = ic['Ca']
    ax5.plot(s_range, Ca_clamped, lw=1.5, label=f'D_min={D_min}')

ax5.plot(s_range, Ca_range, 'k--', lw=1, alpha=0.5, label='No clamp (current)')
ax5.axhline(2000, color='gray', ls=':', alpha=0.5)
ax5.set_xlabel('Mean soil moisture s [-]', fontsize=11)
ax5.set_ylabel('Initial Ca [µmol/L]', fontsize=11)
ax5.set_title('(e) Fix: clamp D_init = max(D_mean, D_min)\nKeeps Ca reasonable for initialization only',
              fontsize=11)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_ylim(0, 8000)

# Panel 6: Summary text
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')
summary = """
SUMMARY

The initialization function pH_to_conc derives initial
cation concentrations from:
  • Target pH (ESDAC annual mean = 7.2)
  • Soil CO₂ (computed from respiration / diffusivity)

For drip irrigation at wet pixels:
  1. s_mean ≈ 0.75 → D ≈ 0.002 m²/d (very low)
  2. Low D → CO₂_air ≈ 7000 µmol/L (extreme)
  3. pH_to_conc asks: "Ca needed for pH=7.2 at CO₂=7000?"
     Answer: Ca = 14,000 µmol/L (vs typical ~1000)
  4. fsolve gets Ca=14,000 as starting point → diverges
     to negative H⁺ and CO₂ → pH=NaN from step 1

The same pixel under rainfed:
  1. s_mean ≈ 0.64 → D ≈ 0.015 m²/d (5× higher)
  2. CO₂_air ≈ 1200 → Ca ≈ 2000 → solver works fine

FIX: Clamp D_init ≥ 0.005 for pH_to_conc only.
The 30-year simulation computes D dynamically —
the clamp only affects the starting point.
"""
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(r"C:\Users\Latitude 5511\Downloads\diagnostic_drip_crash.png",
            dpi=150, bbox_inches='tight')
print("Saved: C:\\Users\\Latitude 5511\\Downloads\\diagnostic_drip_crash.png")
plt.close()

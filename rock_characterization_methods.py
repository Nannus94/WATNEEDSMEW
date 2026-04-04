"""
Rock Powder Characterization — Italy Mine Waste (Mt. Etna basalt)
================================================================
This script documents and visualizes the rock powder properties
used in the EW simulation, as they would be presented in a paper.

Sources:
  - Mineralogy: SGS Report 24/3619, XRD Rietveld refinement (Table 10)
  - PSD coarse (>75um): SGS Report, sieve analysis (Table 3)
  - PSD fine (<75um): Malvern Mastersizer laser diffraction
  - Chemistry: SGS Report, XRF + ICP-OES (Tables 5-6)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════════════════════
# 1. MINERALOGY (XRD, SGS Report Table 10)
# ══════════════════════════════════════════════════════════════════════════════
# Rietveld refinement of pulverised sample — mass %
mineralogy_full = {
    'Labradorite':  42.6,   # (Ca,Na)(Al,Si)4O8 — Ca-rich plagioclase
    'Albite':       18.6,   # NaAlSi3O8 — Na-rich plagioclase
    'Diopside':     18.4,   # CaMgSi2O6 — pyroxene
    'Muscovite':     9.6,   # KAl2(Si3Al)O10(OH)2 — mica (not modeled)
    'Anorthite':     5.2,   # CaAl2Si2O8 — Ca-plagioclase endmember
    'Ilmenite':      2.6,   # (Fe,Ti)2O3 — oxide (not reactive)
    'Magnetite':     1.6,   # Fe3O4 — oxide (not reactive)
    'Sanidine':      0.7,   # KAlSi3O8 — K-feldspar (not modeled)
    'Chlorite':      0.6,   # sheet silicate (not modeled)
}

# Minerals modeled in pyEW (have dissolution kinetics)
modeled = {
    'Labradorite': 42.6,
    'Albite':      18.6,
    'Diopside':    18.4,
    'Anorthite':    5.2,
}
# Total reactive fraction: 84.8%
# Non-reactive: muscovite + oxides + minor = 15.2%

# ══════════════════════════════════════════════════════════════════════════════
# 2. PARTICLE SIZE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
#
# The PSD was measured in two steps:
#
# (a) DRY SIEVE ANALYSIS (SGS Report, Table 3)
#     The bulk sample (554 g) was passed through sieves at:
#     850, 600, 212, 150, 75 um
#     Each fraction was weighed → gives mass % in 5 coarse bins
#     The fraction passing 75 um (21.44% of total) was sent to laser diffraction.
#
# (b) LASER DIFFRACTION (Malvern Mastersizer, <75 um fraction)
#     Measures cumulative volume % as a function of particle diameter.
#     Result is a continuous curve from 0.01 to ~100 um.
#     Key percentiles: Dv(10)=5.47, Dv(50)=38.7, Dv(90)=83.8 um
#
# To combine them:
#   - The 5 sieve fractions give the COARSE bins directly (>75 um)
#   - The Malvern cumulative curve is discretized into FINE bins (<75 um)
#   - The Malvern % are WITHIN the <75 um fraction, so they must be
#     scaled to the total sample: mass%_total = mass%_within_fines × 0.2144
#   - The Malvern also measures some material >75 um (the tail up to ~100 um),
#     but this overlaps with the sieve 75-150 um bin, so we exclude it.

# --- (a) Sieve data: mass % of TOTAL sample ---
sieve = {
    # (lower_um, upper_um): mass % of total
    (75,   150): 13.83,
    (150,  212): 10.00,
    (212,  600): 32.39,
    (600,  850):  8.81,
    (850, 1000): 13.53,   # reported as ">850", upper bound ~1000 um
}

# --- (b) Malvern cumulative curve (selected breakpoints) ---
# These are cumulative VOLUME % WITHIN the <75 um fraction
# I choose bin edges at natural breakpoints of the cumulative curve
malvern_cumulative = {
    #  um:  cum_vol_%
    0.0:    0.00,
    1.0:    1.96,    # very fine clay-size
    5.0:    9.64,    # ~Dv(10) = 5.47 um
    15.0:  21.97,
    30.0:  40.64,    # ~near Dv(50) = 38.7 um
    50.0:  65.33,
    75.0:  85.26,    # sieve cutoff
}
fines_mass_pct = 21.44  # % of total sample that is <75 um

# Convert Malvern cumulative to differential bins, scaled to total sample
sizes = sorted(malvern_cumulative.keys())
fine_bins = []
for i in range(1, len(sizes)):
    d_lo = sizes[i-1]
    d_hi = sizes[i]
    cum_lo = malvern_cumulative[d_lo]
    cum_hi = malvern_cumulative[d_hi]
    # Differential % within fines
    diff_within_fines = cum_hi - cum_lo
    # Scale to total sample
    mass_pct_total = diff_within_fines / 100.0 * fines_mass_pct
    fine_bins.append((d_lo, d_hi, mass_pct_total))

# --- Combine fine + coarse ---
all_bins = fine_bins + [(lo, hi, pct) for (lo, hi), pct in sorted(sieve.items())]
total_recovered = sum(b[2] for b in all_bins)

# Representative diameter = geometric mean of bin edges
# d_repr = sqrt(d_lo * d_hi), in meters
# For bin starting at 0: use 0.5 um as effective lower bound
d_repr = []
psd_frac = []
for lo, hi, pct in all_bins:
    lo_eff = max(lo, 0.5)  # avoid sqrt(0)
    d_repr.append(np.sqrt(lo_eff * hi) * 1e-6)  # um -> m
    psd_frac.append(pct / total_recovered)  # normalize to sum=1

D_IN_PSD = np.array(d_repr)
PSD = np.array(psd_frac)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

# --- Panel (a): Mineralogy pie chart ---
ax1 = fig.add_subplot(gs[0, 0])
labels = list(mineralogy_full.keys())
values = list(mineralogy_full.values())
colors_min = plt.cm.Set3(np.linspace(0, 1, len(labels)))
# Highlight modeled minerals
explode = [0.05 if m in modeled else 0 for m in labels]
wedges, texts, autotexts = ax1.pie(values, labels=labels, autopct='%1.1f%%',
                                     colors=colors_min, explode=explode,
                                     textprops={'fontsize': 8})
for i, m in enumerate(labels):
    if m not in modeled:
        autotexts[i].set_alpha(0.4)
        texts[i].set_alpha(0.4)
ax1.set_title('(a) Mineralogy (XRD)\nBold = modeled in pyEW', fontsize=11)

# --- Panel (b): PSD bar chart ---
ax2 = fig.add_subplot(gs[0, 1])
bin_labels = [f'{lo:.0f}-{hi:.0f}' for lo, hi, _ in all_bins]
bin_pcts = [pct for _, _, pct in all_bins]
colors_psd = ['#4292c6' if hi <= 75 else '#fd8d3c' for _, hi, _ in all_bins]
bars = ax2.bar(range(len(all_bins)), bin_pcts, color=colors_psd, edgecolor='black', linewidth=0.5)
ax2.set_xticks(range(len(all_bins)))
ax2.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Mass % of total sample', fontsize=10)
ax2.set_xlabel('Size class (um)', fontsize=10)
ax2.set_title('(b) Particle Size Distribution\nBlue = Malvern (<75um), Orange = Sieve (>75um)', fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# --- Panel (c): Cumulative PSD ---
ax3 = fig.add_subplot(gs[1, 0])
# Build cumulative from bins
cum_pct = np.cumsum([pct for _, _, pct in all_bins])
d_upper = [hi for _, hi, _ in all_bins]
ax3.plot(d_upper, cum_pct, 'ko-', lw=2, ms=6)
ax3.axhline(50, color='gray', ls='--', alpha=0.5, label='50%')
ax3.axvline(75, color='red', ls=':', alpha=0.7, label='Sieve/Malvern boundary (75 um)')
ax3.set_xlabel('Particle diameter (um)', fontsize=10)
ax3.set_ylabel('Cumulative mass %', fontsize=10)
ax3.set_title('(c) Cumulative PSD', fontsize=11)
ax3.set_xscale('log')
ax3.set_xlim(0.5, 1500)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)

# --- Panel (d): Summary table ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

summary = """ROCK POWDER SUMMARY
Italy Mine Waste (Mt. Etna basalt byproduct)

Source:     SGS Report 24/3619, Malvern Mastersizer
Chemistry:  SiO2=48.8%, Al2O3=19.3%, CaO=10.0%,
            Fe2O3=9.9%, MgO=4.1%, Na2O=4.4%
pH (solid): 9.9

Modeled minerals (84.8% of rock mass):
  Labradorite   42.6%   (Ca,Na)(Al,Si)4O8
  Albite        18.6%   NaAlSi3O8
  Diopside      18.4%   CaMgSi2O6
  Anorthite      5.2%   CaAl2Si2O8

Non-reactive (15.2%):
  Muscovite 9.6%, Ilmenite 2.6%, Magnetite 1.6%,
  Sanidine 0.7%, Chlorite 0.6%

PSD: 11 bins (6 Malvern + 5 sieve)
  Dv(50) of <75um fraction: 38.7 um
  Dominant bin: 212-600 um (32.4% of mass)
  Fine fraction (<75um): 21.4% of mass

Application: 4000 g/m2 (40 t/ha) at years 0, 10, 20
SSA_in: not used (fractal model from PSD)
"""
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=9.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Rock Powder Characterization — Mt. Etna Basalt Mine Waste',
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig(r'C:\Users\Latitude 5511\Downloads\rock_characterization.png',
            dpi=150, bbox_inches='tight')
print('Saved: rock_characterization.png')
plt.close()

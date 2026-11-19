"""
End-to-end example: EPOS + RRNG -> correlation table, correlation
histogram (m/n and ToF versions), and a dissociation-track overlay,
reproducing Saxey (2011) Sections 2.1-2.3 for a SiO2 dataset.

Usage:
    python run_analysis.py dataset.epos dataset.rrng
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from apt_io import read_epos, read_rrng, assign_species_codes
from correlation_table import counts_to_matrix, deviation_matrix
from correlation_histogram import (
    mn_correlation_histogram, tof_correlation_histogram, dissociation_track,
    dissociation_occurrence_report,
)
from fast_pairs import build_pair_counts_fast, build_pair_stream_fast
import csv

# Detector efficiency for YOUR instrument - check your acquisition
# settings/documentation and update this (paper used 0.55 and 0.37 for
# its two instruments as examples).
DETECTOR_EFFICIENCY = 0.62

# NOTE: this script uses the *_fast functions throughout (see fast_pairs.py),
# which use integer species codes instead of Python string labels and
# vectorized numpy indexing instead of per-pulse Python loops. This is
# what makes it practical to run on multi-GB ePOS files (tens of millions
# of ions) inside a normal Jupyter session - the plain string/groupby
# versions in correlation_table.py / correlation_histogram.py are fine for
# small test data but will be slow and memory-heavy at full dataset scale.


def main(epos_path, rrng_path):
    # ---- Load data -------------------------------------------------
    data, pulse_number = read_epos(epos_path)
    ranges = read_rrng(rrng_path)
    codes, code_names = assign_species_codes(data['mn'], ranges)

    print(f"Loaded {len(data)} ions across {pulse_number[-1]} pulses.")
    print(f"Ranges defined: {code_names}")

    # ---- Section 2.1: correlation table ----------------------------
    pair_counts = build_pair_counts_fast(pulse_number, codes, code_names, min_multiplicity=2)
    O, species_order = counts_to_matrix(pair_counts)
    Z, E = deviation_matrix(O)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Z, cmap='RdYlGn_r', vmin=-50, vmax=50)
    ax.set_xticks(range(len(species_order)))
    ax.set_yticks(range(len(species_order)))
    ax.set_xticklabels(species_order, rotation=90)
    ax.set_yticklabels(species_order)
    for i in range(len(species_order)):
        for j in range(len(species_order)):
            if Z[i, j] != 0:
                ax.text(j, i, f"{Z[i, j]:+.0f}", ha='center', va='center', fontsize=7)
    plt.colorbar(im, label='Deviation score  Z_ij')
    ax.set_title('Correlation table (Section 2.1)')
    plt.tight_layout()
    plt.savefig('correlation_table.png', dpi=150)
    print("Saved correlation_table.png")

    # ---- Section 2.2: correlation histogram (m/n space) ------------
    mn_i, mn_j, tof_i, tof_j = build_pair_stream_fast(pulse_number, data['mn'], data['tof'])

    H_mn, xedges, yedges = mn_correlation_histogram(mn_i, mn_j, m_range=(0, 100), bin_width=0.05)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(xedges, yedges, H_mn.T,
                        norm=LogNorm(vmin=1, vmax=H_mn.max()), cmap='viridis')
    ax.set_xlabel('Mass-charge ratio, m1 (Da)')
    ax.set_ylabel('Mass-charge ratio, m2 (Da)')
    plt.colorbar(im, label='Counts')
    ax.set_title('Correlation histogram, m/n space (Section 2.2)')

    # Zoom to the actual region of interest - without this, matplotlib
    # auto-scales to fit the full data range (which for SiO2 spans up
    # to ~1500 Da from large cluster ions), squashing the low-mass
    # region where SiO2 dissociation tracks actually live.
    ax.set_xlim(80, 110)
    ax.set_ylim(80, 110)

    # ---- Section 2.3: overlay CANDIDATE dissociation tracks --------
    # Each entry: (parent m/z, daughter1 m/z, daughter2 m/z, label, color)
    # Add/remove candidates here - see chat for how these were chosen
    # from the correlation table hotspots and mass/charge balance.
    candidates = [
       # (30.0, 44.0, 16.0, 'SiO2(2+) -> SiO+ + O+',  'deeppink'),
       # (22.0, 28.0, 16.0, 'SiO(2+) -> Si+ + O+',  'cyan'),
        #(30.0, 28.0, 16.0, 'SiO2(2+) -> Si+ + O+ [+ neutral O]',  'cyan'),
       # (30.0, 28.0, 32.0, 'SiO2(2+) -> Si+ + O2+',  'orange'),
        #(30.0, 14.0, 32.0, 'SiO2(2+) -> Si(2+) + [neutral O2]',  'cyan'),
        #(22.5, 44.0, 1.0,  'SiOH(2+) -> SiO+ + H+',  'cyan'),
        #(20.0, 14.0, 16.0, 'SiO2(3+) -> Si(2+) + O+ [+ neutral O]', 'red'),
       # (20.0, 28.0, 16.0, 'SiO2(3+) -> Si+ + O(2+) [+ neutral O]', 'red'),
    ]


    plot_lo, plot_hi = 0, 60  # keep in sync with ax.set_xlim/ylim below

    for mp, m1, m2, label, color in candidates:
        m1_prime, m2_prime = dissociation_track(mp, m1, m2)
        valid = (
            (m1_prime >= plot_lo) & (m1_prime <= plot_hi) &
            (m2_prime >= plot_lo) & (m2_prime <= plot_hi)
        )
        ax.plot(m1_prime[valid], m2_prime[valid], color=color, lw=0.5, label=label)
        ax.plot(m2_prime[valid], m1_prime[valid], color=color, lw=0.5)  # symmetric counterpart

    ax.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.savefig('correlation_histogram_mn.png', dpi=150)
    print("Saved correlation_histogram_mn.png")

    # ---- Section 3.2: straightened ToF-space histogram --------------
    H_t, txedges, tyedges = tof_correlation_histogram(tof_i, tof_j, n_bins=500)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(txedges, tyedges, H_t.T,
                        norm=LogNorm(vmin=1, vmax=H_t.max()), cmap='viridis')
    ax.set_xlabel('t2 - t1  (ns)')
    ax.set_ylabel('t2  (ns)')
    plt.colorbar(im, label='Counts')
    ax.set_title('Correlation histogram, ToF space (Section 3.2)')
    plt.tight_layout()
    plt.savefig('correlation_histogram_tof.png', dpi=150)
    print("Saved correlation_histogram_tof.png")

    # ---- Occurrence/ion-loss estimate for each candidate, as CSV ----
    csv_rows = []
    for mp, m1, m2, label, color in candidates:
        print(f"\n=== {label} ===")
        est = dissociation_occurrence_report(
            mp, m1, m2, mn_i, mn_j,
            detector_efficiency=DETECTOR_EFFICIENCY,
            total_ions_detected=len(data),
            tolerance=0.5,
        )
        csv_rows.append({
            'label': label,
            'mp': mp, 'm1': m1, 'm2': m2,
            'detector_efficiency': DETECTOR_EFFICIENCY,
            'n_track_observed': est['n_track_observed'],
            'n_true_dissociation_events': est['n_true_dissociation_events'],
            'total_ions_evaporated_estimate': est.get('total_ions_evaporated_estimate'),
            'rate_ppm': est.get('rate_ppm'),
        })

    csv_path = 'dissociation_occurrence.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved {csv_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python run_analysis.py dataset.epos dataset.rrng")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

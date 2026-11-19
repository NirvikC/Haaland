"""
Counts ions per element (and each element's fraction) within BOTH:
  1. A small angular aperture around a chosen pole center - the SAME
     point-projection model used throughout this codebase
     (select_aperture_angular / angular_radius_from_mm, from
     sequence_periodicity.py), and
  2. A PERCENTAGE WINDOW of the total sample depth, using the ION COUNT
     NUMBER (i.e. the ion's row index in the ePOS file, 0 = first ion
     recorded, N-1 = last ion recorded for an N-ion file) as a no-
     reconstruction depth proxy, NOT the reconstructed z coordinate.
     0% = the very first ion in the WHOLE dataset, 100% = the very last -
     this is deliberately based on the FULL dataset's ion count range,
     not just whatever falls inside the aperture, so changing the
     pole/aperture doesn't shift what "50% depth" means.

     NOTE: ion count number is NOT the same thing as a detection EVENT
     number. An "event" is one pulse, and can contain several ions at
     once (a multi-hit event) - so the event index advances more slowly
     than the ion count (ion count >= event index for the same point in
     the file, often noticeably higher once multi-hit events are
     common). This script uses the ion count (row index), i.e. it treats
     every individual ion as one depth step, regardless of how many
     ions shared its detection event.

WHY ION COUNT AS A DEPTH PROXY: ions are field-evaporated (removed) from
the specimen roughly in depth order, so how far along the recorded ion
sequence an ion sits is a reasonable, purely order-based stand-in for
"how deep into the original tip this ion came from" - see this
codebase's earlier periodicity scripts (e.g. the original
run_sequence_analysis.py, before the z-based variant) for the same idea
used for a different purpose (finding lattice periodicity along the
detection sequence instead of along reconstructed z).

WHAT COUNTS AS "TOTAL" FOR THE FRACTION: the elementwise fraction is
computed against the total number of RANGED ions in the window (i.e.
ions that matched SOME entry in your .rrng file) - unranged/background
ions (species code -1) are reported separately for transparency, but
excluded from the elementwise fractions, matching standard atom-probe
composition-reporting convention (composition is normally quoted among
ranged ions only).

Usage:
    python run_element_fraction_in_depth_window.py your_dataset.epos your_ranges.rrng
"""

import sys
import numpy as np

from custom_epos_io import read_custom_epos, verify_epos
from apt_io import read_rrng, assign_species_codes
from sequence_periodicity import (
   select_aperture, select_aperture_angular, angular_radius_from_mm, build_element_groups,
)

# ---- EDIT THESE ----
POLE_CENTER_X = -1.39   # your guessed pole center, detector x (mm)
POLE_CENTER_Y = 15.07  # your guessed pole center, detector y (mm)

# Aperture uses select_aperture_angular (true angular cone via
# point-projection, r = camera_length * tan(theta)) - same as the rest
# of this codebase. APERTURE_RADIUS is in mm on the detector; the code
# converts it to the equivalent angle internally using CAMERA_LENGTH.
CAMERA_LENGTH = 150     # instrument flight path length (mm) -
                        # Invizo 6000 default; EDIT if yours differs
APERTURE_RADIUS = 2   # aperture radius around the pole (mm)

# Percentage range (0-100) of the TOTAL SAMPLE DEPTH to include, based
# on the ION COUNT NUMBER (row index in the ePOS file) across the FULL
# dataset (see module docstring, point 2) - NOT the aperture's own ion
# count range, and NOT a detection-event index (which can be
# meaningfully lower than the ion count once multi-hit events occur).
SEQ_PERCENT_MIN = 12
SEQ_PERCENT_MAX = 24
# ---------------------


def main(epos_path, rrng_path):
    data = verify_epos(epos_path)
    ranges = read_rrng(rrng_path)

    codes, code_names = assign_species_codes(data['mn'], ranges)
    groups = build_element_groups(code_names)

    print(f"\nAvailable element groups (from your .rrng file):")
    for key, group_codes in groups.items():
        member_names = [code_names[c] for c in group_codes]
        print(f"  '{key}': combines {member_names}")

    # ---- Aperture selection - identical point-projection model used
    # throughout this codebase. ----
    angular_radius_deg = angular_radius_from_mm(APERTURE_RADIUS, CAMERA_LENGTH)
   # aperture_mask = select_aperture_angular(
    #    data['det_x'], data['det_y'], POLE_CENTER_X, POLE_CENTER_Y,
     #   angular_radius_deg, CAMERA_LENGTH)
    aperture_mask = select_aperture(
        data['det_x'], data['det_y'], POLE_CENTER_X, POLE_CENTER_Y,APERTURE_RADIUS)

    print(f"\nIons in aperture (before depth-window restriction): {int(aperture_mask.sum())}")

    # ---- Percentage-of-total-depth window, based on the ION COUNT
    # NUMBER (row index, 0 .. n-1) across the WHOLE dataset - see module
    # docstring, point 2. This is deliberately NOT tied to event_number
    # or pulse_number: it's simply "which ion, out of all ions in the
    # file, in acquisition order" - always one step per ion regardless
    # of multi-hit grouping. ----
    n_total_ions = len(data['mn'])
    ion_count = np.arange(n_total_ions)
    seq_min_total = ion_count.min()
    seq_max_total = ion_count.max()
    total_span = seq_max_total - seq_min_total
    seq_lo = seq_min_total + (SEQ_PERCENT_MIN / 100.0) * total_span
    seq_hi = seq_min_total + (SEQ_PERCENT_MAX / 100.0) * total_span
    print(f"\nTotal dataset ion count range: [{seq_min_total}, {seq_max_total}]")
    print(f"\npole center {POLE_CENTER_X}, {POLE_CENTER_Y}\n")
    print(f"Selected window ({SEQ_PERCENT_MIN:.1f}% - {SEQ_PERCENT_MAX:.1f}% of total depth): "
          f"ion count in [{seq_lo:.0f}, {seq_hi:.0f}]")

    depth_mask = (ion_count >= seq_lo) & (ion_count <= seq_hi)

    # ---- Combine aperture AND depth window ----
    combined_mask = aperture_mask & depth_mask
    n_in_window_all = int(combined_mask.sum())
    n_unranged = int((combined_mask & (codes == -1)).sum())
    n_ranged = n_in_window_all - n_unranged

    print(f"\n{'='*50}")
    print(f"Ions in aperture AND in the {SEQ_PERCENT_MIN:.1f}-{SEQ_PERCENT_MAX:.1f}% "
          f"depth window")
    print(f"{'='*50}")
    print(f"  Total (all, including unranged/noise): {n_in_window_all}")
    print(f"  Unranged/noise (no .rrng match):        {n_unranged}")
    print(f"  Ranged (used for elementwise fractions): {n_ranged}")

    # ---- Per-element counts and fractions (of the RANGED total - see
    # module docstring, "WHAT COUNTS AS TOTAL"). ----
    print(f"\n{'Element':10s} {'Count':>10s} {'Fraction':>10s}")
    print('-' * 32)
    element_counts = {}
    for element, group_codes in groups.items():
        species_mask = np.isin(codes, group_codes)
        n_element = int((species_mask & combined_mask).sum())
        fraction = n_element / n_ranged if n_ranged > 0 else 0.0
        element_counts[element] = {'count': n_element, 'fraction': fraction}
        print(f"{element:10s} {n_element:10d} {fraction:10.4f}")

    return {
        'n_in_window_all': n_in_window_all,
        'n_unranged': n_unranged,
        'n_ranged': n_ranged,
        'element_counts': element_counts,
    }


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run_element_fraction_in_depth_window.py your_dataset.epos your_ranges.rrng")
        sys.exit(1)
    main(sys.argv[1], '../ME_Folder/R6033_01112.RRNG')

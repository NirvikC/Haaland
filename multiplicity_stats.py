"""
Calculate the percentage of single-hit and multi-hit events within
a specified range of ION COUNT (i.e. ion index / row number in the ePOS
file), from an ePOS file.

Note on terminology, since this is easy to mix up:
  - "ion count" / "ion index" = the row number of an ion in the ePOS file
    (0-based). This is what EPOS_DTYPE stores one row per ion, so ion
    index N means "the (N+1)-th ion recorded".
  - "event number" (as used in the previous version of this script) =
    the index of a DETECTION EVENT (one pulse), where a single event can
    contain 1 ion (single hit) or several ions (multiple hit, i.e. several
    ions detected from the same pulse). Because multi-hit events bundle
    more than one ion under one event index, the event count is always
    <= the ion count for the same portion of the file, so a given ion
    index is always >= the event index of the event it belongs to.

This version takes the ION COUNT range as input (ION_MIN/ION_MAX, i.e. row
numbers in the ePOS file) instead of an event-index range, and reports the
single-hit / multi-hit percentages among the EVENTS that fall in that ion
range.

Edge handling: an event is included if ANY of its ions fall inside
[ION_MIN, ION_MAX]. If the requested ion range cuts through the middle of
a multi-hit event (e.g. ION_MAX lands on the first ion of a 3-ion event,
but not the other two), that whole event is still counted once - you
cannot have "half an event" - and a note is printed showing how many
events were only partially covered, plus how many ions that pulled in
beyond the exact requested range.
"""

import sys
import numpy as np

from fast_pairs import read_epos_mmap, _event_boundaries

# ---- EDIT INPUT VARIABLES HERE ----
EPOS_PATH = 'R6033_00293.EPOS'  # Path to your .epos file
ION_MIN = 15000000                 # Start ion count / index (inclusive, 0-based)
ION_MAX = 20000000                 # End ion count / index (inclusive), or None for all
# -----------------------------------


def main():
    print(f"Loading ePOS file (memory-mapped): {EPOS_PATH}")
    # Using read_epos_mmap from fast_pairs.py for efficient memory usage
    data, pulse_number = read_epos_mmap(EPOS_PATH)

    total_ions = len(data)
    print(f"Total ions in dataset: {total_ions:,}")

    # Get event boundaries and sizes using fast_pairs internal helper.
    # starts/ends are ION INDICES (ends exclusive) bounding each event;
    # sizes[i] = ends[i] - starts[i] = number of ions in that event.
    starts, ends, sizes = _event_boundaries(pulse_number)
    total_events = len(sizes)
    print(f"Total events in dataset: {total_events:,}")

    # Validate and clamp the requested ion-count range
    global ION_MIN, ION_MAX
    if ION_MIN is None or ION_MIN < 0:
        ION_MIN = 0
    if ION_MAX is None or ION_MAX >= total_ions:
        ION_MAX = total_ions - 1

    if ION_MIN > ION_MAX:
        print("!! Error: ION_MIN cannot be greater than ION_MAX.")
        sys.exit(1)

    print(f"Analyzing ion count range: [{ION_MIN:,} to {ION_MAX:,}]  "
          f"({ION_MAX - ION_MIN + 1:,} ion indices requested)")

    # Select every event that has at least one ion inside [ION_MIN, ION_MAX].
    # starts <= ION_MAX and ends > ION_MIN is the standard interval-overlap
    # test (ends is exclusive, ION_MAX is inclusive so compare as ends > ION_MIN
    # and starts <= ION_MAX).
    overlap = (starts <= ION_MAX) & (ends > ION_MIN)
    selected_sizes = sizes[overlap]
    selected_starts = starts[overlap]
    selected_ends = ends[overlap]
    num_events_in_range = len(selected_sizes)

    if num_events_in_range == 0:
        print("No events found in the specified ion count range.")
        return

    # How many events were only partially inside the requested range
    # (i.e. their ion span sticks out past ION_MIN or ION_MAX)?
    partial_mask = (selected_starts < ION_MIN) | (selected_ends - 1 > ION_MAX)
    num_partial = int(np.sum(partial_mask))

    # Actual ion index span actually covered by the selected events
    # (may be wider than [ION_MIN, ION_MAX] because of partial events
    # pulled in whole at the boundaries).
    actual_ion_start = int(selected_starts.min())
    actual_ion_end = int(selected_ends.max()) - 1
    actual_ion_count = actual_ion_end - actual_ion_start + 1

    # Count single hits (size == 1) vs multiple hits (size > 1)
    single_hits = int(np.sum(selected_sizes == 1))
    multiple_hits = int(np.sum(selected_sizes > 1))

    # Calculate percentages (of EVENTS, not ions)
    pct_single = (single_hits / num_events_in_range) * 100.0
    pct_multiple = (multiple_hits / num_events_in_range) * 100.0

    # Detailed breakdown of multiplicities if desired
    max_mult = int(np.max(selected_sizes))

    print("\n--- Results ---")
    print(f"Ion count range analyzed: [{ION_MIN:,} to {ION_MAX:,}]")
    if num_partial > 0:
        print(f"  (actual ion span covered, including {num_partial:,} boundary "
              f"events pulled in whole: [{actual_ion_start:,} to {actual_ion_end:,}], "
              f"{actual_ion_count:,} ions)")
    print(f"Events analyzed: {num_events_in_range:,}")
    print(f"  - Single-hit events (1 ion):    {single_hits:,}  ({pct_single:.2f}%)")
    print(f"  - Multi-hit events (>= 2 ions): {multiple_hits:,}  ({pct_multiple:.2f}%)")

    print("\nBreakdown by exact multiplicity count:")
    for k in range(1, min(max_mult + 1, 10)):
        count = int(np.sum(selected_sizes == k))
        if count > 0:
            pct = (count / num_events_in_range) * 100.0
            print(f"  - Multiplicity {k}: {count:,} events ({pct:.2f}%)")
    if max_mult >= 10:
        count = int(np.sum(selected_sizes >= 10))
        pct = (count / num_events_in_range) * 100.0
        print(f"  - Multiplicity >= 10: {count:,} events ({pct:.2f}%)")


if __name__ == '__main__':
    # Allow passing epos path via command line argument or use EPOS_PATH variable above
    path = sys.argv[1] if len(sys.argv) > 1 else EPOS_PATH
    EPOS_PATH = path
    main()

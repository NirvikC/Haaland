"""
Vectorized replacements for build_pair_counts / build_pair_stream, designed
for tens of millions of ions. Avoids any Python-level loop over individual
pulses by grouping ions of the same event-size together and using numpy
fancy-indexing to build all pairs for each group size in one shot.

Also includes a memory-mapped ePOS loader so the 1.3 GB file isn't fully
duplicated in RAM if you don't need every column at once.
"""

import numpy as np
from itertools import combinations
from collections import Counter

from apt_io import EPOS_DTYPE


def read_epos_mmap(path):
    """
    Memory-map the ePOS file instead of loading it fully into RAM.
    Good default for large files - numpy only pages in the parts you touch.
    Returns the memmap array (same dtype/behaviour as read_epos's first
    return value) plus the cumulative pulse_number array (this one IS
    materialised in RAM, but it's only 4 bytes/ion ~ 120 MB for 30M ions).
    """
    data = np.memmap(path, dtype=EPOS_DTYPE, mode='r')
    pulse_number = np.cumsum(data['delta_pulse'].astype(np.int64))
    return data, pulse_number


def _event_boundaries(pulse_number):
    """
    pulse_number is non-decreasing (ions are stored in acquisition order),
    so event boundaries can be found with a single diff instead of a groupby.
    Returns start indices, end indices (exclusive), and sizes for every event.
    """
    change = np.nonzero(np.diff(pulse_number))[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(pulse_number)]))
    sizes = ends - starts
    return starts, ends, sizes


def build_pair_counts_fast(pulse_number, codes, code_names, min_multiplicity=2, max_multiplicity=8):
    """
    Same output as build_pair_counts (a Counter of {(species_i, species_j): count}),
    but vectorized and memory-light throughout.

    Takes integer species `codes` directly (as produced by
    apt_io.assign_species_codes), NOT string labels - this avoids ever
    materialising a Python string object per ion, which is what actually
    caused memory blowups at ~20M-ion scale. `code_names` maps code -> name
    for building the final human-readable Counter keys at the very end
    (only len(code_names) strings are ever created).
    """
    starts, ends, sizes = _event_boundaries(pulse_number)
    codes = np.asarray(codes, dtype=np.int64)
    n_species = len(code_names)

    pair_counts = Counter()

    for k in range(min_multiplicity, max_multiplicity + 1):
        sel = np.nonzero(sizes == k)[0]
        if sel.size == 0:
            continue
        idx = starts[sel][:, None] + np.arange(k)[None, :]  # (n_events_k, k)
        codes_k = codes[idx]  # small ints, cheap compared to strings

        for a, b in combinations(range(k), 2):
            ca, cb = codes_k[:, a], codes_k[:, b]
            good = (ca >= 0) & (cb >= 0)  # exclude unranged (-1) ions
            ca, cb = ca[good], cb[good]
            lo = np.minimum(ca, cb)
            hi = np.maximum(ca, cb)
            keys = lo * n_species + hi  # single integer key per pair
            uniq_keys, counts = np.unique(keys, return_counts=True)
            lo_u = uniq_keys // n_species
            hi_u = uniq_keys % n_species
            for l, h, c in zip(lo_u, hi_u, counts):
                pair_counts[(code_names[l], code_names[h])] += int(c)

    return pair_counts


def build_pair_stream_fast(pulse_number, mn, tof, min_multiplicity=2, max_multiplicity=8,
                            voltage=None):
    """
    Vectorized equivalent of build_pair_stream. Returns concatenated
    mn_i, mn_j, tof_i, tof_j arrays (both orderings of every pair),
    built without any per-event Python loop.

    If `voltage` is provided (e.g. data['standing_v'], or standing_v +
    pulse_v combined beforehand depending on your pulsing mode), also
    returns v_i, v_j - the per-ion accelerating voltage (V0) recorded for
    each ion in the pair. Since both ions in a pair share the same pulse,
    v_i and v_j will normally be identical (same event) - this is what
    lets you convert a fitted Vd/V0 fraction into an actual Vd in volts
    for that specific event, rather than only a dimensionless fraction.
    """
    starts, ends, sizes = _event_boundaries(pulse_number)
    mn = np.asarray(mn)
    tof = np.asarray(tof)
    if voltage is not None:
        voltage = np.asarray(voltage)

    mn_i_parts, mn_j_parts, tof_i_parts, tof_j_parts = [], [], [], []
    v_i_parts, v_j_parts = [], []

    for k in range(min_multiplicity, max_multiplicity + 1):
        sel = np.nonzero(sizes == k)[0]
        if sel.size == 0:
            continue
        idx = starts[sel][:, None] + np.arange(k)[None, :]  # (n_events_k, k)
        mn_k = mn[idx]
        tof_k = tof[idx]
        if voltage is not None:
            v_k = voltage[idx]

        for a, b in combinations(range(k), 2):
            # both orderings, matching build_pair_stream's use of permutations
            mn_i_parts.append(mn_k[:, a]); mn_j_parts.append(mn_k[:, b])
            tof_i_parts.append(tof_k[:, a]); tof_j_parts.append(tof_k[:, b])
            mn_i_parts.append(mn_k[:, b]); mn_j_parts.append(mn_k[:, a])
            tof_i_parts.append(tof_k[:, b]); tof_j_parts.append(tof_k[:, a])
            if voltage is not None:
                v_i_parts.append(v_k[:, a]); v_j_parts.append(v_k[:, b])
                v_i_parts.append(v_k[:, b]); v_j_parts.append(v_k[:, a])

    mn_i_out = np.concatenate(mn_i_parts)
    mn_j_out = np.concatenate(mn_j_parts)
    tof_i_out = np.concatenate(tof_i_parts)
    tof_j_out = np.concatenate(tof_j_parts)

    if voltage is not None:
        v_i_out = np.concatenate(v_i_parts)
        v_j_out = np.concatenate(v_j_parts)
        return mn_i_out, mn_j_out, tof_i_out, tof_j_out, v_i_out, v_j_out

    return mn_i_out, mn_j_out, tof_i_out, tof_j_out

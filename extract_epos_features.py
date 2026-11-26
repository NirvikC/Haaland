"""
Window-by-window feature extraction from ePOS + RRNG files, for the
"undercounted oxygen" regression model. Uses the existing modules
(apt_io, fast_pairs, correlation_table, correlation_histogram) unchanged.

For every window of ions (default: 1M consecutive ions) it produces ONE
row of features, written to a CSV (one CSV for all runs/windows):

  A. Correlation-table features (multi-hit pairs)
       O_ij__<A>__<B>   observed pair count of species A with species B
       Z_ij__<A>__<B>   Saxey deviation score for that pair (off-diagonal
                        only; the diagonal Z is ~0 by construction after
                        the diagonal-neutralisation step)
       E_ij__<A>__<B>   expected count under independent evaporation
     for every unordered pair among TARGET_SPECIES (O+, O2+, SiO+, Si+ by
     default). The table is built from ALL ranged species in the window
     (so E and Z are computed correctly) and only the target cells are
     kept. Pairs that never occur get O=E=Z=0.
       n_pairs_total    total number of ion pairs in the window's table

  B. 1D-spectrum & dissociation features (every ion / pair stream)
       n_ions, n_unranged, frac_unranged
       bg_density_per_Da   unranged-ion density (counts/Da) measured only
                           in mass regions that are not inside any
                           .rrng range = background level
       For each peak in PEAKS (e.g. O+, O2+, Si+, SiO+):
         peak_<p>          ions in [m-hw, m+hw)
         tail_hi_<p>       ions in the window just ABOVE the peak
         tail_lo_<p>       ions in the window just BELOW the peak
         tail_hi_excess_<p> tail_hi minus background expectation (>=0)
         tail_hi_over_peak_<p>, tail_asym_<p>  =(hi-lo)/peak
       For each pathway in NEUTRAL_LOSS_PATHWAYS (e.g. SiO2+ -> Si+ + O2):
         band_lo/band_hi   [m1'.min, m1'.max] from dissociation_track
         n_in_band, n_bg_expected, n_excess   UNRANGED ions in the band, the
                           fitted continuum there, and the visible excess
         A_best, A_low, A_high   channel amplitude (TOTAL ions that went
                           through the pathway, incl. the part hidden under
                           ranged peaks) from the unranged-ions /
                           per-peak-tail / template fit in
                           correlation_histogram.score_single_ion_neutral_loss
                           (A_low..A_high = profile-likelihood interval over
                           the kept baseline variants)
         n_true_events, rate_ppm (+ _low / _high)   A / efficiency and ppm
         visible_fraction  part of the template on unranged mass axis
         chi2_red, chi2_red_null, delta_dev   fit quality with / without the
                           template and the deviance improvement it brings
         excess_over_peak  A_best / ions in the m1 peak
         unr_*             simple density-ratio features that count only
                           unranged ions in the band (no model)
       For each pathway in DISSOCIATION_2D_PATHWAYS (two charged fragments):
         d2d_<label>__*    parallel columns to nl_<label>__* using 2D pair
                           track scoring, shuffled-pair background
                           subtraction, and efficiency^2 event scaling.

IMPORTANT: the neutral-loss tail and ordinary peak tails look alike in a 1D
spectrum, so A_best is poorly determined when A_low = 0. For regression use
A_high (upper limit) and delta_dev / chi2_red rather than A_best alone, and
check chi2_red: a value well above 1 means an unranged peak or bump is
missing from the model (see NL_EXTRA_PEAK_MASSES / NL_EXTRA_MASK_INTERVALS).
The fit is ~10-20 s per pathway per window; set NL_PROFILE_CI = False to skip
the slowest part (A_low / A_high are then just the spread over variants).

  C. Species (charge-state) ratios - from SPECIES_RATIOS in the config
       <label>            numerator counts / denominator counts (NaN if the
                          denominator is 0)
       <label>__frac      num / (num + den), bounded 0..1 and better
                          behaved than the raw ratio when counts are small
     WARNING (label leakage): if your training label is the O deficit
     2*N_Si - N_O, do NOT add a ratio that rebuilds O/Si (e.g. all-O over
     all-Si). That would hand the model the answer. Ratios between charge
     states / molecular forms of the SAME element (O+/O(2+), Si+/Si(2+),
     SiO+/SiO2+) are the intended use.

  D. Field and pulsing proxies (from standing_v and pulse_v)
       standing_v_{mean,std,min,max,drift}, pulse_v_{...}
       drift = fitted change across the window (V), i.e. how fast the
       field is rising; pulse_fraction = mean(pulse_v)/mean(standing_v)
       (NaN if standing_v is 0). Some exports store 0/constant voltages -
       a warning is printed per run if so, and those columns are useless.

  E. Detection rate (from delta_pulse)
       pulses_elapsed     pulses covered by the window
       ions_per_pulse     n_ions / pulses_elapsed   (x100 = detection rate %)
       events_per_pulse   detection events / pulses_elapsed
       mean_pulses_between_events, median_..., std_...  (event-level gaps)

  G. TARGET (columns starting with 'target_' - never use them as inputs)
       target_N_O, target_N_Si   O / Si ATOM counts from the ranged ions of
                                 the window (O2+ = 2 O, SiO+ = 1 Si + 1 O,
                                 SiO2+ = 1 Si + 2 O ... parsed from the
                                 .rrng species names)
       target_O_over_Si          N_O / N_Si
       target_d                  1 - (N_O/N_Si)/2 : relative O deficit. For an
                                 ideal SiO2 sample it equals the fraction of
                                 O lost to undercounting (window-size
                                 independent). Negative = O excess.
       target_d_bgcorr           same, after subtracting the local background
                                 under each range (BG_FLANK_DA, see config)
       target_d_stderr           Poisson error of target_d (use 1/stderr^2 as
                                 sample weight when window sizes differ)
       target_N_O_bgcorr, target_N_Si_bgcorr
     Assumption: the true composition is SiO2 (O/Si = 2). target_d then
     measures the net O undercount (Si loss, if any, makes it a lower bound
     of the O loss).

  H. Window-size independent versions of the count features
       <col>__per_pair   O_ij / E_ij divided by n_pairs_total
       <col>__per_ion    peak_*, tail_*, band counts, unranged counts ...
                         divided by n_ions
       O_over_E__<A>__<B>  O_ij / E_ij (NaN if E_ij = 0)
     With DROP_RAW_COUNTS = True the raw (window-size dependent) count
     columns are left out of the CSV; n_ions and n_pairs_total are always
     kept (use them for weights). Z_ij grows with sqrt(N) - prefer O_over_E.

  F. Metadata you supply (laser energy, temperature, ...) - see
     METADATA_COLUMNS / RUN_METADATA / WINDOW_METADATA in the config.

Event alignment: an event is never split. Each detection event belongs to
the window that contains its FIRST ion, so consecutive windows tile the
file exactly (no event is counted twice or dropped) and the actual ion
span can differ from the requested one by a few ions; it is reported as
ion_lo/ion_hi.
"""

import csv
import re
from itertools import combinations_with_replacement

import numpy as np

from apt_io import read_rrng, assign_species_codes
from fast_pairs import (
    read_epos_mmap,
    _event_boundaries,
    build_pair_counts_fast,
    build_pair_stream_fast,
)
from correlation_table import counts_to_matrix, deviation_matrix
from correlation_histogram import (
    score_single_ion_neutral_loss,
    single_ion_band_bounds,
    score_dissociation_candidate,
    background_score,
    estimate_dissociation_occurrence,
)

# ---- EDIT THESE ----
RUNS = [
    # (run_id, epos_path, rrng_path)
    ('run1', 'R6033_00918.EPOS', 'R6033_00918.RRNG'),
]
OUT_CSV = 'window_features.csv'

WINDOW_SIZE_IONS = 1_000_000     # ions per window
WINDOW_STEP_IONS = None          # None = non-overlapping (step = size)
KEEP_PARTIAL_LAST_WINDOW = False  # a short last window has different statistics

# Explicit (start_ion, end_ion) windows per run (0-based ion indices;
# start_ion is inclusive, end_ion is exclusive or None for end of file).
# Windows do not have to be contiguous or sorted; window_idx in the CSV /
# WINDOW_METADATA is the position in this list.
EXPLICIT_WINDOWS = {
    'run1': [
        (0, 1_500_000),
        (1_500_000, 4_000_000),
        (6_000_000, None),
    ],
}

# Legacy start-only windows (used if run_id is not in EXPLICIT_WINDOWS):
# Window k covers ions WINDOW_STARTS[run_id][k] ... WINDOW_STARTS[run_id][k+1] - 1.
# The last window runs to the end of the file unless LAST_WINDOW_END[run_id]
# (an ion count, exclusive) is given.
WINDOW_STARTS = {
    # 'run1': [0, 1_500_000, 4_000_000, 9_000_000],
}
LAST_WINDOW_END = {
    # 'run1': 12_000_000,
}

# Names must match code_names from your .rrng (read_rrng builds e.g.
# 'O(1+)', 'O2(1+)', 'SiO(1+)', 'Si(1+)' when there is no Name: tag).
TARGET_SPECIES = [
    'O(1+)', 'O2(1+)', 'SiO(1+)', 'Si(1+)', 'Si(2+)', 'Si(3+)',
    'SiO(2+)', 'SiO2(1+)', 'SiO2(2+)',
]

# (mp, m1, m2, label) - same convention as run_analysis.py's
# neutral_loss_candidates: parent m/n, charged fragment m/n, neutral mass.
NEUTRAL_LOSS_PATHWAYS = [
    (60.0, 28.0, 32.0, 'path1'),
    (30.0, 14.0, 32.0, 'path2'),
    (60.0, 44.0, 16.0, 'path3'),
]

# 2D two-charged-ion dissociation pathways: (mp, m1, m2, label)
DISSOCIATION_2D_PATHWAYS = [
      (30.0, 44.0, 16.0, 'reaction1'),
        (22.0, 28.0, 16.0, 'reaction2'),
        (30.0, 28.0, 16.0, 'reaction3'),
        #(30.0, 28.0, 32.0, 'reaction4'),
        (22.5, 44.0, 1.0,  'reaction5'),
        (23.0, 44.0, 2.0,  'reaction6'),
        (20.0, 14.0, 16.0, 'reaction7'),
        (20.0, 28.0, 16.0, 'reaction8'),
]
D2D_TOLERANCE = 0.5
D2D_EXCLUDE_RADIUS = 1.0
D2D_N_SHUFFLES = 5
D2D_SEED = 0

EXCLUDE_PEAK_HALFWIDTH = 0.4   # core of the m1 peak (for the peak-ratio columns)
DETECTOR_EFFICIENCY = 0.62

# 1D neutral-loss fit (correlation_histogram.score_single_ion_neutral_loss).
# Same meaning as in run_analysis.py.
NL_EXCLUDE_PEAK_MASSES = (14.0, 16.0, 18.0, 28.0, 30.0, 32.0, 44.0, 60.0, 29.0, 45.0, 61.0)
NL_EXCLUDE_HALFWIDTH = 0.6     # removed around each of those masses (they also
                               # become tail sources in the fit)
NL_EXTRA_PEAK_MASSES = (17.0,)  # UNRANGED peaks whose tails must be modelled (ranged ones are automatic)
NL_EXTRA_MASK_INTERVALS = ((25.5, 26.5),)   # unranged bump at ~26 Da to drop from the fit
NL_RANGE_MARGIN = 0.1          # widen every .rrng range by this much in the fit
NL_PROFILE_CI = True           # profile-likelihood A_low / A_high (slower)

# Peaks for the tail/peak-ratio features: (label, centre m/n). The peak
# window is centre +/- PEAK_HALFWIDTH (match your .rrng range half-width);
# tails are TAIL_WIDTH wide, directly above and below it. Make sure a
# tail window does not run into a neighbouring peak (e.g. Si 28/29/30 are
# ~1 Da apart: keep PEAK_HALFWIDTH + TAIL_WIDTH < ~0.6).
PEAKS = [
    ('Op', 16.0), ('Si2p', 28.0), ('Si1p', 14.0),
    ('O2p', 32.0), ('SiOp', 44.0), ('SiO2p', 60.0),
]
PEAK_HALFWIDTH = 0.4
TAIL_WIDTH = 0.25

SPECTRUM_RANGE = (0.0, 90.0)   # m/n range used for background estimation
SPECTRUM_BIN = 0.05

# Species ratios: (label, [numerator species], [denominator species]).
# Species names must match the .rrng-derived names (same rule as
# TARGET_SPECIES); a list means "sum of these species". Note the
# difference between O(2+) (doubly charged atom, ~8 Da) and O2(1+)
# (molecular ion, ~32 Da). Edit freely - add/remove tuples.
SPECIES_RATIOS = [
    ('Si1_over_Si2',      ['Si(1+)'],  ['Si(2+)']),
    ('Si2_over_Si3',      ['Si(2+)'],  ['Si(3+)']),
    ('SiO1_over_SiO2plus', ['SiO(1+)'], ['SiO(2+)']),
    ('SiO2_over_SiO2plus', ['SiO2(1+)'], ['SiO2(2+)']),
]

# Target / normalisation
BG_FLANK_DA = 2.0         # width of the free (unranged) flank on each side of a
                          # range used for its local background density
DROP_RAW_COUNTS = True   # True: omit raw count columns (keep per-ion / per-pair ones)

# Metadata you know for each run/window (not stored in the ePOS). Every
# name in METADATA_COLUMNS becomes a CSV column. Values come from
# RUN_METADATA[run_id], optionally overridden for a specific window by
# WINDOW_METADATA[(run_id, window_idx)]. Missing values become NaN.
METADATA_COLUMNS = ['laser_energy_pJ', 'temperature_K']
RUN_METADATA = {
    'run1': {'laser_energy_pJ': 50, 'temperature_K': 50},
}
WINDOW_METADATA = {
    ('run1', 1): {'laser_energy_pJ': 65},   # optional per-window override
}
# ---------------------


def event_aligned_bounds(starts, ends, ion_min, ion_max):
    """Align the inclusive ion range [ion_min, ion_max] to whole events:
    an event belongs to the window that contains its FIRST ion, so
    adjacent windows never share or split an event.
    Returns (ion_lo, ion_hi, ev_lo, ev_hi): the half-open ion slice
    [ion_lo, ion_hi) and the first/last event index (inclusive).
    Raises ValueError if no event starts inside the range."""
    ev_lo = int(np.searchsorted(starts, ion_min, side='left'))
    ev_hi = int(np.searchsorted(starts, ion_max, side='right') - 1)
    if ev_lo >= len(starts) or ev_hi < ev_lo:
        raise ValueError(f"no detection event starts inside ions "
                         f"[{ion_min}, {ion_max}]")
    return int(starts[ev_lo]), int(ends[ev_hi]), ev_lo, ev_hi


def correlation_features(pulse_w, codes_w, code_names, target_species):
    """O_ij, Z_ij, E_ij for every unordered pair of target_species."""
    pair_counts = build_pair_counts_fast(pulse_w, codes_w, code_names,
                                         min_multiplicity=2)
    feats = {'n_pairs_total': float(sum(pair_counts.values()))}
    if pair_counts:
        O, order = counts_to_matrix(pair_counts)
        Z, E = deviation_matrix(O)
        idx = {s: i for i, s in enumerate(order)}
    else:
        O = Z = E = None
        idx = {}

    for a, b in combinations_with_replacement(target_species, 2):
        i, j = idx.get(a), idx.get(b)
        if i is None or j is None:
            o = z = e = 0.0
        else:
            o, z, e = float(O[i, j]), float(Z[i, j]), float(E[i, j])
        npt = feats['n_pairs_total']
        feats[f'O_ij__{a}__{b}'] = o
        feats[f'E_ij__{a}__{b}'] = e
        feats[f'O_ij__{a}__{b}__per_pair'] = o / npt if npt > 0 else np.nan
        feats[f'E_ij__{a}__{b}__per_pair'] = e / npt if npt > 0 else np.nan
        feats[f'O_over_E__{a}__{b}'] = o / e if e > 0 else np.nan
        if a != b:   # diagonal Z is ~0 by construction (neutralised)
            feats[f'Z_ij__{a}__{b}'] = z
    return feats


def _count(sorted_mn, lo, hi):
    """Number of values in [lo, hi) from a sorted array."""
    return int(np.searchsorted(sorted_mn, hi, side='left')
               - np.searchsorted(sorted_mn, lo, side='left'))


def _in_any_range(x, ranges):
    mask = np.zeros(np.shape(x), dtype=bool)
    for r in ranges:
        mask |= (x >= r['lo']) & (x <= r['hi'])
    return mask


def dissociation_2d_features(pulse_w, mn_w, tof_w, mn_sorted, n_ions, n_pairs_total=np.nan):
    """2D two-charged-ion dissociation track features using column names
    parallel to the 1D neutral-loss features (prefixed with `d2d_<label>__`)."""
    feats = {}
    if not DISSOCIATION_2D_PATHWAYS:
        return feats

    mn_i, mn_j, _, _ = build_pair_stream_fast(pulse_w, mn_w, tof_w)
    total_evap = n_ions / DETECTOR_EFFICIENCY if DETECTOR_EFFICIENCY else np.nan
    eff2 = DETECTOR_EFFICIENCY ** 2 if DETECTOR_EFFICIENCY else np.nan

    for mp, m1, m2, label in DISSOCIATION_2D_PATHWAYS:
        b_lo, b_hi, _ = single_ion_band_bounds(mp, m1, m2)
        if len(mn_i) > 0:
            n_raw = float(score_dissociation_candidate(
                mp, m1, m2, mn_i, mn_j,
                tolerance=D2D_TOLERANCE,
                exclude_radius=D2D_EXCLUDE_RADIUS,
            ))
            bg_mean, bg_std = background_score(
                mp, m1, m2, mn_i, mn_j,
                tolerance=D2D_TOLERANCE,
                exclude_radius=D2D_EXCLUDE_RADIUS,
                n_shuffles=D2D_N_SHUFFLES,
                seed=D2D_SEED,
            )
            n_net = max(0.0, n_raw - bg_mean)
            n_low_obs = max(0.0, n_raw - (bg_mean + bg_std))
            n_high_obs = max(0.0, n_raw - max(0.0, bg_mean - bg_std))
            est = estimate_dissociation_occurrence(n_net, DETECTOR_EFFICIENCY, n_ions)
            n_true = float(est['n_true_dissociation_events'])
            n_true_lo = float(n_low_obs / eff2) if eff2 else np.nan
            n_true_hi = float(n_high_obs / eff2) if eff2 else np.nan
        else:
            n_raw = bg_mean = bg_std = n_net = n_low_obs = n_high_obs = 0.0
            n_true = n_true_lo = n_true_hi = 0.0

        rate = lambda n: float(1e6 * n / total_evap) if total_evap else np.nan
        n_peak_m1 = _count(mn_sorted, m1 - EXCLUDE_PEAK_HALFWIDTH,
                           m1 + EXCLUDE_PEAK_HALFWIDTH)

        p = f'd2d_{label}'
        feats[f'{p}__band_lo'] = b_lo
        feats[f'{p}__band_hi'] = b_hi
        feats[f'{p}__n_in_band'] = n_raw
        feats[f'{p}__n_bg_expected'] = float(bg_mean)
        feats[f'{p}__n_excess'] = float(n_net)
        feats[f'{p}__A_best'] = float(n_net)
        feats[f'{p}__A_low'] = float(n_low_obs)
        feats[f'{p}__A_high'] = float(n_high_obs)
        feats[f'{p}__n_true_events'] = n_true
        feats[f'{p}__n_true_events_low'] = n_true_lo
        feats[f'{p}__n_true_events_high'] = n_true_hi
        feats[f'{p}__rate_ppm'] = rate(n_true)
        feats[f'{p}__rate_ppm_low'] = rate(n_true_lo)
        feats[f'{p}__rate_ppm_high'] = rate(n_true_hi)
        feats[f'{p}__visible_fraction'] = np.nan
        feats[f'{p}__chi2_red'] = np.nan
        feats[f'{p}__chi2_red_null'] = np.nan
        feats[f'{p}__delta_dev'] = np.nan
        feats[f'{p}__n_variants'] = float(D2D_N_SHUFFLES)
        feats[f'{p}__excess_over_peak'] = n_net / n_peak_m1 if n_peak_m1 > 0 else np.nan
        for nm in ('n_in_band', 'n_bg_expected', 'n_excess'):
            feats[f'{p}__{nm}__per_ion'] = feats[f'{p}__{nm}'] / n_ions if n_ions else np.nan
            feats[f'{p}__{nm}__per_pair'] = (
                feats[f'{p}__{nm}'] / n_pairs_total
                if np.isfinite(n_pairs_total) and n_pairs_total > 0 else np.nan
            )

    return feats


def spectrum_features(mn, codes, ranges):
    """1D-spectrum background, peak-tail and neutral-loss-band features."""
    mn = np.asarray(mn, dtype=np.float64)
    mn_sorted = np.sort(mn)
    n_ions = len(mn)
    unranged = (codes == -1)
    feats = {
        'n_ions': float(n_ions),
        'n_unranged': float(unranged.sum()),
        'frac_unranged': float(unranged.mean()) if n_ions else np.nan,
    }

    # ---- background density, measured only outside every .rrng range ----
    lo_r, hi_r = SPECTRUM_RANGE
    edges = np.arange(lo_r, hi_r + SPECTRUM_BIN, SPECTRUM_BIN)
    counts, _ = np.histogram(mn, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    free_bin = ~_in_any_range(centres, ranges)
    n_free = int(free_bin.sum())
    bg_density = (counts[free_bin].sum() / (n_free * SPECTRUM_BIN)
                  if n_free else np.nan)
    feats['bg_density_per_Da'] = float(bg_density)

    # ---- peak / tail features ----
    for label, m in PEAKS:
        peak = _count(mn_sorted, m - PEAK_HALFWIDTH, m + PEAK_HALFWIDTH)
        hi = _count(mn_sorted, m + PEAK_HALFWIDTH, m + PEAK_HALFWIDTH + TAIL_WIDTH)
        lo = _count(mn_sorted, m - PEAK_HALFWIDTH - TAIL_WIDTH, m - PEAK_HALFWIDTH)
        bg_exp = bg_density * TAIL_WIDTH if np.isfinite(bg_density) else 0.0
        feats[f'peak_{label}'] = float(peak)
        feats[f'tail_hi_{label}'] = float(hi)
        feats[f'tail_lo_{label}'] = float(lo)
        feats[f'tail_hi_excess_{label}'] = float(max(0.0, hi - bg_exp))
        for nm, val in (('peak', peak), ('tail_hi', hi), ('tail_lo', lo),
                        ('tail_hi_excess', max(0.0, hi - bg_exp))):
            feats[f'{nm}_{label}__per_ion'] = val / n_ions if n_ions else np.nan
        feats[f'tail_hi_over_peak_{label}'] = hi / peak if peak > 0 else np.nan
        feats[f'tail_asym_{label}'] = (hi - lo) / peak if peak > 0 else np.nan

    # ---- neutral-loss band features ----
    grid_n = 20000
    for mp, m1, m2, label in NEUTRAL_LOSS_PATHWAYS:
        n_true = n_true_lo = n_true_hi = np.nan
        try:
            st = score_single_ion_neutral_loss(
                mp, m1, m2, mn,
                exclude_peak_mass=m1,
                exclude_peak_halfwidth=NL_EXCLUDE_HALFWIDTH,
                exclude_peak_masses=NL_EXCLUDE_PEAK_MASSES,
                extra_peak_masses=NL_EXTRA_PEAK_MASSES,
                extra_mask_intervals=NL_EXTRA_MASK_INTERVALS,
                ranges=ranges, range_margin=NL_RANGE_MARGIN,
                profile_ci=NL_PROFILE_CI)
        except ValueError as err:     # no usable fit window (band too close to 0 ...)
            print(f"   !! neutral-loss fit failed for {label}: {err}")
            b_lo, b_hi, _ = single_ion_band_bounds(mp, m1, m2)
            st = None
        total_evap = n_ions / DETECTOR_EFFICIENCY
        if st is not None:
            n_true = st['A_best'] / DETECTOR_EFFICIENCY
            n_true_lo = st['A_low'] / DETECTOR_EFFICIENCY
            n_true_hi = st['A_high'] / DETECTOR_EFFICIENCY
        else:
            st = {'band_lo': b_lo, 'band_hi': b_hi, 'n_in_band': np.nan,
                  'n_bg_expected': np.nan, 'n_excess': np.nan, 'A_best': np.nan,
                  'A_low': np.nan, 'A_high': np.nan, 'visible_fraction': np.nan,
                  'baseline_chi2_red_median': np.nan, 'chi2_red_null_median': np.nan,
                  'delta_dev_median': np.nan, 'n_variants': np.nan}
        rate = lambda n: float(1e6 * n / total_evap) if total_evap else np.nan
        n_peak_m1 = _count(mn_sorted, m1 - EXCLUDE_PEAK_HALFWIDTH,
                           m1 + EXCLUDE_PEAK_HALFWIDTH)
        p = f'nl_{label}'
        feats[f'{p}__band_lo'] = st['band_lo']
        feats[f'{p}__band_hi'] = st['band_hi']
        feats[f'{p}__n_in_band'] = float(st['n_in_band'])
        feats[f'{p}__n_bg_expected'] = float(st['n_bg_expected'])
        feats[f'{p}__n_excess'] = float(st['n_excess'])
        feats[f'{p}__A_best'] = float(st['A_best'])
        feats[f'{p}__A_low'] = float(st['A_low'])
        feats[f'{p}__A_high'] = float(st['A_high'])
        feats[f'{p}__n_true_events'] = float(n_true)
        feats[f'{p}__n_true_events_low'] = float(n_true_lo)
        feats[f'{p}__n_true_events_high'] = float(n_true_hi)
        feats[f'{p}__rate_ppm'] = rate(n_true)
        feats[f'{p}__rate_ppm_low'] = rate(n_true_lo)
        feats[f'{p}__rate_ppm_high'] = rate(n_true_hi)
        feats[f'{p}__visible_fraction'] = float(st['visible_fraction'])
        feats[f'{p}__chi2_red'] = float(st['baseline_chi2_red_median'])
        feats[f'{p}__chi2_red_null'] = float(st.get('chi2_red_null_median', np.nan))
        feats[f'{p}__delta_dev'] = float(st.get('delta_dev_median', np.nan))
        feats[f'{p}__n_variants'] = float(st['n_variants'])
        feats[f'{p}__excess_over_peak'] = st['A_best'] / n_peak_m1 if n_peak_m1 > 0 else np.nan
        for nm in ('n_in_band', 'n_bg_expected', 'n_excess'):
            feats[f'{p}__{nm}__per_ion'] = feats[f'{p}__{nm}'] / n_ions if n_ions else np.nan

        # Unranged-only version (real ranged peaks inside the band excluded)
        band_lo, band_hi = st['band_lo'], st['band_hi']
        in_band = (mn >= band_lo) & (mn <= band_hi)
        n_unr_in_band = int((in_band & unranged).sum())
        grid = np.linspace(band_lo, band_hi, grid_n)
        free_width = float((~_in_any_range(grid, ranges)).mean() * (band_hi - band_lo))
        outside = free_bin & ~((centres >= band_lo) & (centres <= band_hi))
        n_out = int(outside.sum())
        bg_out = (counts[outside].sum() / (n_out * SPECTRUM_BIN)
                  if n_out else bg_density)
        n_unr_expected = bg_out * free_width if np.isfinite(bg_out) else 0.0
        feats[f'{p}__unr_n_in_band'] = float(n_unr_in_band)
        feats[f'{p}__unr_free_width_Da'] = free_width
        feats[f'{p}__unr_bg_density_outside'] = float(bg_out)
        feats[f'{p}__unr_n_excess'] = float(max(0.0, n_unr_in_band - n_unr_expected))
        feats[f'{p}__unr_density_ratio'] = (
            (n_unr_in_band / free_width) / bg_out
            if free_width > 0 and bg_out and np.isfinite(bg_out) and bg_out > 0
            else np.nan)
        feats[f'{p}__unr_excess_over_peak'] = (
            feats[f'{p}__unr_n_excess'] / n_peak_m1 if n_peak_m1 > 0 else np.nan)
        feats[f'{p}__unr_n_in_band__per_ion'] = n_unr_in_band / n_ions if n_ions else np.nan
        feats[f'{p}__unr_n_excess__per_ion'] = feats[f'{p}__unr_n_excess'] / n_ions if n_ions else np.nan
    return feats


_ELEM = re.compile(r'([A-Z][a-z]?)(\d*)')


def atoms_in_species(name, element):
    """Number of `element` atoms in a species name such as 'O(1+)', 'O2(1+)',
    'SiO(1+)', 'SiO2(1+)', 'Si2+' (the charge part is stripped first)."""
    f = re.sub(r'\(\d*[+-]+\)\s*$', '', name.strip())
    f = re.sub(r'\d*[+-]+$', '', f) if f == name.strip() else f
    return sum(int(n) if n else 1 for el, n in _ELEM.findall(f) if el == element)


def composition_features(mn, codes, code_names, ranges):
    """O / Si atom counts, the relative O deficit d = 1 - (N_O/N_Si)/2 (the ML
    target) and its background-corrected / error versions."""
    mn = np.asarray(mn, dtype=np.float64)
    per_code = np.bincount(codes[codes >= 0], minlength=len(code_names)).astype(float)
    a_O = np.array([atoms_in_species(nm, 'O') for nm in code_names], dtype=float)
    a_Si = np.array([atoms_in_species(nm, 'Si') for nm in code_names], dtype=float)
    N_O, N_Si = float((per_code * a_O).sum()), float((per_code * a_Si).sum())

    # local background under every range from free flanks of width BG_FLANK_DA
    lo_r, hi_r = SPECTRUM_RANGE
    edges = np.arange(lo_r, hi_r + SPECTRUM_BIN, SPECTRUM_BIN)
    counts, _ = np.histogram(mn, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    free = ~_in_any_range(centres, ranges)
    glob = (counts[free].sum() / (free.sum() * SPECTRUM_BIN)) if free.any() else 0.0
    bg_code = np.zeros(len(code_names))
    for c, r in enumerate(ranges):
        fl = free & (((centres >= r['lo'] - BG_FLANK_DA) & (centres < r['lo'])) |
                     ((centres > r['hi']) & (centres <= r['hi'] + BG_FLANK_DA)))
        dens = counts[fl].sum() / (fl.sum() * SPECTRUM_BIN) if fl.any() else glob
        bg_code[c] = min(dens * (r['hi'] - r['lo']), per_code[c])
    N_O_bg = max(0.0, N_O - float((bg_code * a_O).sum()))
    N_Si_bg = max(0.0, N_Si - float((bg_code * a_Si).sum()))

    def d_of(no, nsi):
        return 1.0 - (no / nsi) / 2.0 if nsi > 0 else np.nan

    if N_Si > 0:   # Poisson error propagation of d = 1 - N_O / (2 N_Si)
        var = N_O / (4 * N_Si ** 2) + N_O ** 2 / (4 * N_Si ** 3)
        stderr = float(np.sqrt(var))
    else:
        stderr = np.nan
    return {
        'target_N_O': N_O, 'target_N_Si': N_Si,
        'target_O_over_Si': N_O / N_Si if N_Si > 0 else np.nan,
        'target_d': d_of(N_O, N_Si),
        'target_N_O_bgcorr': N_O_bg, 'target_N_Si_bgcorr': N_Si_bg,
        'target_d_bgcorr': d_of(N_O_bg, N_Si_bg),
        'target_d_stderr': stderr,
    }


_RAW_SUFFIXES = ('__n_in_band', '__n_bg_expected', '__n_excess', '__A_best', '__A_low',
                 '__A_high', '__n_true_events', '__n_true_events_low',
                 '__n_true_events_high', '__unr_n_in_band', '__unr_n_excess')
_RAW_PREFIXES = ('O_ij__', 'E_ij__', 'Z_ij__', 'peak_', 'tail_hi_excess_', 'tail_hi_',
                 'tail_lo_')


def is_raw_count(key):
    """Window-size dependent columns (dropped when DROP_RAW_COUNTS)."""
    if key.endswith(('__per_ion', '__per_pair')) or key.startswith('O_over_E__'):
        return False
    if key.startswith(('tail_hi_over_peak_', 'tail_asym_')):
        return False
    return (key.startswith(_RAW_PREFIXES) or key.endswith(_RAW_SUFFIXES)
            or key in ('n_unranged', 'pulses_elapsed'))


def species_ratio_features(codes_w, code_names):
    """Charge-state / molecular-form ratios from SPECIES_RATIOS."""
    per_code = np.bincount(codes_w[codes_w >= 0], minlength=len(code_names))
    by_name = {}
    for c, name in enumerate(code_names):      # several ranges can share a name
        by_name[name] = by_name.get(name, 0) + int(per_code[c])
    feats = {}
    for label, num_names, den_names in SPECIES_RATIOS:
        num = sum(by_name.get(s, 0) for s in num_names)
        den = sum(by_name.get(s, 0) for s in den_names)
        feats[label] = num / den if den > 0 else np.nan
        feats[f'{label}__frac'] = num / (num + den) if (num + den) > 0 else np.nan
    return feats


def _voltage_stats(v, prefix):
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    out = {f'{prefix}_mean': float(v.mean()), f'{prefix}_std': float(v.std()),
           f'{prefix}_min': float(v.min()), f'{prefix}_max': float(v.max())}
    if n > 1:
        x = np.arange(n) - (n - 1) / 2.0
        slope = float((x * v).sum() / (x * x).sum())   # V per ion
        out[f'{prefix}_drift'] = slope * (n - 1)       # V across the window
    else:
        out[f'{prefix}_drift'] = 0.0
    return out


def field_features(standing_v, pulse_v):
    """standing_v / pulse_v statistics and pulse fraction."""
    feats = _voltage_stats(standing_v, 'standing_v')
    feats.update(_voltage_stats(pulse_v, 'pulse_v'))
    sm = feats['standing_v_mean']
    feats['pulse_fraction'] = feats['pulse_v_mean'] / sm if sm > 0 else np.nan
    return feats


def rate_features(pulse_number, delta_pulse_events, ion_lo, ion_hi, n_events):
    """Detection-rate features. pulses_elapsed is the number of pulses
    covered by the window: the pulse counter at its last ion minus the
    counter just before its first ion."""
    prev = int(pulse_number[ion_lo - 1]) if ion_lo > 0 else 0
    pulses = int(pulse_number[ion_hi - 1]) - prev
    n_ions = ion_hi - ion_lo
    gaps = np.asarray(delta_pulse_events, dtype=np.float64)
    return {
        'pulses_elapsed': float(pulses),
        'ions_per_pulse': n_ions / pulses if pulses > 0 else np.nan,
        'events_per_pulse': n_events / pulses if pulses > 0 else np.nan,
        'mean_pulses_between_events': float(gaps.mean()) if len(gaps) else np.nan,
        'median_pulses_between_events': float(np.median(gaps)) if len(gaps) else np.nan,
        'std_pulses_between_events': float(gaps.std()) if len(gaps) else np.nan,
    }


def metadata_features(run_id, window_idx):
    md = dict(RUN_METADATA.get(run_id, {}))
    md.update(WINDOW_METADATA.get((run_id, window_idx), {}))
    return {c: float(md[c]) if c in md else np.nan for c in METADATA_COLUMNS}


def window_features(ion_min, ion_max, data, mn, pulse_number, codes, code_names,
                    ranges, starts, ends):
    """All ePOS-derived features for ONE window (inclusive ion range,
    event-aligned). `data` is the (memmapped) structured ePOS array."""
    ion_lo, ion_hi, ev_lo, ev_hi = event_aligned_bounds(starts, ends, ion_min, ion_max)
    sl = slice(ion_lo, ion_hi)
    feats = {'ion_min': ion_min, 'ion_max': ion_max,
             'ion_lo': ion_lo, 'ion_hi': ion_hi - 1}
    corr_feats = correlation_features(pulse_number[sl], codes[sl],
                                      code_names, TARGET_SPECIES)
    feats.update(corr_feats)
    feats.update(spectrum_features(mn[sl], codes[sl], ranges))
    feats.update(dissociation_2d_features(
        pulse_number[sl], mn[sl], data['tof'][sl],
        np.sort(np.asarray(mn[sl], dtype=np.float64)),
        len(mn[sl]),
        n_pairs_total=corr_feats.get('n_pairs_total', np.nan),
    ))
    feats.update(species_ratio_features(codes[sl], code_names))
    feats.update(composition_features(mn[sl], codes[sl], code_names, ranges))
    feats.update(field_features(data['standing_v'][sl], data['pulse_v'][sl]))
    ev_gaps = data['delta_pulse'][starts[ev_lo:ev_hi + 1]]
    feats.update(rate_features(pulse_number, ev_gaps, ion_lo, ion_hi,
                               ev_hi - ev_lo + 1))
    if DROP_RAW_COUNTS:
        feats = {k: v for k, v in feats.items() if not is_raw_count(k)}
    return feats


def window_list(n_ions, size, step, keep_partial):
    step = step or size
    wins = []
    for s in range(0, n_ions, step):
        e = min(s + size, n_ions)
        if e - s < size and not keep_partial:
            break
        wins.append((s, e - 1))
    return wins


def window_list_from_ranges(n_ions, window_ranges):
    """Windows from explicit (start, end) ion pairs (0-based; start
    inclusive, end exclusive or None for end of file). Returns inclusive
    (ion_min, ion_max) tuples."""
    wins = []
    for idx, pair in enumerate(window_ranges):
        if len(pair) != 2:
            raise ValueError(f"window {idx} must be a (start, end) pair")
        s_raw, e_raw = pair
        s = int(s_raw)
        e = n_ions if e_raw is None else min(int(e_raw), n_ions)
        if s < 0 or s >= n_ions:
            print(f"   !! window {idx} start {s:,} outside [0, {n_ions:,}); skipped.")
            continue
        if e <= s:
            raise ValueError(f"window {idx} end ({e}) must be greater than start ({s})")
        wins.append((s, e - 1))
    if not wins:
        raise ValueError("no valid explicit windows remained for this run")
    return wins


def window_list_from_starts(n_ions, window_starts, last_end=None):
    """Windows from explicit start ion counts: window k = [start_k,
    start_{k+1} - 1] (inclusive); the last runs to last_end (exclusive)
    or the end of the file."""
    s = [int(x) for x in window_starts]
    if not s or s[0] < 0 or any(b <= a for a, b in zip(s, s[1:])):
        raise ValueError("window starts must be non-negative and strictly increasing")
    kept = [x for x in s if x < n_ions]
    if len(kept) < len(s):
        print(f"   !! {len(s) - len(kept)} window start(s) >= file length "
              f"({n_ions:,} ions) were dropped.")
    end = n_ions if last_end is None else min(int(last_end), n_ions)
    if not kept or end <= kept[-1]:
        raise ValueError("last window end must be greater than the last start")
    bounds = kept + [end]
    return [(bounds[i], bounds[i + 1] - 1) for i in range(len(kept))]


def process_run(run_id, epos_path, rrng_path):
    print(f"[{run_id}] loading {epos_path}")
    data, pulse_number = read_epos_mmap(epos_path)
    ranges = read_rrng(rrng_path)
    mn = np.asarray(data['mn'], dtype=np.float32)
    codes, code_names = assign_species_codes(mn, ranges)
    print(f"[{run_id}] O / Si atoms per ranged species (check): " +
          ", ".join(f"{nm}: {atoms_in_species(nm, 'O')}O {atoms_in_species(nm, 'Si')}Si"
                    for nm in dict.fromkeys(code_names)))
    ambiguous = [nm for nm in code_names if '(' not in nm and re.search(r'\d[+-]+$', nm)]
    if ambiguous:
        print(f"[{run_id}] !! species names without '(charge)' are ambiguous for the "
              f"atom count (e.g. 'O2+' = O 2+ or O2 +): {sorted(set(ambiguous))}. "
              f"Rename them like 'O(2+)' / 'O2(1+)' in the .rrng.")
    missing = [s for s in TARGET_SPECIES if s not in code_names]
    if missing:
        print(f"[{run_id}] !! TARGET_SPECIES not found in rrng names {missing}; "
              f"available: {code_names}. Their pair features will be all 0.")
    ratio_names = {s for _, num, den in SPECIES_RATIOS for s in list(num) + list(den)}
    missing_r = sorted(s for s in ratio_names if s not in code_names)
    if missing_r:
        print(f"[{run_id}] !! SPECIES_RATIOS species not in rrng names {missing_r}; "
              f"ratios using them will be NaN (or based on the remaining species).")
    for col in ('standing_v', 'pulse_v'):
        sample = np.asarray(data[col][:min(len(data), 100000)])
        if len(np.unique(sample)) <= 1:
            print(f"[{run_id}] !! '{col}' looks constant/zero in this file - "
                  f"its field features carry no information.")
    if run_id not in RUN_METADATA and not any(k[0] == run_id for k in WINDOW_METADATA):
        print(f"[{run_id}] !! no metadata for this run ({METADATA_COLUMNS} will be NaN).")
    starts, ends, _ = _event_boundaries(pulse_number)
    if run_id in EXPLICIT_WINDOWS:
        wins = window_list_from_ranges(len(mn), EXPLICIT_WINDOWS[run_id])
    elif run_id in WINDOW_STARTS:
        wins = window_list_from_starts(len(mn), WINDOW_STARTS[run_id],
                                       LAST_WINDOW_END.get(run_id))
    else:
        wins = window_list(len(mn), WINDOW_SIZE_IONS, WINDOW_STEP_IONS,
                           KEEP_PARTIAL_LAST_WINDOW)
    print(f"[{run_id}] {len(mn):,} ions -> {len(wins)} windows")
    rows = []
    for k, (a, b) in enumerate(wins):
        try:
            f = window_features(a, b, data, mn, pulse_number, codes, code_names,
                                ranges, starts, ends)
        except ValueError as err:
            print(f"[{run_id}] window {k} skipped: {err}")
            continue
        f = {'run_id': run_id, 'window_idx': k, **f,
             **metadata_features(run_id, k)}
        rows.append(f)
        print(f"[{run_id}] window {k + 1}/{len(wins)} done")
    return rows


def main():
    all_rows = []
    for run_id, epos_path, rrng_path in RUNS:
        all_rows.extend(process_run(run_id, epos_path, rrng_path))
    if not all_rows:
        print("No windows produced (file shorter than WINDOW_SIZE_IONS?).")
        return
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"Saved {OUT_CSV}  ({len(all_rows)} rows, {len(all_rows[0])} columns)")


if __name__ == '__main__':
    main()

"""
Correlation histograms ('Saxey plots'), molecular-ion-dissociation
track overlays, and 1D single-ion + neutral-loss spectrum analysis.
"""

import numpy as np
import pandas as pd
from itertools import permutations
import matplotlib.pyplot as plt

def build_pair_stream(pulse_number, mn, tof, min_multiplicity=2, max_multiplicity=8):
    """
    For every multi-hit event, emit BOTH orderings of every ion pair
    (i,j) and (j,i), so the resulting 2D histogram is diagonally
    symmetric, matching the paper's convention.

    Returns arrays: mn_i, mn_j, tof_i, tof_j  (one row per ordered pair)
    """
    df = pd.DataFrame({'pulse': pulse_number, 'mn': mn, 'tof': tof})

    mn_i_list, mn_j_list, tof_i_list, tof_j_list = [], [], [], []

    for _, grp in df.groupby('pulse'):
        n = len(grp)
        if n < min_multiplicity or n > max_multiplicity:
            continue
        mn_vals = grp['mn'].to_numpy()
        tof_vals = grp['tof'].to_numpy()
        for a, b in permutations(range(n), 2):
            mn_i_list.append(mn_vals[a])
            mn_j_list.append(mn_vals[b])
            tof_i_list.append(tof_vals[a])
            tof_j_list.append(tof_vals[b])

    return (np.array(mn_i_list), np.array(mn_j_list),
            np.array(tof_i_list), np.array(tof_j_list))


def mn_correlation_histogram(mn_i, mn_j, m_range=(0, 100), bin_width=0.02):
    """
    Direct reproduction of Figure 3/4/6-style m/n vs m/n correlation
    histograms (curved trails, Section 2.2).
    """
    bins = np.arange(m_range[0], m_range[1] + bin_width, bin_width)
    H, xedges, yedges = np.histogram2d(mn_i, mn_j, bins=[bins, bins])
    return H, xedges, yedges


def tof_correlation_histogram(tof_i, tof_j, n_bins=500):
    """
    Section 3.2's recommended representation: t2 vs (t2 - t1). This
    straightens the delayed-coincidence trails into vertical lines,
    making them easy to range/select. Here 'i' is treated as ion 1
    and 'j' as ion 2 for each ordered pair.
    """
    delta_t = tof_j - tof_i
    H, xedges, yedges = np.histogram2d(delta_t, tof_j, bins=n_bins)
    return H, xedges, yedges


def score_dissociation_candidate(mp, m1, m2, mn_i, mn_j, tolerance=0.15,
                                  exclude_radius=1.0):
    """
    Quantitatively scores how well a candidate dissociation pathway
    (mp -> m1 + m2) is supported by REAL data, instead of relying on
    eyeballing whether a plotted curve visually overlaps a busy scatter.

    Counts how many real (mn_i, mn_j) pairs fall within `tolerance` Da of
    the theoretical track. Points within `exclude_radius` of the trivial
    (m1, m2) coincidence point itself are excluded, since that point is
    usually a genuine, very bright peak regardless of whether any real
    dissociation is happening (it's just the ordinary undissociated
    ion-pair coincidence) - counting it would make every candidate look
    like a "hit" for free.

    Returns: n_on_track (count of real points along the curved part of
    the track, away from the trivial endpoint) - compare this across
    candidates; also compare against a random-pair control (see
    background_score below) to judge if it's actually above noise.
    """
    m1_prime, m2_prime = dissociation_track(mp, m1, m2, n_points=2000)

    from scipy.spatial import cKDTree
    track_points = np.column_stack([m1_prime, m2_prime])
    tree = cKDTree(track_points)

    data_points = np.column_stack([mn_i, mn_j])
    dist, idx = tree.query(data_points)
    on_track = dist < tolerance

    # Exclude points too close to the trivial (m1, m2) endpoint
    near_endpoint = np.hypot(mn_i - m1, mn_j - m2) < exclude_radius
    on_track = on_track & (~near_endpoint)

    return int(on_track.sum())


def background_score(mp, m1, m2, mn_i, mn_j, tolerance=0.15, exclude_radius=1.0,
                      n_shuffles=5, seed=0):
    """
    Control/null estimate: how many points would land 'on track' just by
    chance, if the mn_i/mn_j pairing were random (no real correlation).
    Shuffles mn_j relative to mn_i several times and averages the
    resulting on-track count. A real candidate's score (from
    score_dissociation_candidate) should be well ABOVE this background
    level to be taken seriously; if it's similar to background, the
    apparent "hits" are just coincidental scatter, not a real track.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_shuffles):
        shuffled_j = rng.permutation(mn_j)
        scores.append(score_dissociation_candidate(
            mp, m1, m2, mn_i, shuffled_j, tolerance, exclude_radius))
    return float(np.mean(scores)), float(np.std(scores))


def rank_candidates(candidates, mn_i, mn_j, tolerance=0.15, exclude_radius=1.0):
    """
    candidates: list of (mp, m1, m2, label) tuples.
    Scores each candidate against real data AND against a shuffled-data
    background, then prints a ranked table so you can see which (if any)
    candidates are genuinely above noise rather than just visually
    plausible.
    """
    results = []
    for mp, m1, m2, label in candidates:
        real_score = score_dissociation_candidate(mp, m1, m2, mn_i, mn_j,
                                                    tolerance, exclude_radius)
        bg_mean, bg_std = background_score(mp, m1, m2, mn_i, mn_j,
                                            tolerance, exclude_radius)
        # Simple significance proxy: how many std above background
        sigma = (real_score - bg_mean) / bg_std if bg_std > 0 else float('inf')
        results.append((label, real_score, bg_mean, sigma))

    results.sort(key=lambda r: -r[3])
    print(f"{'Candidate':45s} {'real':>8s} {'bg_mean':>10s} {'sigma_above_bg':>15s}")
    for label, real_score, bg_mean, sigma in results:
        print(f"{label:45s} {real_score:8d} {bg_mean:10.1f} {sigma:15.1f}")
    return results


def estimate_dissociation_occurrence(n_track_observed, detector_efficiency,
                                      total_ions_detected=None):
    """
    Paper Section 3.3's method for estimating true dissociation occurrence
    from observed correlation-histogram track counts (two charged ions).
    """
    n_true_events = n_track_observed / (detector_efficiency ** 2)
    result = {
        'n_track_observed': n_track_observed,
        'n_true_dissociation_events': n_true_events,
    }
    if total_ions_detected is not None:
        total_ions_evaporated = total_ions_detected / detector_efficiency
        result['total_ions_evaporated_estimate'] = total_ions_evaporated
        result['rate_ppm'] = 1e6 * n_true_events / total_ions_evaporated
    return result


def dissociation_occurrence_report(mp, m1, m2, mn_i, mn_j, detector_efficiency,
                                    total_ions_detected, tolerance=0.5,
                                    exclude_radius=1.0):
    """
    Convenience wrapper: scores a two-charged-ion candidate pathway against
    real pair data and estimates occurrence and ppm rate.
    """
    n_observed = score_dissociation_candidate(
        mp, m1, m2, mn_i, mn_j, tolerance=tolerance, exclude_radius=exclude_radius)
    est = estimate_dissociation_occurrence(
        n_observed, detector_efficiency, total_ions_detected)

    print(f"Observed on-track pairs (both fragments detected): {est['n_track_observed']}")
    print(f"Detector efficiency: {detector_efficiency}")
    print(f"Estimated TRUE dissociation events "
          f"(n_observed / efficiency^2): {est['n_true_dissociation_events']:.0f}")
    if 'rate_ppm' in est:
        print(f"Estimated total ions evaporated "
              f"(total_detected / efficiency): {est['total_ions_evaporated_estimate']:.0f}")
        print(f"Dissociation rate: {est['rate_ppm']:.1f} ppm "
              f"(per million evaporated ions)")
    return est


def dissociation_track(mp, m1, m2, n_points=200):
    """
    Eq. (2.8): parametric dissociation track for pathway  mp -> m1 + m2,
    swept over the fractional potential drop Vd/V0 in [0, 1).

        m1'(Vd/V0) = m1 * (1 - Vd/V0) / (1 - (mp/m1)*(Vd/V0))

    Returns arrays (m1_prime, m2_prime) tracing the curve from the
    (m1, m2) "no dissociation" endpoint toward the (mp, m2)-type
    coincidence point as Vd/V0 -> 1.
    """
    frac = np.linspace(0, 0.999, n_points)  # avoid singularity at Vd/V0 = 1 when mp==m1... etc.

    def shifted(m_target, m_other):
        return m_target * mp / (frac * m_target + (1 - frac) * mp)
        #return m_target * (1 - frac) / (1 - (mp / m_target) * frac)

    m1_prime = shifted(m1, m2)
    m2_prime = shifted(m2, m1)
    return m1_prime, m2_prime


def single_ion_band_bounds(mp, m1, m2, n_points=2000, max_mass=None):
    """
    Computes the theoretical shifted mass range [band_lo, band_hi] for the
    single detected charged product m1 when parent mp dissociates into
    m1 (charged) + m2 (neutral). m2 is used only for mass balance in
    dissociation_track(mp, m1, m2) and m2_prime is discarded.
    """
    m1_prime, _ = dissociation_track(mp, m1, m2, n_points=n_points)
    upper_bound = mp if max_mass is None else max_mass
    valid = np.isfinite(m1_prime) & (m1_prime >= 0.0) & (m1_prime <= upper_bound)

    if np.any(valid):
        vals = m1_prime[valid]
    else:
        vals = np.asarray([m1], dtype=float)

    band_lo = float(np.min(vals))
    band_hi = float(np.max(vals))
    return band_lo, band_hi, m1_prime


def score_single_ion_neutral_loss(mp, m1, m2, mn_all, exclude_peak_mass=None,
                                   exclude_peak_halfwidth=0.5,
                                   bg_width=1.0, bg_gap=0.25):
    """
    Quantifies 1-ionic + 1-neutral dissociation (e.g. SiO2+ -> Si+ + [O2])
    on the 1D `data['mn']` array (all ions, unfiltered by multiplicity).

    Calculates:
      - `n_in_band`: total ions in [m1_prime.min(), m1_prime.max()],
        optionally excluding a narrow core around `exclude_peak_mass`
        (e.g. nominal Si+ at ~28 Da) so the main undissociated peak
        does not swamp tail counts.
      - `n_bg_expected`: expected background ion count scaled by mass-width
        ratio from adjacent left/right control windows outside the band.
      - `n_excess`: max(0.0, n_in_band - n_bg_expected).
    """
    mn = np.asarray(mn_all, dtype=float)
    band_lo, band_hi, _ = single_ion_band_bounds(mp, m1, m2)

    in_band_mask = (mn >= band_lo) & (mn <= band_hi)
    if exclude_peak_mass is not None and exclude_peak_halfwidth > 0:
        in_band_mask &= np.abs(mn - exclude_peak_mass) > exclude_peak_halfwidth

    n_in_band = int(np.sum(in_band_mask))

    # Build left and right background windows just outside [band_lo, band_hi]
    bg_windows = []
    left_hi = band_lo - bg_gap
    left_lo = left_hi - bg_width
    if left_hi > left_lo:
        bg_windows.append((left_lo, left_hi))

    right_lo = band_hi + bg_gap
    right_hi = right_lo + bg_width
    if right_hi > right_lo:
        bg_windows.append((right_lo, right_hi))

    bg_counts = 0
    bg_total_width = 0.0
    for w_lo, w_hi in bg_windows:
        w_mask = (mn >= w_lo) & (mn <= w_hi)
        if exclude_peak_mass is not None and exclude_peak_halfwidth > 0:
            w_mask &= np.abs(mn - exclude_peak_mass) > exclude_peak_halfwidth
        bg_counts += int(np.sum(w_mask))
        bg_total_width += (w_hi - w_lo)

    # Effective scored mass width inside the band
    grid = np.linspace(band_lo, band_hi, 10000)
    grid_mask = np.ones_like(grid, dtype=bool)
    if exclude_peak_mass is not None and exclude_peak_halfwidth > 0:
        grid_mask &= np.abs(grid - exclude_peak_mass) > exclude_peak_halfwidth
    band_eff_width = float(np.mean(grid_mask) * (band_hi - band_lo))

    if bg_total_width > 0 and band_eff_width > 0:
        n_bg_expected = bg_counts * (band_eff_width / bg_total_width)
    else:
        n_bg_expected = 0.0

    n_excess = max(0.0, float(n_in_band) - n_bg_expected)

    return {
        'band_lo': band_lo,
        'band_hi': band_hi,
        'n_in_band': n_in_band,
        'n_bg_expected': n_bg_expected,
        'n_excess': n_excess,
        'bg_counts_raw': bg_counts,
        'bg_total_width': bg_total_width,
        'band_eff_width': band_eff_width,
    }


def neutral_loss_occurrence_report(mp, m1, m2, mn_all, detector_efficiency,
                                    total_ions_detected,
                                    exclude_peak_mass=None,
                                    exclude_peak_halfwidth=0.5,
                                    bg_width=1.0, bg_gap=0.25):
    """
    Reports 1D band counts, background-subtracted excess counts, and
    estimated true dissociation occurrence for a 1-ionic + 1-neutral
    pathway. Only ONE charged fragment is detected, so event count is
    corrected by a single power of `detector_efficiency`.
    """
    stats = score_single_ion_neutral_loss(
        mp, m1, m2, mn_all,
        exclude_peak_mass=exclude_peak_mass,
        exclude_peak_halfwidth=exclude_peak_halfwidth,
        bg_width=bg_width,
        bg_gap=bg_gap,
    )
    n_true_events = stats['n_excess'] / detector_efficiency
    total_ions_evaporated = total_ions_detected / detector_efficiency
    rate_ppm = 1e6 * n_true_events / total_ions_evaporated if total_ions_evaporated > 0 else 0.0

    print(f"Band [m1_prime.min(), m1_prime.max()]: [{stats['band_lo']:.4f}, {stats['band_hi']:.4f}] Da")
    print(f"Raw ions in band: {stats['n_in_band']}")
    print(f"Expected local background in band: {stats['n_bg_expected']:.1f}")
    print(f"Excess tail ions (above local bg): {stats['n_excess']:.1f}")
    print(f"Estimated TRUE dissociation events (excess / efficiency): {n_true_events:.0f}")
    print(f"Estimated total ions evaporated (total_detected / efficiency): {total_ions_evaporated:.0f}")
    print(f"Dissociation rate: {rate_ppm:.1f} ppm (per million evaporated ions)")

    return {
        **stats,
        'detector_efficiency': detector_efficiency,
        'n_true_dissociation_events': n_true_events,
        'total_ions_evaporated_estimate': total_ions_evaporated,
        'rate_ppm': rate_ppm,
    }


def plot_1d_neutral_loss_spectrum(mn_all, pathways, out_path='neutral_loss_1d_spectrum.png',
                                  m_range=(25.0, 65.0), bin_width=0.05):
    """
    Builds a plain 1D histogram of `mn_all` (all ions, not filtered by
    multiplicity and not pair-streamed) and overlays shaded vertical
    bands `[m1_prime.min(), m1_prime.max()]` for 1-ionic + 1-neutral
    dissociation pathways.
    """
    mn = np.asarray(mn_all, dtype=float)
    bins = np.arange(m_range[0], m_range[1] + bin_width, bin_width)
    counts, edges = np.histogram(mn, bins=bins)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(mn, bins=bins, histtype='stepfilled', color='steelblue',
            alpha=0.75, label='All ions m/n (unfiltered)')
    ax.set_yscale('log')
    ax.set_ylim(bottom=max(0.8, 0.5))

    for mp, m1, m2, label, color in pathways:
        band_lo, band_hi, _ = single_ion_band_bounds(mp, m1, m2)
        ax.axvspan(
            band_lo, band_hi,
            color=color, alpha=0.25,
            label=f"{label} [{band_lo:.2f}, {band_hi:.2f}] Da"
        )

    ax.set_xlim(m_range[0], m_range[1])
    ax.set_xlabel('Mass-to-charge ratio, m/n (Da)')
    ax.set_ylabel('Ion counts (log scale)')
    ax.set_title('1D Mass Spectrum with 1-Ionic + 1-Neutral Dissociation Band')
    ax.legend(loc='upper right', fontsize=7.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return counts, edges


def breakup_fraction_from_mass(mp, m1, m1_observed):
    """
    Inverse of Eq. (2.8): given an observed (measured) daughter-ion mass
    m1_observed, solve for the fractional potential drop Vd/V0 at which
    the parent ion (mass mp) must have dissociated to produce that shift.
    """
    m1_observed = np.asarray(m1_observed, dtype=float)
    denom = m1**2 - mp * m1_observed
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = m1 * (m1 - m1_observed) / denom
    frac = np.where((frac >= 0) & (frac < 1), frac, np.nan)
    return frac


def breakup_voltage_from_mass(mp, m1, m1_observed, v0):
    """
    Same as breakup_fraction_from_mass, but scaled by the ACTUAL recorded
    accelerating voltage (V0) for that event to give a real Vd in volts.
    """
    frac = breakup_fraction_from_mass(mp, m1, m1_observed)
    return frac * np.asarray(v0, dtype=float)


def breakup_location_histogram(mp, m1, m2, mn_i, mn_j, tolerance=0.15, bins=40):
    """
    Selects the subset of real ion-pair data points lying close to the
    theoretical dissociation track for pathway mp -> m1 + m2, converts
    each one to its implied Vd/V0 via breakup_fraction_from_mass, and
    returns a 1D histogram of breakup locations along the flight path.
    """
    track_m1, track_m2 = dissociation_track(mp, m1, m2, n_points=2000)

    from scipy.spatial import cKDTree
    track_points = np.column_stack([track_m1, track_m2])
    tree = cKDTree(track_points)
    data_points = np.column_stack([mn_i, mn_j])
    dist, _ = tree.query(data_points)
    on_track = dist < tolerance

    frac = breakup_fraction_from_mass(mp, m1, mn_i[on_track])
    frac = frac[~np.isnan(frac)]

    counts, edges = np.histogram(frac, bins=bins, range=(0, 1))
    return counts, edges, frac

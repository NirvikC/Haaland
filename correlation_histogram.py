"""
Correlation histograms ('Saxey plots') and molecular-ion-dissociation
track overlays, following Saxey (2011) Sections 2.2 and 2.3.
"""

import numpy as np
import pandas as pd
from itertools import permutations


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
    from observed correlation-histogram track counts.

    Physical reasoning: seeing a dissociation event in the correlation
    histogram requires BOTH fragment ions to be independently detected.
    If the detector catches any single ion with probability
    `detector_efficiency` (e.g. 0.55 for 55%), the probability of catching
    BOTH fragments of one dissociation event is detector_efficiency**2.
    So the true number of dissociation events is:

        n_true_events = n_track_observed / detector_efficiency**2

    If `total_ions_detected` is given (e.g. len(data), your total detected
    ion count), this also estimates:
      - total_ions_evaporated: the true number of ions that left the
        surface (only a fraction `detector_efficiency` of these are ever
        detected at all, matching the paper's "34 million ions field
        evaporated" figure, which is corrected by a SINGLE power of
        efficiency, not squared, since it's about single-ion detection)
      - rate_ppm: the dissociation rate expressed as parts-per-million of
        total evaporated ions, directly comparable to the paper's
        "~600 per million evaporated ions" result for GaN2+.

    Returns a dict with all computed quantities.
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
    Convenience wrapper: scores a candidate pathway against real data
    (score_dissociation_candidate) and immediately converts the result
    into an estimated true occurrence count and ppm rate, printing a
    short summary in the same style as the paper's worked GaN2+ example.
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
        return m_target * (1 - frac) / (1 - (mp / m_target) * frac)

    m1_prime = shifted(m1, m2)
    m2_prime = shifted(m2, m1)
    return m1_prime, m2_prime


def breakup_fraction_from_mass(mp, m1, m1_observed):
    """
    Inverse of Eq. (2.8): given an observed (measured) daughter-ion mass
    m1_observed, solve for the fractional potential drop Vd/V0 at which
    the parent ion (mass mp) must have dissociated to produce that shift.
    This is what actually tells you WHERE along the flight path (in
    potential-drop terms) a given detected event's breakup occurred -
    not just that it lies "somewhere on the track".

        Vd/V0 = m1*(m1 - m1_observed) / (m1**2 - mp*m1_observed)

    Values near 0 = breakup happened almost immediately at the surface.
    Values near 1 = breakup happened very late, after most of the
    accelerating field had already been traversed.
    Returns NaN for points that don't correspond to a physically valid
    (0 <= Vd/V0 < 1) solution - i.e. points that aren't actually on this
    particular track.
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
    accelerating voltage (V0) for that event to give a real Vd in volts,
    not just the dimensionless Vd/V0 fraction.

    v0 should be the per-event voltage recorded for that ion in your
    ePOS data (typically data['standing_v'] for laser-pulsed acquisitions,
    or standing_v + pulse_v for voltage-pulsed acquisitions - check which
    applies to your SiO2 dataset's acquisition mode).
    """
    frac = breakup_fraction_from_mass(mp, m1, m1_observed)
    return frac * np.asarray(v0, dtype=float)


def breakup_location_histogram(mp, m1, m2, mn_i, mn_j, tolerance=0.15, bins=40):
    """
    Selects the subset of real ion-pair data points lying close to the
    theoretical dissociation track for pathway mp -> m1 + m2, converts
    each one to its implied Vd/V0 via breakup_fraction_from_mass, and
    returns a 1D histogram of breakup locations along the flight path -
    directly reproducing the qualitative "dissociates near the surface"
    vs "dissociates late" comparisons made in the paper (Section 2.3),
    but as a quantitative distribution instead of an eyeballed track.

    `tolerance` (Da) sets how close a real data point must be to the
    theoretical track curve (evaluated at the same m1' value) to be
    counted as belonging to this dissociation pathway.
    """
    track_m1, track_m2 = dissociation_track(mp, m1, m2, n_points=2000)

    # For each real (mn_i, mn_j) pair, find the closest point on the track
    # and keep only those within `tolerance` of it (i.e. genuinely on the line)
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

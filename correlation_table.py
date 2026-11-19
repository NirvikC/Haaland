"""
Correlation table analysis, following Saxey (2011) Section 2.1.

Given per-ion (pulse_number, species) data, this builds:
  - O_ij : observed ion-pair coincidence counts within multi-hit events
  - E_ij : expected counts under independent evaporation (Eq. 2.2)
  - Z_ij : deviation score (Eq. 2.3)
with the diagonal-neutralisation correction described in the paper
(observed self-pairs are biased low by detector pile-up, so they are
replaced with values that converge toward their own expectation value
before finalising the off-diagonal deviations).
"""

import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
from scipy.stats import norm


def build_pair_counts(pulse_number, species, min_multiplicity=2, exclude_unranged=True):
    """
    Group ions by pulse and enumerate all unordered ion-pairs within each
    multi-hit event. Returns a Counter of {(species_i, species_j): count}
    with species_i <= species_j (unordered, as in the paper: n_ij = n_ji).
    """
    df = pd.DataFrame({'pulse': pulse_number, 'species': species})
    if exclude_unranged:
        df = df[df['species'] != '']

    pair_counts = Counter()
    for _, grp in df.groupby('pulse'):
        labels = grp['species'].tolist()
        if len(labels) < min_multiplicity:
            continue
        for a, b in combinations(labels, 2):
            key = tuple(sorted((a, b)))
            pair_counts[key] += 1

    return pair_counts


def counts_to_matrix(pair_counts):
    """Convert the pair-count dict into a symmetric observed-count matrix O_ij."""
    species = sorted({s for pair in pair_counts for s in pair})
    idx = {s: i for i, s in enumerate(species)}
    n = len(species)
    O = np.zeros((n, n))

    for (a, b), c in pair_counts.items():
        i, j = idx[a], idx[b]
        if i == j:
            O[i, i] += c        # true self-pairs stay on the diagonal
        else:
            O[i, j] += c
            O[j, i] += c

    return O, species


def element_key_from_species_name(name):
    """
    Reduces a species name like 'W(2+)' or 'W(3+)' down to its base
    formula key, ignoring ONLY charge state - e.g. both above reduce to
    'W'. Atom-count multipliers are kept intact, so 'W2(3+)' reduces to
    'W2', NOT 'W' - a W2 dimer is a physically different species from a
    monatomic W ion (different formation mechanism, different expected
    correlation behavior), so it stays a separate row/column in the
    correlation table rather than being folded into plain 'W'.

    (This previously also stripped atom-count digits, which incorrectly
    merged e.g. 'W2(3+)' and 'W(2+)' into one 'W' key - fixed.)

    Method: strip only the '(...)' charge annotation. Compounds like
    'NbO(1+)' reduce to 'NbO' (kept as one compound key, not split into
    'Nb' and 'O' separately) - this collapses charge states of the same
    formula together, nothing else.
    """
    import re
    return re.sub(r'\(.*?\)', '', name)   # strip '(2+)' etc. only


def collapse_matrix_to_elements(O, species_order):
    """
    Collapses a charge-state-resolved observed-count matrix O (indexed by
    species_order, e.g. 'W(2+)', 'W(3+)', 'Mo(1+)', ...) down to a
    formula-resolved matrix, summing counts across every charge-state
    combination of the same formula - e.g. O[W(2+), Mo(1+)] +
    O[W(2+), Mo(2+)] + O[W(3+), Mo(1+)] + ... all sum into one
    O_elem[W, Mo] cell. W2 (dimer) is NOT folded into W (monomer) - they
    keep separate rows/columns, since they're physically distinct
    species; only the charge-state annotation is stripped, not the
    atom-count.

    Returns (O_elem, element_order) in the SAME format as
    counts_to_matrix()'s output, so it drops straight into
    deviation_matrix()/probability_matrix() unchanged - collapsing
    happens at the raw-count level, BEFORE expected counts and Z-scores
    are computed, which is the statistically correct order: summing
    already-computed Z_ij values across charge states would not be
    valid (Z isn't additive), but summing the underlying O_ij counts and
    then recomputing E and Z once at the formula level is.

    NOTE on the diagonal: O's diagonal holds TRUE self-pairs (two ions
    of the exact same charge-resolved species in one multi-hit event -
    see counts_to_matrix()). When collapsing, cross-charge-state pairs
    of the SAME formula (e.g. one W(2+) + one W(3+) in the same event)
    move onto the collapsed diagonal too, since they're now "the same
    formula with itself". That's the correct behavior for a
    formula-level self-pair count, but it means the collapsed diagonal
    is not simply a subset of the original diagonal - it also gains
    same-formula cross-charge-state pairs that were off-diagonal before
    collapsing.
    """
    element_order = sorted({element_key_from_species_name(s) for s in species_order})
    idx = {e: i for i, e in enumerate(element_order)}
    n = len(element_order)
    O_elem = np.zeros((n, n))

    species_to_elem_idx = [idx[element_key_from_species_name(s)] for s in species_order]
    for i, ei in enumerate(species_to_elem_idx):
        for j, ej in enumerate(species_to_elem_idx):
            if O[i, j] != 0:
                O_elem[ei, ej] += O[i, j]

    return O_elem, element_order


def expected_matrix(O):
    """
    Eq. (2.2): E_ij = n_i * n_j / N
    n_k = row/column sum (total pairs touching species k)
    N   = total number of pairs in the table
    """
    n_k = O.sum(axis=1)
    N = O.sum() / 2.0  # each off-diagonal pair counted twice, diagonal once effectively double-counted too
    # Use the convention from the paper: N = sum over the whole table of n_ij (upper triangle incl. diagonal)
    N = np.triu(O).sum()
    E = np.outer(n_k, n_k) / (2 * N) if N > 0 else np.zeros_like(O)
    return E, n_k, N


def probability_matrix(Z, E, min_expected=5):
    """
    Converts the unbounded Z-score deviation matrix (Eq. 2.3) into a
    normalized [0, 1] scale using the standard normal CDF:

        P_ij = Phi(Z_ij)

    where Phi is the cumulative distribution function of the standard
    normal distribution. This is a genuine probability statement, not
    an arbitrary rescaling: under the null hypothesis of independent
    evaporation, Z_ij is (approximately, for large counts) a standard
    normal variable, so Phi(Z_ij) is literally "the probability that a
    standard normal variable is less than or equal to the observed
    deviation" - equivalently, it's the one-sided percentile of how
    extreme the observed count is relative to the independent-
    evaporation expectation.

    Interpretation of the resulting scale:
      - 0.5  = no deviation from independence (Z=0, O_ij == E_ij exactly)
      - -> 1 = strong POSITIVE correlation (species i and j co-evaporate
               far more often than chance; Z >> 0)
      - -> 0 = strong NEGATIVE correlation / anti-correlation (species i
               and j co-evaporate far LESS often than chance; Z << 0)

    This preserves the direction of the original Z-score (unlike, say,
    a two-tailed p-value, which is symmetric and loses the sign) while
    bounding everything into a fixed, directly comparable [0,1] range -
    useful for comparing correlation STRENGTH across different datasets
    or species pairs where the raw Z magnitude isn't directly comparable
    (Z depends on absolute counts, not just the underlying correlation
    strength).

    Cells with expected count below min_expected (statistically
    insignificant, same threshold used to zero them out in Z) are set
    to the neutral value 0.5 rather than left at their (unreliable) Z
    value, since there isn't enough data to say anything - 0.5 signals
    "no information" rather than misleadingly implying no correlation
    at all (which 0 or 1 would suggest) or a spurious deviation.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        P = norm.cdf(Z)
    P[~np.isfinite(P)] = 0.5
    P[E < min_expected] = 0.5
    return P


def neutralise_diagonal(O, n_iter=25):
    """
    Iteratively replace the (unreliable) observed diagonal O_ii with values
    that converge toward their own expectation E_ii, so the diagonal does
    not bias the off-diagonal Z_ij values (paper, Section 2.1).
    """
    O_adj = O.copy().astype(float)
    for _ in range(n_iter):
        E, n_k, N = expected_matrix(O_adj)
        np.fill_diagonal(O_adj, np.diag(E))
    return O_adj


def deviation_matrix(O, neutralise_diag=True):
    """
    Full pipeline: observed counts -> (optionally neutralised) -> expected
    -> Z-score deviation matrix (Eq. 2.3).
    Returns Z, E, species-ordering-independent (matrices only).
    """
    O_use = neutralise_diagonal(O) if neutralise_diag else O.astype(float)
    E, n_k, N = expected_matrix(O_use)

    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (O_use - E) / np.sqrt(E)
    Z[~np.isfinite(Z)] = 0.0

    # Zero out cells with expected < 5, as the paper does (statistically insignificant)
    Z[E < 5] = 0.0

    return Z, E


def correlation_table(pulse_number, species, min_multiplicity=2):
    """
    Convenience wrapper: raw per-ion arrays -> deviation matrix + species order.
    """
    pair_counts = build_pair_counts(pulse_number, species, min_multiplicity)
    O, species_order = counts_to_matrix(pair_counts)
    Z, E = deviation_matrix(O)
    P = probability_matrix(Z, E)
    return Z, E, P, species_order, O

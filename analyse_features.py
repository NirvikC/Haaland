"""
Feature analysis for the window-by-window undercounting data set
(window_features.csv from extract_epos_features.py), designed for SMALL
numbers of windows (n ~ 20).

    python analyse_features.py                 # uses the CONFIG below
    python analyse_features.py my.csv out_dir  # CSV and output folder
    python analyse_features.py --quick         # fewer resamples (fast test)

Steps (everything is written to OUT_DIR):

  1. Signal check (no ML)
       01_target_vs_window.png   target_d vs window index with target_d_stderr
                                 error bars, weighted mean, standardised
                                 residuals. Chi-square test for heterogeneity
                                 (is the spread larger than the error bars?),
                                 I^2, extra scatter tau, trend test, and the
                                 same per run when there are several runs.
  2. Cut the features down BEFORE looking at the target
       features_kept.csv         kept features, family, cluster, representative
       04_cluster_heatmap.png    clustered |Spearman| heatmap + dendrogram
       Dropped: ids, target_*, raw (window-size dependent) counts, constant
       columns, columns with too many NaN. Hierarchical clustering on
       1 - |Spearman|, one representative (medoid) per cluster.
  3. Univariate screening (no ML)
       spearman_screen.csv       Spearman rho + bootstrap CI, permutation p,
                                 family-wise p from the permutation maximum
                                 of |rho|, and partial rho controlling for
                                 window index and laser energy / temperature
       05_rho_dotplot.png        sorted rho with CI (representatives)
       06_scatter_top.png        scatter grid of the top features
  4. Physics baseline (optional, PHYSICS_COLUMNS): target_d against the
     1D neutral-loss and 2D dissociation estimates.    04b_physics_check.png
  5. ML that survives n ~ 20 (honest cross-validation only)
       cv_results.csv            out-of-fold R2 / RMSE / Spearman for: mean,
                                 metadata-only, position-only, lasso, elastic
                                 net, PLS, top-k OLS (selection inside the
                                 folds), random forest, gradient boosting.
                                 CV = leave-one-run-out (>= 3 runs) or
                                 leave-one-window-out with a buffer of CV_GAP
                                 neighbouring windows removed from training.
                                 p-values: shuffled-target AND circular-shift
                                 permutation of the target (the shift keeps
                                 the autocorrelation, so it is the fairer one).
       stability_selection.csv/png   selection frequency along the lasso path
       pls_vip.csv               PLS VIP scores
       family_importance.csv/png permuting whole feature families
       07_cv_predictions.png     out-of-fold prediction of the best model
       final_model.csv           top-k (about n/5) model, standardised coefs
       feature_summary.csv       everything per feature + a vote count
       summary.txt               plain-language summary of all of the above

Read the results as: a feature matters only if it is picked by SEVERAL
independent criteria (family-wise p, partial correlation, stability, VIP) and
the CV R2 of the model using it beats both the mean baseline and the
permutation null. With ~20 windows, "nothing stands out" is a legitimate
result.
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from joblib import Parallel, delayed

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (ElasticNetCV, LassoCV, LinearRegression,
                                  RidgeCV, lasso_path)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ======================= CONFIG =======================
CSV_PATH = 'window_features.csv'
OUT_DIR = '.'

TARGET = 'target_d'              # or 'target_d_bgcorr'
STDERR = 'target_d_stderr'
RUN_COL = 'run_id'
POS_COL = 'window_idx'           # position of the window within its run (depth/time proxy)
METADATA_COLS = ['laser_energy_pJ', 'temperature_K']
CONTROL_COLS = ['window_idx'] + METADATA_COLS   # controlled for in partial correlations

MAX_NAN_FRAC = 0.2               # drop feature columns with more NaN than this
N_CLUSTERS = 15                  # representatives = clusters
ML_FEATURES = 'representatives'  # or 'all' (all kept features go into the ML models)

N_BOOT = 2000                    # bootstrap resamples for the rho CI
N_PERM = 10000                   # permutations for the rho p-values
N_STAB = 500                     # subsamples for stability selection
STAB_FRAC = 0.7                  # fraction of windows per subsample
STAB_THRESHOLD = 0.8             # "chosen in most subsamples"
STAB_Q = None                    # avg. number of features selected on the lasso path used;
                                 # None -> max(3, final model size). Smaller = stricter
TOP_K = None                     # final model size; None -> max(1, n // 5)

CV_GAP = 1                       # windows removed on each side of the test window (1 run)
PERM_CV = 100                    # target permutations for linear / PLS models
PERM_CV_TREES = 20               # target permutations for RF / GBM (slow)
N_JOBS = -1

# e.g. ['nl_path1__rate_ppm_high', 'd2d_SiO2_2p_to_SiOp_Op__rate_ppm_high'] (step 4)
PHYSICS_COLUMNS = []

SEED = 0
# ======================================================[cite: 10, 13]

ID_COLS = {RUN_COL, POS_COL, 'ion_min', 'ion_max', 'ion_lo', 'ion_hi'}
WEIGHT_COLS = {'n_ions', 'n_pairs_total'}      # kept in the CSV, never used as features
_RAW_SUFFIXES = ('__n_in_band', '__n_bg_expected', '__n_excess', '__A_best', '__A_low',
                 '__A_high', '__n_true_events', '__n_true_events_low',
                 '__n_true_events_high', '__unr_n_in_band', '__unr_n_excess')
_RAW_PREFIXES = ('O_ij__', 'E_ij__', 'Z_ij__', 'peak_', 'tail_hi_excess_', 'tail_hi_',
                 'tail_lo_')


def is_raw_count(key):
    """Window-size dependent columns (same rule as extract_epos_features.py)."""
    if key.endswith(('__per_ion', '__per_pair')) or key.startswith('O_over_E__'):
        return False
    if key.startswith(('tail_hi_over_peak_', 'tail_asym_')):
        return False
    return (key.startswith(_RAW_PREFIXES) or key.endswith(_RAW_SUFFIXES)
            or key in ('n_unranged', 'pulses_elapsed'))


FAMILY_RULES = [
    ('metadata', lambda c: c in METADATA_COLS),
    ('pair_correlation', lambda c: re.match(r'^(O_ij|E_ij|O_over_E|Z_ij)__', c)),
    ('neutral_loss', lambda c: c.startswith(('nl_', 'd2d_'))),
    ('peak_tails', lambda c: re.match(r'^(peak_|tail_)', c)),
    ('species_ratio', lambda c: re.search(r'__frac$|^(O|Si|SiO)\d?_over_', c)),
    ('field_pulsing', lambda c: re.match(r'^(standing_v|pulse_v|pulse_fraction)', c)),
    ('detection_rate', lambda c: re.match(
        r'^(ions_per_pulse|events_per_pulse|.*pulses_between_events|pulses_elapsed)', c)),
    ('detector', lambda c: re.match(r'^(det_|minsep|multi_)', c)),
    ('background_ranging', lambda c: re.match(r'^(bg_density|frac_unranged|unranged|ranged_)', c)),
][cite: 10, 13]


def family_of(col):
    for name, rule in FAMILY_RULES:
        if rule(col):
            return name
    return 'other'


LOG = []


def pfmt(p):
    return '<1e-16' if (np.isfinite(p) and p < 1e-16) else f"{p:.3g}"


def say(msg=''):
    print(msg)
    LOG.append(str(msg))


# ---------------------------------------------------------------- utilities
def rank_z(M):
    """Column ranks, centred and scaled to unit L2 norm: dot products of the
    columns are Spearman correlations."""
    R = stats.rankdata(np.asarray(M, dtype=float), axis=0)
    R = R - R.mean(axis=0)
    nrm = np.sqrt((R ** 2).sum(axis=0))
    nrm[nrm == 0] = np.nan
    return R / nrm


def spearman_vec(X, y):
    return rank_z(X).T @ rank_z(y[:, None])[:, 0]


def r2_score_oof(y, pred):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ------------------------------------------------------------------ loading
def load(csv_path):
    df = pd.read_csv(csv_path)
    for c in (TARGET, STDERR):
        if c not in df.columns:
            sys.exit(f"column '{c}' not found in {csv_path}")
    if RUN_COL not in df.columns:
        df[RUN_COL] = 'run'
    if POS_COL not in df.columns:
        df[POS_COL] = np.arange(len(df))
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[np.isfinite(df[TARGET])].copy()
    df = df.sort_values([RUN_COL, POS_COL]).reset_index(drop=True)
    return df


# -------------------------------------------------------- step 1: signal check
def step1(df, out):
    y = df[TARGET].to_numpy(float)
    se = df[STDERR].to_numpy(float)
    ok = np.isfinite(se) & (se > 0)
    if not ok.all():
        say(f"  !! {int((~ok).sum())} window(s) without a valid {STDERR}; excluded from step 1")
    d = df[ok]
    y, se = y[ok], se[ok]
    pos = d[POS_COL].to_numpy(float)
    runs = d[RUN_COL].astype(str).to_numpy()
    w = 1.0 / se ** 2
    mu = float(np.sum(w * y) / np.sum(w))
    Q = float(np.sum(w * (y - mu) ** 2))
    dof = len(y) - 1
    p_het = float(stats.chi2.sf(Q, dof)) if dof > 0 else np.nan
    I2 = max(0.0, (Q - dof) / Q) if Q > 0 else 0.0
    denom = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau = float(np.sqrt(max(0.0, (Q - dof) / denom))) if denom > 0 else np.nan
    say(f"  n windows = {len(y)}, weighted mean {TARGET} = {mu:.4f}, "
        f"median stderr = {np.median(se):.4f}, std of {TARGET} = {y.std(ddof=1):.4f}")
    say(f"  heterogeneity: Q = {Q:.1f} on {dof} dof, p = {p_het:.3g}, I^2 = {I2:.2f}, "
        f"extra scatter tau = {tau:.4f}")
    uniq = list(dict.fromkeys(runs))
    if len(uniq) > 1:
        Qw = 0.0
        for r in uniq:
            m = runs == r
            mr = np.sum(w[m] * y[m]) / np.sum(w[m])
            Qr = float(np.sum(w[m] * (y[m] - mr) ** 2))
            Qw += Qr
            say(f"    run {r}: n={int(m.sum())}, mean={mr:.4f}, Q={Qr:.1f} "
                f"(p={pfmt(stats.chi2.sf(Qr, max(1, m.sum() - 1)))})")
        Qb = Q - Qw
        say(f"    between-run Q = {Qb:.1f} on {len(uniq) - 1} dof, "
            f"p = {pfmt(stats.chi2.sf(Qb, len(uniq) - 1))}")
    # trend with window position (drift): Spearman, overall and within runs
    rho_t, p_t = stats.spearmanr(pos, y) if len(y) > 3 else (np.nan, np.nan)
    say(f"  trend of {TARGET} with {POS_COL}: Spearman rho = {rho_t:+.2f} (p = {p_t:.3g})"
        + ("   <-- drift: expect many features to correlate through it" if p_t < 0.05 else ""))
    if p_het > 0.05:
        verdict = ("The spread between windows is consistent with the error bars: "
                   "there is little real window-to-window variation to explain.")
    else:
        verdict = (f"The windows differ by more than the error bars (p = {pfmt(p_het)}); "
                   f"extra scatter about {tau:.3f} in {TARGET}.")
    say("  -> " + verdict)

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1]})
    cmap = plt.get_cmap('tab10')
    for i, r in enumerate(uniq):
        m = runs == r
        ax[0].errorbar(pos[m], y[m], yerr=se[m], fmt='o-', ms=5, lw=0.8, capsize=3,
                       color=cmap(i % 10), label=f'{r}')
        ax[1].plot(pos[m], (y[m] - mu) / se[m], 'o-', ms=5, lw=0.8, color=cmap(i % 10))
    ax[0].axhline(mu, color='k', ls='--', lw=1, label=f'weighted mean {mu:.3f}')
    ax[0].set_ylabel(TARGET)
    ax[0].set_title(f"Signal check: chi2 = {Q:.1f} / {dof} dof, p = {pfmt(p_het)}, "
                    f"I2 = {I2:.2f}, tau = {tau:.3f}")
    ax[0].legend(fontsize=8)
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].axhspan(-1, 1, color='gray', alpha=0.2)
    ax[1].set_ylabel('(d - mean) / stderr')
    ax[1].set_xlabel(POS_COL)
    plt.tight_layout()
    plt.savefig(os.path.join(out, '01_target_vs_window.png'), dpi=150)
    plt.close(fig)
    return {'Q': Q, 'dof': dof, 'p_het': p_het, 'I2': I2, 'tau': tau, 'trend_p': p_t,
            'verdict': verdict}


# ------------------------------------------------ step 2: select / cluster features
def select_features(df):
    cols = []
    dropped = {'id/target/weight': 0, 'raw count': 0, 'non-numeric': 0,
               'too many NaN': 0, 'constant': 0}
    for c in df.columns:
        if c in ID_COLS or c in WEIGHT_COLS or c.startswith('target_'):
            dropped['id/target/weight'] += 1
            continue
        if is_raw_count(c):
            dropped['raw count'] += 1
            continue
        s = pd.to_numeric(df[c], errors='coerce')
        if s.isna().all():
            dropped['non-numeric'] += 1
            continue
        if s.isna().mean() > MAX_NAN_FRAC:
            dropped['too many NaN'] += 1
            continue
        if s.nunique(dropna=True) <= 1:
            dropped['constant'] += 1
            continue
        cols.append(c)
    X = df[cols].apply(pd.to_numeric, errors='coerce')
    n_imp = int(X.isna().sum().sum())
    X = X.fillna(X.median())
    # exact duplicates after rank transform carry no extra information
    say(f"  kept {len(cols)} features; dropped: " +
        ", ".join(f"{k} {v}" for k, v in dropped.items() if v) +
        (f"; {n_imp} NaN value(s) filled with the column median" if n_imp else ""))
    return X


def cluster_features(X, n_clusters):
    p = X.shape[1]
    Z = rank_z(X.to_numpy())
    C = np.nan_to_num(Z.T @ Z, nan=0.0)
    np.fill_diagonal(C, 1.0)
    D = 1.0 - np.abs(C)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)
    k = max(1, min(n_clusters, p))
    if p == 1:
        return np.array([1]), np.array([0]), None, C
    link = hierarchy.linkage(squareform(D, checks=False), method='average')
    lab = hierarchy.fcluster(link, t=k, criterion='maxclust')
    reps = []
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        if len(idx) == 1:
            reps.append(idx[0])
        else:   # medoid: highest mean |rho| to the others in its cluster (target-blind)
            sub = np.abs(C[np.ix_(idx, idx)]).mean(axis=1)
            reps.append(idx[int(np.argmax(sub))])
    return lab, np.array(reps), link, C


def plot_heatmap(X, lab, reps, link, C, fam, out):
    p = X.shape[1]
    if link is None or p < 2:
        return
    order = hierarchy.leaves_list(link)
    fig = plt.figure(figsize=(max(8, 0.16 * p + 4), max(7, 0.16 * p + 3)))
    axd = fig.add_axes([0.02, 0.28, 0.12, 0.68])
    hierarchy.dendrogram(link, orientation='left', no_labels=True, ax=axd,
                         color_threshold=0, above_threshold_color='gray')
    axd.axis('off')
    axh = fig.add_axes([0.15, 0.28, 0.68, 0.68])
    im = axh.imshow(np.abs(C)[np.ix_(order, order)][::-1], vmin=0, vmax=1, cmap='viridis',
                    aspect='auto')
    names = [X.columns[i] for i in order]
    rep_set = set(reps.tolist())
    axh.set_xticks(range(p))
    axh.set_xticklabels([('* ' if order[j] in rep_set else '') + names[j] for j in range(p)],
                        rotation=90, fontsize=max(4, 8 - p // 25))
    axh.set_yticks([])
    cax = fig.add_axes([0.85, 0.28, 0.015, 0.68])
    plt.colorbar(im, cax=cax, label='|Spearman|')
    axc = fig.add_axes([0.835, 0.28, 0.01, 0.68])
    fams = sorted(set(fam.values()))
    fcol = {f: plt.get_cmap('tab10')(i % 10) for i, f in enumerate(fams)}
    axc.imshow(np.array([[matplotlib.colors.to_rgb(fcol[fam[names[j]]])]
                         for j in range(p)][::-1]), aspect='auto')
    axc.axis('off')
    handles = [matplotlib.patches.Patch(color=fcol[f], label=f) for f in fams]
    fig.legend(handles=handles, loc='lower right', fontsize=7, title='family (strip)')
    fig.suptitle(f"{p} features, {len(rep_set)} clusters (* = representative)", fontsize=10)
    plt.savefig(os.path.join(out, '04_cluster_heatmap.png'), dpi=150)
    plt.close(fig)


# ------------------------------------------------ step 3: univariate screening
def partial_spearman(X, y, controls, names):
    """Partial Spearman rho of each X column with y, controlling for `controls`
    (dict name -> array). A control equal to the feature itself is skipped."""
    n = len(y)
    ry = stats.rankdata(y)
    ctrl_r = {k: stats.rankdata(v) for k, v in controls.items()}
    out = np.full(X.shape[1], np.nan)
    pv = np.full(X.shape[1], np.nan)
    for j, nm in enumerate(names):
        cs = [v for k, v in ctrl_r.items() if k != nm]
        k = len(cs)
        if n - k - 3 < 3:
            continue
        A = np.column_stack([np.ones(n)] + cs) if cs else np.ones((n, 1))
        Q, _ = np.linalg.qr(A)
        rx = stats.rankdata(X[:, j])
        ex = rx - Q @ (Q.T @ rx)
        ey = ry - Q @ (Q.T @ ry)
        den = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
        if den == 0:
            continue
        r = float((ex * ey).sum() / den)
        out[j] = r
        df_ = n - 2 - k
        t = r * np.sqrt(df_ / max(1e-12, 1 - r ** 2))
        pv[j] = 2 * stats.t.sf(abs(t), df_)
    return out, pv


def get_controls(df):
    ctrl = {}
    for c in CONTROL_COLS:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors='coerce')
            if v.notna().all() and v.nunique() > 1:
                ctrl[c] = v.to_numpy(float)
    return ctrl


def step3(df, X, y, lab, reps, fam, out, rng, quick_counts):
    n, p = X.shape
    Xv = X.to_numpy(float)
    names = list(X.columns)
    n_boot, n_perm = quick_counts['n_boot'], quick_counts['n_perm']

    zx, zy = rank_z(Xv), rank_z(y[:, None])[:, 0]
    rho = zx.T @ zy
    # permutation null of rho (shuffled windows)
    perms = np.argsort(rng.random((n_perm, n)), axis=1)
    rho_null = zy[perms] @ np.nan_to_num(zx)                 # (n_perm, p)
    absn = np.abs(rho_null)
    rep_idx = np.array(sorted(reps))
    max_rep = absn[:, rep_idx].max(axis=1)
    max_all = absn.max(axis=1)
    p_perm = (1 + (absn >= np.abs(rho)[None, :] - 1e-12).sum(axis=0)) / (n_perm + 1)
    p_fw_rep = (1 + (max_rep[:, None] >= np.abs(rho)[None, :] - 1e-12).sum(axis=0)) / (n_perm + 1)
    p_fw_all = (1 + (max_all[:, None] >= np.abs(rho)[None, :] - 1e-12).sum(axis=0)) / (n_perm + 1)

    # bootstrap CI of rho
    boots = np.full((n_boot, p), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if np.all(yb == yb[0]):
            continue
        boots[b] = spearman_vec(Xv[idx], yb)
    ci_lo = np.nanpercentile(boots, 2.5, axis=0)
    ci_hi = np.nanpercentile(boots, 97.5, axis=0)

    ctrl = get_controls(df)
    prho, pp = partial_spearman(Xv, y, ctrl, names)
    say(f"  partial correlations control for: {list(ctrl) if ctrl else 'nothing (no varying controls)'}")

    tab = pd.DataFrame({
        'feature': names, 'family': [fam[c] for c in names], 'cluster': lab,
        'is_representative': [i in set(rep_idx.tolist()) for i in range(p)],
        'rho': rho, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'ci_excludes_0': (ci_lo > 0) | (ci_hi < 0),
        'p_perm': p_perm, 'p_fwer_reps': p_fw_rep, 'p_fwer_all': p_fw_all,
        'rho_partial': prho, 'p_partial': pp,
    })
    tab['abs_rho'] = tab['rho'].abs()
    tab = tab.sort_values('abs_rho', ascending=False).reset_index(drop=True)
    tab.drop(columns='abs_rho').to_csv(os.path.join(out, 'spearman_screen.csv'), index=False)

    reps_tab = tab[tab.is_representative].copy()
    n_fw = int((reps_tab.p_fwer_reps < 0.05).sum())
    say(f"  {len(reps_tab)} representatives; family-wise p < 0.05 for {n_fw}; "
        f"expected by chance without correction: {0.05 * len(reps_tab):.1f}")
    say("  top representatives:")
    for _, r in reps_tab.head(8).iterrows():
        say(f"    {r.feature:55s} rho={r.rho:+.2f} [{r.ci_lo:+.2f},{r.ci_hi:+.2f}] "
            f"p_fwer={r.p_fwer_reps:.3f} partial={r.rho_partial:+.2f} (p={r.p_partial:.3f})")

    # --- plots
    fams = sorted(set(fam.values()))
    fcol = {f: plt.get_cmap('tab10')(i % 10) for i, f in enumerate(fams)}
    d = reps_tab.sort_values('rho')
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(d) + 1.5)))
    yy = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        c = fcol[r.family]
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=c, lw=1.5)
        ax.plot(r.rho, i, 'o', color=c, mfc=(c if r.p_fwer_reps < 0.05 else 'white'), ms=7)
        if np.isfinite(r.rho_partial):
            ax.plot(r.rho_partial, i, 'd', color='k', ms=4)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(d.feature, fontsize=7)
    ax.set_xlabel(f'Spearman rho with {TARGET}   (filled = family-wise p < 0.05; '
                  f'black diamond = partial rho)')
    ax.set_xlim(-1, 1)
    handles = [matplotlib.lines.Line2D([], [], color=fcol[f], marker='o', ls='', label=f)
               for f in fams]
    ax.legend(handles=handles, fontsize=7, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out, '05_rho_dotplot.png'), dpi=150)
    plt.close(fig)

    top = reps_tab.head(8)
    ncol = 4
    nrow = int(np.ceil(len(top) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False,
                            constrained_layout=True)
    runs = df[RUN_COL].astype(str).to_numpy()
    uniq = list(dict.fromkeys(runs))
    color_vals, cb_label, cmap_c = None, None, None
    if len(uniq) == 1:
        for cand in METADATA_COLS + [POS_COL]:
            if cand in df.columns and pd.to_numeric(df[cand], errors='coerce').nunique() > 1:
                color_vals = pd.to_numeric(df[cand], errors='coerce').to_numpy(float)
                cb_label, cmap_c = cand, 'viridis'
                break
    for k, (ax, (_, r)) in enumerate(zip(axs.ravel(), top.iterrows())):
        xv = X[r.feature].to_numpy(float)
        if color_vals is not None:
            sc = ax.scatter(xv, y, c=color_vals, cmap=cmap_c, s=28)
        else:
            for i, u in enumerate(uniq):
                m = runs == u
                ax.scatter(xv[m], y[m], color=plt.get_cmap('tab10')(i % 10), s=28, label=u)
        if np.ptp(xv) > 0:
            cf = np.polyfit(xv, y, 1)
            xs = np.linspace(xv.min(), xv.max(), 20)
            ax.plot(xs, np.polyval(cf, xs), 'k-', lw=0.8)
        ax.set_title(f"{r.feature[:45]}\nrho={r.rho:+.2f}, p_fw={r.p_fwer_reps:.2f}", fontsize=7.5)
        ax.set_ylabel(TARGET, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axs.ravel()[len(top):]:
        ax.axis('off')
    if color_vals is not None:
        fig.colorbar(sc, ax=axs.ravel().tolist(), label=cb_label, shrink=0.6, pad=0.01)
    elif len(uniq) > 1:
        axs[0, 0].legend(fontsize=7, title=RUN_COL)
    plt.savefig(os.path.join(out, '06_scatter_top.png'), dpi=150)
    plt.close(fig)
    return tab


# ------------------------------------------------------- step 4: physics check
def step4(df, y, out):
    cols = [c for c in PHYSICS_COLUMNS if c in df.columns]
    miss = [c for c in PHYSICS_COLUMNS if c not in df.columns]
    if miss:
        say(f"  !! PHYSICS_COLUMNS not in the CSV: {miss}")
    if not cols:
        return
    fig, axs = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.8), squeeze=False)
    for ax, c in zip(axs[0], cols):
        v = pd.to_numeric(df[c], errors='coerce').to_numpy(float)
        m = np.isfinite(v)
        ax.scatter(v[m], y[m], s=28)
        if m.sum() > 3:
            r, p = stats.spearmanr(v[m], y[m])
            ax.set_title(f"{c[-38:]}\nrho={r:+.2f} (p={p:.2g})", fontsize=8)
            say(f"  physics predictor {c}: Spearman rho = {r:+.2f} (p = {p:.3g})")
        ax.set_xlabel(c[-30:], fontsize=7)
        ax.set_ylabel(TARGET)
    plt.tight_layout()
    plt.savefig(os.path.join(out, '04b_physics_check.png'), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ step 5: ML + CV
def make_splits(runs, pos, gap):
    n = len(runs)
    uniq = list(dict.fromkeys(runs))
    splits = []
    if len(uniq) >= 3:
        for u in uniq:
            te = np.where(runs == u)[0]
            tr = np.where(runs != u)[0]
            splits.append((tr, te))
        return splits, 'leave-one-run-out'
    for i in range(n):
        same = runs == runs[i]
        bad = same & (np.abs(pos - pos[i]) <= gap)
        splits.append((np.where(~bad)[0], np.array([i])))
    return splits, f'leave-one-window-out with a buffer of {gap} window(s)'


def _alpha_kw(n):
    """LassoCV / ElasticNetCV: number of alphas ('n_alphas' in older scikit-learn,
    'alphas=<int>' in newer versions)."""
    import inspect
    return ({'n_alphas': n} if 'n_alphas' in inspect.signature(LassoCV).parameters
            else {'alphas': n})


def _inner_cv(ntr):
    return KFold(n_splits=max(2, min(5, ntr // 3)), shuffle=False)


def topk_select(Xtr, ytr, k):
    r = np.nan_to_num(np.abs(spearman_vec(Xtr, ytr)))
    return np.argsort(-r)[:k]


def build_models(meta_idx, pos_idx, k_top, n_trees):
    """name -> callable(Xtr, ytr, Xte) -> predictions."""
    def mean_model(Xtr, ytr, Xte):
        return np.full(len(Xte), ytr.mean())

    def sub_linear(idx):
        def f(Xtr, ytr, Xte):
            if len(idx) == 0:
                return np.full(len(Xte), np.nan)
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
            m.fit(Xtr[:, idx], ytr)
            return m.predict(Xte[:, idx])
        return f

    def lasso(Xtr, ytr, Xte):
        m = make_pipeline(StandardScaler(), LassoCV(cv=_inner_cv(len(ytr)), max_iter=20000,
                                                    **_alpha_kw(30)))
        m.fit(Xtr, ytr)
        return m.predict(Xte)

    def enet(Xtr, ytr, Xte):
        m = make_pipeline(StandardScaler(), ElasticNetCV(
            l1_ratio=[0.3, 0.7, 0.95], cv=_inner_cv(len(ytr)), max_iter=20000, **_alpha_kw(20)))
        m.fit(Xtr, ytr)
        return m.predict(Xte)

    def pls(nc):
        def f(Xtr, ytr, Xte):
            k = max(1, min(nc, Xtr.shape[1], len(ytr) - 2))
            m = make_pipeline(StandardScaler(), PLSRegression(n_components=k, scale=False))
            m.fit(Xtr, ytr)
            return np.ravel(m.predict(Xte))
        return f

    def topk_ols(Xtr, ytr, Xte):
        idx = topk_select(Xtr, ytr, k_top)
        m = make_pipeline(StandardScaler(), LinearRegression())
        m.fit(Xtr[:, idx], ytr)
        return m.predict(Xte[:, idx])

    def rf(Xtr, ytr, Xte):
        m = RandomForestRegressor(n_estimators=n_trees, max_depth=3, min_samples_leaf=3,
                                  max_features=0.5, random_state=0, n_jobs=1)
        m.fit(Xtr, ytr)
        return m.predict(Xte)

    def gbm(Xtr, ytr, Xte):
        m = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05,
                                      subsample=0.7, random_state=0)
        m.fit(Xtr, ytr)
        return m.predict(Xte)

    models = {'mean': mean_model}
    if len(meta_idx):
        models['metadata_only'] = sub_linear(meta_idx)
    if len(pos_idx):
        models['position_only'] = sub_linear(pos_idx)
    models.update({'lasso': lasso, 'elastic_net': enet, 'pls1': pls(1), 'pls2': pls(2),
                   f'top{k_top}_ols': topk_ols, 'random_forest': rf, 'grad_boosting': gbm})
    return models


def oof_predict(fn, X, y, splits):
    pred = np.full(len(y), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for tr, te in splits:
            pred[te] = fn(X[tr], y[tr], X[te])
    return pred


def null_targets(y, runs, kind, n, rng, gap):
    """Permuted targets. 'shuffle' = random permutation; 'shift' = circular
    shift within each run (keeps autocorrelation)."""
    uniq = list(dict.fromkeys(runs))
    out = []
    if kind == 'shuffle':
        for _ in range(n):
            out.append(y[rng.permutation(len(y))])
        return out
    runs_idx = [np.where(runs == u)[0] for u in uniq]
    single = len(runs_idx) == 1
    if single:
        m = len(y)
        offs = [o for o in range(1, m) if min(o, m - o) > gap]
        for o in offs[:n]:
            out.append(np.roll(y, o))
        return out
    for _ in range(n):
        yy = y.copy()
        for ix in runs_idx:
            if len(ix) > 2:
                yy[ix] = np.roll(y[ix], int(rng.integers(1, len(ix))))
        out.append(yy)
    return out


def step5(df, X, y, tab, fam, out, rng, qc):
    n = len(y)
    if n < 8:
        say(f"  only {n} windows - ML skipped (need at least 8).")
        return None
    names = list(X.columns)
    if ML_FEATURES == 'all':
        ml_cols = names
    else:
        ml_cols = [r.feature for _, r in tab[tab.is_representative].iterrows()]
        ml_cols = [c for c in names if c in set(ml_cols)]
    Xm = X[ml_cols].to_numpy(float)
    runs = df[RUN_COL].astype(str).to_numpy()
    pos = df[POS_COL].to_numpy(float)
    k_top = TOP_K or max(1, n // 5)
    splits, cv_name = make_splits(runs, pos, CV_GAP)
    say(f"  ML on {len(ml_cols)} features ({ML_FEATURES}), n = {n}, CV = {cv_name}, "
        f"final model size k = {k_top}")

    # metadata / position columns for the baseline models (added to the matrix)
    meta = [c for c in METADATA_COLS if c in df.columns
            and pd.to_numeric(df[c], errors='coerce').notna().all()
            and pd.to_numeric(df[c], errors='coerce').nunique() > 1]
    pos_ok = pd.Series(pos).nunique() > 1
    extra = [pd.to_numeric(df[c], errors='coerce').to_numpy(float) for c in meta]
    if pos_ok:
        extra.append(pos)
    Xfull = np.column_stack([Xm] + extra) if extra else Xm
    p0 = Xm.shape[1]
    meta_idx = np.arange(p0, p0 + len(meta))
    pos_idx = np.array([p0 + len(meta)]) if pos_ok else np.array([], dtype=int)
    # the ML models only see the first p0 columns
    models = build_models(meta_idx, pos_idx, k_top, qc['rf_trees'])
    feat_models = {'lasso', 'elastic_net', 'pls1', 'pls2', f'top{k_top}_ols',
                   'random_forest', 'grad_boosting'}

    def wrap(name, fn):
        if name in feat_models:
            return lambda Xtr, ytr, Xte: fn(Xtr[:, :p0], ytr, Xte[:, :p0])
        return fn

    models = {k: wrap(k, v) for k, v in models.items()}

    rows, oofs = [], {}
    for name, fn in models.items():
        pred = oof_predict(fn, Xfull, y, splits)
        oofs[name] = pred
        ok = np.isfinite(pred)
        r2 = r2_score_oof(y[ok], pred[ok]) if ok.sum() > 2 else np.nan
        rm = float(np.sqrt(np.mean((y[ok] - pred[ok]) ** 2))) if ok.any() else np.nan
        sp = stats.spearmanr(pred[ok], y[ok])[0] if ok.sum() > 3 and np.ptp(pred[ok]) > 0 else np.nan
        rows.append({'model': name, 'cv_r2': r2, 'cv_rmse': rm, 'cv_spearman': sp})
    res = pd.DataFrame(rows)

    # permutation nulls (shuffle + circular shift)
    def null_r2(name, fn, yp):
        pred = oof_predict(fn, Xfull, yp, splits)
        ok = np.isfinite(pred)
        return r2_score_oof(yp[ok], pred[ok]) if ok.sum() > 2 else np.nan

    trees = {'random_forest', 'grad_boosting'}
    for kind in ('shuffle', 'shift'):
        pvals = []
        for name, fn in models.items():
            if name == 'mean':
                pvals.append(np.nan)
                continue
            nn = qc['perm_trees'] if name in trees else qc['perm_cv']
            yps = null_targets(y, runs, kind, nn, rng, CV_GAP)
            if not yps:
                pvals.append(np.nan)
                continue
            null = np.array(Parallel(n_jobs=N_JOBS)(
                delayed(null_r2)(name, fn, yp) for yp in yps))
            obs = res.loc[res.model == name, 'cv_r2'].iloc[0]
            pvals.append((1 + np.sum(null >= obs)) / (1 + len(null)) if np.isfinite(obs) else np.nan)
        res[f'p_{kind}'] = pvals
    res.to_csv(os.path.join(out, 'cv_results.csv'), index=False)
    say("  cross-validated results (out-of-fold):")
    say(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    base = res.loc[res.model == 'mean', 'cv_r2'].iloc[0]
    cand = res[~res.model.isin(['mean', 'metadata_only', 'position_only'])]
    best = cand.loc[cand.cv_r2.idxmax()] if cand.cv_r2.notna().any() else None

    # --- stability selection (lasso path on random subsamples)
    Xz = StandardScaler().fit_transform(Xm)
    yc = y - y.mean()
    amax = np.max(np.abs(Xz.T @ yc)) / n
    alphas = amax * np.logspace(0, -1.5, 25)
    m_sub = max(5, int(round(STAB_FRAC * n)))

    def one_sub(seed):
        r = np.random.default_rng(seed)
        idx = r.choice(n, m_sub, replace=False)
        Xs = StandardScaler().fit_transform(Xm[idx])
        Xs = np.nan_to_num(Xs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, coefs, _ = lasso_path(Xs, y[idx] - y[idx].mean(), alphas=alphas, max_iter=5000)
        return coefs

    subs = Parallel(n_jobs=N_JOBS)(delayed(one_sub)(int(s)) for s in
                                    rng.integers(0, 2 ** 31 - 1, qc['n_stab']))
    C = np.array(subs)                                  # (n_sub, p, n_alpha)
    freq = (C != 0).mean(axis=0)                        # (p, n_alpha)
    sgn = np.sign(C)
    # Meinshausen-Buehlmann: only use the part of the path where on average at most
    # q features are selected; then E[false selections] <= q^2 / ((2*thr - 1) * p)
    q = STAB_Q or max(3, k_top)
    avg_sel = (C != 0).sum(axis=1).mean(axis=0)         # (n_alpha,) mean #selected
    allowed = np.where(avg_sel <= q)[0]
    if len(allowed) == 0:
        allowed = np.array([0])
    fsub = freq[:, allowed]
    stab = fsub.max(axis=1)
    a_best = allowed[fsub.argmax(axis=1)]
    p_ml = len(ml_cols)
    ev_bound = q ** 2 / ((2 * STAB_THRESHOLD - 1) * p_ml)
    pos_frac = np.array([np.mean(sgn[:, j, a_best[j]][sgn[:, j, a_best[j]] != 0] > 0)
                         if np.any(sgn[:, j, a_best[j]] != 0) else np.nan
                         for j in range(len(ml_cols))])
    stab_tab = pd.DataFrame({'feature': ml_cols, 'family': [fam[c] for c in ml_cols],
                             'stability': stab, 'frac_positive_sign': pos_frac})
    stab_tab = stab_tab.sort_values('stability', ascending=False).reset_index(drop=True)
    stab_tab.to_csv(os.path.join(out, 'stability_selection.csv'), index=False)
    n_stable = int((stab_tab.stability >= STAB_THRESHOLD).sum())
    say(f"  stability selection (path limited to <= {q} features on average): {n_stable} "
        f"feature(s) with selection frequency >= {STAB_THRESHOLD}; expected number of false "
        f"selections <= {ev_bound:.1f}")
    for _, r in stab_tab.head(6).iterrows():
        say(f"    {r.feature:55s} freq={r.stability:.2f}  sign(+)={r.frac_positive_sign:.2f}")
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.28 * min(20, len(stab_tab)) + 1.5)))
    top = stab_tab.head(20)[::-1]
    ax.barh(top.feature, top.stability, color='steelblue')
    ax.axvline(STAB_THRESHOLD, color='crimson', ls='--', label=f'threshold {STAB_THRESHOLD}')
    ax.set_xlabel(f'selection frequency (lasso path limited to <= {q} features on average)')
    ax.tick_params(axis='y', labelsize=7)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, '08_stability_selection.png'), dpi=150)
    plt.close(fig)

    # --- PLS VIP (components chosen by CV between 1 and 2)
    nc = 2 if (res.loc[res.model == 'pls2', 'cv_r2'].iloc[0] >
               res.loc[res.model == 'pls1', 'cv_r2'].iloc[0]) else 1
    nc = max(1, min(nc, Xm.shape[1], n - 2))

    def vip_of(Xa, ya):
        pl = PLSRegression(n_components=nc, scale=False).fit(StandardScaler().fit_transform(Xa), ya)
        Wt = pl.x_weights_
        T = pl.x_scores_
        ssy = (pl.y_loadings_[0] ** 2) * (T ** 2).sum(axis=0)
        wn = (Wt / np.linalg.norm(Wt, axis=0)) ** 2
        return np.sqrt(Xa.shape[1] * (wn @ ssy) / ssy.sum())

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        vip = np.nan_to_num(vip_of(Xm, y))
        bv = []
        for _ in range(qc['n_stab'] // 5):
            idx = rng.choice(n, n, replace=True)
            if np.ptp(y[idx]) > 0:
                bv.append(np.nan_to_num(vip_of(Xm[idx], y[idx])) > 1)
    vip_tab = pd.DataFrame({'feature': ml_cols, 'family': [fam[c] for c in ml_cols],
                            'vip': vip, 'frac_boot_vip_gt1': np.mean(bv, axis=0) if bv else np.nan})
    vip_tab = vip_tab.sort_values('vip', ascending=False)
    vip_tab.to_csv(os.path.join(out, 'pls_vip.csv'), index=False)
    say(f"  PLS ({nc} component(s)): {int((vip > 1).sum())} feature(s) with VIP > 1")

    # --- family importance (permute whole families, honest: on held-out rows)
    fam_list = sorted(set(fam[c] for c in ml_cols))
    fam_cols = {f: [i for i, c in enumerate(ml_cols) if fam[c] == f] for f in fam_list}
    imp_rows = []
    for mname in ('random_forest', 'lasso'):
        fn_fit = {'random_forest': lambda Xa, ya: RandomForestRegressor(
                      n_estimators=qc['rf_trees'], max_depth=3, min_samples_leaf=3,
                      max_features=0.5, random_state=0, n_jobs=1).fit(Xa, ya),
                  'lasso': lambda Xa, ya: make_pipeline(
                      StandardScaler(), LassoCV(cv=_inner_cv(len(ya)), max_iter=20000,
                                                **_alpha_kw(30))).fit(Xa, ya)}[mname]
        delta = {f: [] for f in fam_list}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for tr, te in splits:
                mdl = fn_fit(Xm[tr], y[tr])
                base_err = (mdl.predict(Xm[te]) - y[te]) ** 2
                for f in fam_list:
                    for _ in range(qc['imp_rep']):
                        js = rng.choice(tr, len(te))
                        Xp = Xm[te].copy()
                        Xp[:, fam_cols[f]] = Xm[js][:, fam_cols[f]]
                        delta[f].extend(((mdl.predict(Xp) - y[te]) ** 2 - base_err).tolist())
        vy = float(np.var(y))
        for f in fam_list:
            d = np.array(delta[f])
            imp_rows.append({'model': mname, 'family': f, 'n_features': len(fam_cols[f]),
                             'delta_mse_over_var_y': d.mean() / vy if vy > 0 else np.nan,
                             'sem': d.std(ddof=1) / np.sqrt(len(d)) / vy if len(d) > 1 and vy > 0 else np.nan})
    imp = pd.DataFrame(imp_rows)
    imp.to_csv(os.path.join(out, 'family_importance.csv'), index=False)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(fam_list) + 1.8))
    yy = np.arange(len(fam_list))
    for k, (mname, off) in enumerate((('random_forest', -0.18), ('lasso', 0.18))):
        d = imp[imp.model == mname].set_index('family').reindex(fam_list)
        ax.barh(yy + off, d.delta_mse_over_var_y, height=0.34, xerr=d['sem'],
                label=mname, color=plt.get_cmap('tab10')(k))
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{f} ({len(fam_cols[f])})" for f in fam_list], fontsize=8)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('increase in held-out squared error / var(target) when the family is scrambled')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, '09_family_importance.png'), dpi=150)
    plt.close(fig)
    say("  family importance (RF, increase of held-out error when the family is scrambled):")
    d = imp[imp.model == 'random_forest'].sort_values('delta_mse_over_var_y', ascending=False)
    for _, r in d.head(4).iterrows():
        say(f"    {r.family:20s} {r.delta_mse_over_var_y:+.3f} +/- {r['sem']:.3f}")

    # --- observed vs predicted of the best model
    if best is not None:
        fig, ax = plt.subplots(figsize=(5, 4.6))
        pr = oofs[best.model]
        ax.scatter(y, pr, c=pos, cmap='viridis', s=30)
        lim = [min(y.min(), np.nanmin(pr)), max(y.max(), np.nanmax(pr))]
        ax.plot(lim, lim, 'k--', lw=0.8)
        ax.set_xlabel(f'observed {TARGET}')
        ax.set_ylabel('out-of-fold prediction')
        ax.set_title(f"{best.model}: CV R2 = {best.cv_r2:.2f} (mean baseline {base:.2f})", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out, '07_cv_predictions.png'), dpi=150)
        plt.close(fig)

    # --- final small model: top-k by stability (ties by |rho|)
    rho_map = dict(zip(tab.feature, tab.rho))
    order = stab_tab.assign(arho=stab_tab.feature.map(lambda c: abs(rho_map.get(c, 0)))) \
                    .sort_values(['stability', 'arho'], ascending=False)
    chosen = order.feature.head(k_top).tolist()
    Xs_ = StandardScaler().fit_transform(X[chosen].to_numpy(float))
    ols = LinearRegression().fit(Xs_, y)
    pred_in = ols.predict(Xs_)
    r2_in = r2_score_oof(y, pred_in)
    cbs = []
    for _ in range(qc['n_boot'] // 4):
        idx = rng.integers(0, n, n)
        if np.ptp(y[idx]) > 0:
            Xb = StandardScaler().fit_transform(X[chosen].to_numpy(float)[idx])
            cbs.append(LinearRegression().fit(Xb, y[idx]).coef_)
    cbs = np.array(cbs)
    fin = pd.DataFrame({'feature': chosen,
                        'standardised_coef': ols.coef_,
                        'ci_lo': np.percentile(cbs, 2.5, axis=0),
                        'ci_hi': np.percentile(cbs, 97.5, axis=0),
                        'stability': [float(stab_tab.set_index('feature').loc[c, 'stability'])
                                      for c in chosen]})
    fin['in_sample_r2'] = r2_in
    topk_name = f'top{k_top}_ols'
    fin['honest_cv_r2_(selection_inside_folds)'] = float(
        res.loc[res.model == topk_name, 'cv_r2'].iloc[0])
    fin.to_csv(os.path.join(out, 'final_model.csv'), index=False)
    say(f"  final model ({k_top} feature(s) by stability): in-sample R2 = {r2_in:.2f} "
        f"(optimistic: features were chosen with the same data); "
        f"honest CV R2 of the same workflow = {fin.iloc[0, -1]:.2f}")
    for _, r in fin.iterrows():
        say(f"    {r.feature:55s} coef={r.standardised_coef:+.3f} "
            f"[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]  stability={r.stability:.2f}")
    return {'res': res, 'stab': stab_tab, 'vip': vip_tab, 'best': best, 'base': base,
            'imp': imp, 'k_top': k_top}


# ------------------------------------------------------ combined table / verdict
def combine(tab, ml, out):
    t = tab.copy()
    if ml is not None:
        t = t.merge(ml['stab'][['feature', 'stability', 'frac_positive_sign']],
                    on='feature', how='left')
        t = t.merge(ml['vip'][['feature', 'vip', 'frac_boot_vip_gt1']], on='feature', how='left')
    else:
        t['stability'] = np.nan
        t['vip'] = np.nan
    t['vote_fwer'] = (t.p_fwer_reps < 0.10) & t.is_representative
    t['vote_partial'] = t.p_partial < 0.05
    t['vote_stability'] = t.stability >= STAB_THRESHOLD
    t['vote_vip'] = t.vip >= 1.0
    t['votes'] = t[['vote_fwer', 'vote_partial', 'vote_stability', 'vote_vip']].sum(axis=1)
    t['abs_rho'] = t.rho.abs()
    t = t.sort_values(['votes', 'abs_rho'], ascending=False).drop(columns='abs_rho')
    t.to_csv(os.path.join(out, 'feature_summary.csv'), index=False)
    return t


def verdict(s1, ml, summ):
    say("\n=== SUMMARY ===")
    say(f"Signal check: {s1['verdict']}")
    if ml is None:
        say("ML: skipped (too few windows).")
    else:
        res, base = ml['res'], ml['base']
        b = ml['best']
        if b is not None:
            ps = b.get('p_shift', np.nan)
            say(f"Best feature model: {b.model}, CV R2 = {b.cv_r2:.2f} (mean baseline {base:.2f}); "
                f"p(shuffle) = {b.get('p_shuffle', np.nan):.3f}, p(shift) = {ps:.3f}")
            refs = {}
            for nm in ('mean', 'metadata_only', 'position_only'):
                v = res.loc[res.model == nm, 'cv_r2']
                if len(v) and np.isfinite(v.iloc[0]):
                    refs[nm] = float(v.iloc[0])
            ref_name = max(refs, key=refs.get)
            ref = refs[ref_name]
            say(f"  baselines (CV R2): " + ", ".join(f"{k} {v:.2f}" for k, v in refs.items()))
            if ref_name != 'mean' and ref > 0.1:
                say(f"  note: {ref_name.replace('_', ' ')} alone gives CV R2 = {ref:.2f}; "
                    f"window position / metadata already explain much of the target, so features "
                    f"that merely track them are not independent evidence.")
            if np.isfinite(ps) and ps < 0.1 and b.cv_r2 > ref + 0.05:
                say("  -> The model beats ALL baselines (mean, metadata-only, position-only) and "
                    "the shift-permutation null; the features with the most votes below are "
                    "the candidates.")
            elif b.cv_r2 > 0.1 and ref_name != 'mean' and b.cv_r2 <= ref + 0.05:
                say(f"  -> The features do not improve on {ref_name.replace('_', ' ')} alone: "
                    f"the apparent signal is drift / metadata, not an independent feature effect.")
            else:
                say("  -> No convincing predictive signal beyond the baselines / permutation null "
                    "with this many windows. Do not interpret feature rankings as established.")
    top = summ.head(5)
    say("Features with the most agreeing criteria (votes: family-wise p<0.1, partial p<0.05, "
        "stability, VIP>1):")
    for _, r in top.iterrows():
        say(f"  {int(r.votes)} votes  {r.feature:55s} [{r.family}] rho={r.rho:+.2f}")


# ------------------------------------------------------------------------ main
def main(argv):
    global CSV_PATH, OUT_DIR
    args = [a for a in argv[1:] if not a.startswith('--')]
    quick = '--quick' in argv
    if len(args) >= 1:
        CSV_PATH = args[0]
    if len(args) >= 2:
        OUT_DIR = args[1]
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    qc = {'n_boot': N_BOOT, 'n_perm': N_PERM, 'n_stab': N_STAB, 'perm_cv': PERM_CV,
          'perm_trees': PERM_CV_TREES, 'rf_trees': 200, 'imp_rep': 5}
    if quick:
        qc.update(n_boot=300, n_perm=2000, n_stab=100, perm_cv=15, perm_trees=6,
                  rf_trees=60, imp_rep=2)

    df = load(CSV_PATH)
    say(f"Loaded {CSV_PATH}: {len(df)} windows, {df[RUN_COL].nunique()} run(s), "
        f"{df.shape[1]} columns")
    y = df[TARGET].to_numpy(float)

    say("\n[1] Signal check")
    s1 = step1(df, OUT_DIR)

    say("\n[2] Feature selection and clustering")
    X = select_features(df)
    if X.shape[1] == 0:
        sys.exit("no usable feature columns")
    fam = {c: family_of(c) for c in X.columns}
    lab, reps, link, C = cluster_features(X, N_CLUSTERS)
    plot_heatmap(X, lab, reps, link, C, fam, OUT_DIR)
    pd.DataFrame({'feature': X.columns, 'family': [fam[c] for c in X.columns],
                  'cluster': lab, 'is_representative': [i in set(reps.tolist())
                                                        for i in range(X.shape[1])]}
                 ).to_csv(os.path.join(OUT_DIR, 'features_kept.csv'), index=False)
    fc = pd.Series(fam).value_counts()
    say("  families: " + ", ".join(f"{k} {v}" for k, v in fc.items()))
    say(f"  {len(reps)} clusters -> {len(reps)} representatives")
    leak = [c for c in X.columns if re.search(r'(^|__)O_over_Si|N_O|N_Si', c)]
    if leak:
        say(f"  !! possible label leakage (O/Si-type columns): {leak}")

    say("\n[3] Univariate screening")
    tab = step3(df, X, y, lab, reps, fam, OUT_DIR, rng, qc)
    top_rho = tab.rho.abs().max()
    if top_rho > 0.97:
        say(f"  !! a feature has |rho| = {top_rho:.2f} with the target: check for leakage")

    say("\n[4] Physics baseline")
    step4(df, y, OUT_DIR)

    say("\n[5] ML with honest cross-validation")
    ml = step5(df, X, y, tab, fam, OUT_DIR, rng, qc)

    summ = combine(tab, ml, OUT_DIR)
    verdict(s1, ml, summ)
    with open(os.path.join(OUT_DIR, 'summary.txt'), 'w') as fh:
        fh.write("\n".join(LOG))
    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == '__main__':
    main(sys.argv)

#!/usr/bin/env python3
"""
Measurement noise and initial values of the ZL2030 data: the evidence behind the in-silico error model
====================================================================================================

Reads the raw ZL2030 measurement files in experiments/ (not the processed data frames, which average replicates)
and estimates, for each data type, the quantities the in-silico noise model of the paper is built on:

  plate counts  replicate plates of one subsample (technical) and subsamples of one condition (sampling):
                pooled standard deviations of log10 CFU, their dependence on the count magnitude and medium,
                normality of the log10 residuals  -> Gaussian noise on log10 with sigma = 0.15 (baseline)
  MALDI         colonies identified per plate (N), replicate plates of one condition: Pearson dispersion and
                Dirichlet-multinomial concentration, missed species vs. share -> Multinomial(N = 48)
  NGS           reads per sample, 15 replicate pairs: dispersion relative to multinomial sampling, Dirichlet-
                multinomial concentration A (moment and maximum likelihood, by depth), relative error vs. share,
                zeros -> Dirichlet-multinomial(N_R = 56 000, A = 2000 baseline, 300 high, 5000 low)
  x0            the 180 log10 initial values of the frozen ZL2030 calibration (12 species x 15 experiments):
                normal fit, normality tests, comparison with the newer fits and with the day-0 plate counts
                -> log10 x0 ~ N(2.9, 0.69^2)

Only aggregated statistics and figures are written; no measurement values leave this repository.
Run from the root of this repository:
    python scripts/zl2030_noise_analysis.py [--out out/noise_analysis/] [--copy-to ../fusion_model_paper/report/figures/]
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# taxa of the ZL2030 calibration (11 prevalent genera + the remainder), as in out/dm_fit/Result_calibration_zl2030_dm.json
MODEL_TAXA = ['Acinetobacter', 'Brochothrix', 'Carnobacterium', 'Chryseobacterium', 'Corynebacterium', 'Kocuria',
              'Lactobacillus', 'Lactococcus', 'Leuconostoc', 'Pseudomonas', 'Psychrobacter']
NGS_DEPTH_SPLIT = 26000      # pairs with both samples at least this deep vs. shallower pairs
NGS_DEPTH_INSILICO = 56000   # N_R of the in-silico data
NGS_A_SCENARIOS = {'low': 5000., 'baseline': 2000., 'high': 300.}
MALDI_N_SCENARIOS = {'low': 96, 'baseline': 48, 'high': 24}
PC_SIGMA_SCENARIOS = {'low': 0.10, 'baseline': 0.15, 'high': 0.30}
FROZEN_X0_FILE = os.path.join(ROOT, '..', 'fusion_model_paper', 'model_paper', 'setup', 'Result_calibration_zl2030_copy.json')
# the x0 bounds of these fits are (0, 5)
NEWER_FITS = {'multistart_logk': 'out/multistart_logk/Result_calibration_zl2030_multistart.json',
              'refine_corrected': 'out/refine_corrected/Result_calibration_zl2030_refine.json',
              'refine_lamexp05': 'out/refine_lamexp05/Result_calibration_zl2030_refine.json',
              'dm_fit': 'out/dm_fit/Result_calibration_zl2030_dm.json',
              'dm_fit2': 'out/dm_fit2/Result_calibration_zl2030_dm.json'}

# colours (validated categorical palette, light surface)
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'magenta': '#e87ba4',
     'violet': '#6250d6', 'grey': '#52514e', 'light': '#c3c2b7'}
MEDIUM_COLOUR = {'PC': C['blue'], 'MRS': C['orange'], 'VRBD': C['aqua']}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
                     'legend.frameon': False, 'figure.dpi': 110, 'savefig.dpi': 200})


def pooled_sd(sds, ns):
    """Pooled standard deviation of groups with sample SDs sds (ddof=1) and sizes ns."""
    sds, ns = np.asarray(sds, float), np.asarray(ns, float)
    use = ns > 1
    return float(np.sqrt(np.sum((ns[use] - 1) * sds[use]**2) / np.sum(ns[use] - 1))) if use.any() else np.nan


def short_bins(labels):
    """'(0.001, 0.01]' -> '0.001-0.01'."""
    return [re.sub(r'[\(\]\s]', '', str(b)).replace(',', '-') for b in labels]


def savefig(fig, out, name):
    fig.savefig(os.path.join(out, name + '.png'), bbox_inches='tight')
    fig.savefig(os.path.join(out, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


# ======================================================================================================================
# Plate counts
# ======================================================================================================================
def read_platecount_rows():
    """One row per raw table line: a subsample on one medium with its replicate plates (CFU/g)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'experiments', 'microbiology', '*.xlsx'))):
        df = pd.read_excel(f)
        name_col = df.columns[0]
        rep_cols = [c for c in df.columns if str(c).startswith('Replicate')]
        for _, r in df.iterrows():
            name = r[name_col]
            if not isinstance(name, str) or not re.match(r'^[VP]\d\d_', name.strip()):
                continue
            vals = pd.to_numeric(r[rep_cols], errors='coerce').dropna().values.astype(float)
            n_zero = int(np.sum(vals <= 0))
            vals = vals[vals > 0]
            fspl = name.strip().split('_')
            if len(fspl) < 8:
                continue
            rows.append(dict(exp=fspl[0], variant='-'.join(fspl[3:-5]), atm=fspl[2], day=fspl[-4], batch=fspl[-3],
                             subsample=fspl[-2], medium=re.sub(r'\d+$', '', fspl[-1]), n=len(vals), n_zero=n_zero,
                             log10=np.log10(vals) if len(vals) else np.array([]),
                             sd_log10=float(np.std(np.log10(vals), ddof=1)) if len(vals) > 1 else np.nan,
                             cv=float(np.std(vals, ddof=1) / np.mean(vals)) if len(vals) > 1 else np.nan,
                             mean_log10=float(np.mean(np.log10(vals))) if len(vals) else np.nan))
    return pd.DataFrame(rows)


def analyse_platecount(out):
    df = read_platecount_rows()
    df = df[df.n > 0]
    res = {'n_rows': int(len(df)), 'n_plates': int(df.n.sum()), 'n_experiments': int(df.exp.nunique()),
           'media': sorted(df.medium.unique().tolist()), 'n_zero_plates': int(df.n_zero.sum())}
    # --- technical: replicate plates of one subsample -----------------------------------------------------------------
    tech = df[df.n > 1]
    res['technical'] = {'n_subsamples': int(len(tech)), 'n_plates': int(tech.n.sum()),
                        'sd_log10_pooled': pooled_sd(tech.sd_log10, tech.n), 'cv_pooled': pooled_sd(tech.cv, tech.n),
                        'per_medium': {m: {'n_subsamples': int(len(g)), 'sd_log10_pooled': pooled_sd(g.sd_log10, g.n),
                                           'cv_pooled': pooled_sd(g.cv, g.n)} for m, g in tech.groupby('medium')}}
    # magnitude dependence
    tech = tech.assign(mag=np.floor(tech.mean_log10).astype(int))
    mag_tab = []
    for (m, b), g in tech.groupby(['medium', 'mag']):
        if len(g) >= 5:
            mag_tab.append(dict(medium=m, log10_bin=f'{b}-{b+1}', n_subsamples=len(g), sd_log10_pooled=pooled_sd(g.sd_log10, g.n),
                                cv_pooled=pooled_sd(g.cv, g.n)))
    mag_tab = pd.DataFrame(mag_tab)
    mag_tab.to_csv(os.path.join(out, 'platecount_sd_by_magnitude.csv'), index=False)
    # standardized residuals of the plates about their subsample mean (expectation-corrected for the group size)
    resid = np.concatenate([(r.log10 - r.log10.mean()) * np.sqrt(r.n / (r.n - 1)) for r in tech.itertuples()])
    s_t = res['technical']['sd_log10_pooled']
    z = resid / s_t
    res['technical']['residuals'] = {'n': int(len(z)), 'kurtosis_excess': float(stats.kurtosis(z)), 'skew': float(stats.skew(z)),
                                     'frac_abs_gt_2sd': float(np.mean(np.abs(z) > 2)), 'frac_abs_gt_3sd': float(np.mean(np.abs(z) > 3)),
                                     'shapiro_p': float(stats.shapiro(z[:5000]).pvalue) if len(z) > 3 else np.nan}
    # --- sampling: subsamples (meat pieces) of one condition --------------------------------------------------------------
    cond = df.groupby(['exp', 'variant', 'atm', 'day', 'medium'])
    sub = cond.agg(n_sub=('mean_log10', 'size'), sd_sub=('mean_log10', lambda x: np.std(x, ddof=1) if len(x) > 1 else np.nan),
                   n_plates=('n', 'sum')).reset_index()
    sub2 = sub[sub.n_sub > 1]
    res['sampling'] = {'n_conditions': int(len(sub2)), 'n_subsamples': int(sub2.n_sub.sum()),
                       'sd_log10_pooled': pooled_sd(sub2.sd_sub, sub2.n_sub),
                       'per_medium': {m: {'n_conditions': int(len(g)), 'sd_log10_pooled': pooled_sd(g.sd_sub, g.n_sub)}
                                      for m, g in sub2.groupby('medium')},
                       'per_day': {d: {'n_conditions': int(len(g)), 'sd_log10_pooled': pooled_sd(g.sd_sub, g.n_sub)}
                                   for d, g in sub2.groupby(sub2.day.eq('00').map({True: 'day 0', False: 'later days'}))}}
    # the subsample means contain the technical error of the mean of n plates: remove it to get the pure sampling part
    n_bar = float(tech.n.mean())
    s_s = res['sampling']['sd_log10_pooled']
    res['sampling']['sd_log10_pure_sampling'] = float(np.sqrt(max(s_s**2 - s_t**2 / n_bar, 0.)))
    res['sampling']['sd_log10_single_plate_total'] = float(np.sqrt(res['sampling']['sd_log10_pure_sampling']**2 + s_t**2))
    res['scenarios_log10_sigma'] = PC_SIGMA_SCENARIOS
    res['relative_10pct_in_log10'] = float(0.1 / np.log(10.))
    # --- day-0 totals across experiments (for the x0 discussion) --------------------------------------------------------------
    d0 = df[(df.day == '00') & (df.variant == '')].groupby(['exp', 'medium']).mean_log10.mean().reset_index()
    res['day0_totals'] = {m: {'n_experiments': int(len(g)), 'mean_log10': float(g.mean_log10.mean()),
                              'sd_log10_between_experiments': float(g.mean_log10.std(ddof=1)),
                              'min_log10': float(g.mean_log10.min()), 'max_log10': float(g.mean_log10.max())}
                          for m, g in d0.groupby('medium')}
    # --- figure --------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for m, g in mag_tab.groupby('medium'):
        x = [int(b.split('-')[0]) + 0.5 for b in g.log10_bin]
        ax[0].plot(x, g.sd_log10_pooled, 'o-', color=MEDIUM_COLOUR.get(m, C['grey']), lw=2, ms=6, label=m)
    ax[0].axhline(s_t, color=C['grey'], lw=1, ls='--')
    ax[0].text(ax[0].get_xlim()[1], s_t, f' pooled {s_t:.2f}', va='center', fontsize=8, color=C['grey'])
    ax[0].axhline(0.1 / np.log(10), color=C['light'], lw=1, ls=':')
    ax[0].text(ax[0].get_xlim()[0], 0.1 / np.log(10) + 0.004, '10 % relative noise = 0.04', fontsize=8, color=C['grey'])
    ax[0].set_xlabel('log10 CFU/g of the subsample'); ax[0].set_ylabel('SD of log10 between replicate plates')
    ax[0].set_title('a  Replicate SD vs. count level'); ax[0].legend(title='medium', loc='upper right')
    ax[0].set_ylim(0, max(0.3, mag_tab.sd_log10_pooled.max() * 1.15))
    xs = np.linspace(-4, 4, 200)
    ax[1].hist(z, bins=np.arange(-4.25, 4.5, 0.5), density=True, color=C['blue'], alpha=0.85, edgecolor='white')
    ax[1].plot(xs, stats.norm.pdf(xs), color=C['orange'], lw=2, label='standard normal')
    ax[1].set_xlabel('replicate residual of log10 CFU / pooled SD'); ax[1].set_ylabel('density'); ax[1].legend()
    ax[1].set_title(f'b  Standardized residuals (n = {len(z)})')
    (osm, osr), _ = stats.probplot(z, dist='norm')
    ax[2].plot(osm, osr, '.', color=C['blue'], ms=4, alpha=0.6)
    ax[2].plot([-4, 4], [-4, 4], color=C['orange'], lw=1.5)
    ax[2].set_xlabel('normal quantile'); ax[2].set_ylabel('observed quantile'); ax[2].set_title('c  Normal Q-Q plot')
    ax[2].set_xlim(-4, 4); ax[2].set_ylim(-5, 5)
    savefig(fig, out, 'zl2030_platecount_noise')
    # components figure
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    labels = ['replicate plates\n(technical)', 'subsamples of\none condition', 'single plate,\ntotal', 'in-silico\nbaseline']
    vals = [s_t, s_s, res['sampling']['sd_log10_single_plate_total'], 0.15]
    cols = [C['blue'], C['blue'], C['blue'], C['orange']]
    ax.bar(labels, vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f'{v:.3f}', ha='center', fontsize=8)
    ax.axhline(0.1 / np.log(10), color=C['grey'], lw=1, ls=':'); ax.text(3.4, 0.1 / np.log(10) + 0.003, '10 % relative', fontsize=8, ha='right', color=C['grey'])
    ax.set_ylabel('SD of log10 CFU/g'); ax.set_title('Plate counts: components of the error (ZL2030)'); ax.set_ylim(0, 0.22)
    savefig(fig, out, 'zl2030_platecount_components')
    return res


# ======================================================================================================================
# MALDI
# ======================================================================================================================
def read_maldi_plates():
    """{plate name: Series of identified colonies per genus} from the raw MALDI tables."""
    plates, dup = {}, 0
    for f in sorted(glob.glob(os.path.join(ROOT, 'experiments', 'MALDI', '*.xlsx'))):
        raw = pd.read_excel(f, header=None)
        is_name = raw.map(lambda c: isinstance(c, str) and re.match(r'^V\d\d_', c.strip()) is not None)
        hdr = int(np.argmax(is_name.sum(axis=1).values > 0))
        header = raw.iloc[hdr].values
        body = raw.iloc[hdr + 1:]
        taxa = body.iloc[:, 0]
        body = body[taxa.notna()]
        genus = body.iloc[:, 0].astype(str).str.strip().str.split().str[0].replace({'Leucononstoc': 'Leuconostoc'}).values
        for j, name in enumerate(header):
            if not (isinstance(name, str) and re.match(r'^V\d\d_', name.strip())):
                continue
            counts = pd.to_numeric(body.iloc[:, j], errors='coerce').fillna(0.).values
            s = pd.Series(counts).groupby(genus).sum()
            key = re.sub(r'_IDENExp\d+$', '', name.strip())
            if key in plates:
                dup += 1
                continue
            plates[key] = s
    return plates, dup


def dm_loglik(K, N, A, p):
    """log Dirichlet-multinomial likelihood of count rows K (plates x categories) with totals N and alpha = A p,
    without the multinomial coefficient. Categories with p = 0 and k = 0 contribute 0."""
    a = A * np.asarray(p, float)
    use = a > 0
    K = np.asarray(K, float)[:, use]
    a = a[use]
    return float(np.sum(gammaln(A) - gammaln(N + A)) + np.sum(gammaln(K + a) - gammaln(a)))


def dm_fit_A(groups, bounds=(-2., 7.)):
    """Maximum-likelihood concentration A (plug-in pooled proportions per group) and 95 % profile interval.
    groups: list of (K, N) with K plates x categories, N totals."""
    def neg2(la):
        A = 10.**la
        return -2. * sum(dm_loglik(K, N, A, K.sum(0) / K.sum()) for K, N in groups)
    r = minimize_scalar(neg2, bounds=bounds, method='bounded', options={'xatol': 1e-4})
    best, f0 = r.x, r.fun
    ci = []
    for lo, hi in ((bounds[0], best), (best, bounds[1])):
        g = lambda la: neg2(la) - f0 - stats.chi2.ppf(0.95, 1)
        try:
            ci.append(10.**brentq(g, lo, hi))
        except ValueError:
            ci.append(10.**(lo if lo != best else hi))
    return float(10.**best), tuple(ci)


def pearson_group(K):
    """Pearson chi-square of a plates x categories count table against the pooled proportions, its dof and the moment
    coefficients (a, b) with E[X^2 / dof] = (a + b A) / (1 + A) under a Dirichlet-multinomial with concentration A."""
    K = np.asarray(K, float)
    N = K.sum(1)
    p = K.sum(0) / K.sum()
    use = p > 0
    E = np.outer(N, p[use])
    X2 = float(np.sum((K[:, use] - E)**2 / E))
    dof = (len(N) - 1) * (use.sum() - 1)
    Nt = N.sum()
    a = b = 0.
    for i in range(len(N)):
        others = np.delete(N, i)
        a += (1 - N[i] / Nt)**2 * N[i] + N[i] / Nt**2 * np.sum(others**2)
        b += (1 - N[i] / Nt)**2 + N[i] / Nt**2 * np.sum(others)
    return X2, int(dof), a / (len(N) - 1), b / (len(N) - 1)


def moment_A(stats_list):
    """Moment estimate of A from [(X2, dof, a, b)]: sum X2 = sum dof (a + b A) / (1 + A)."""
    X2 = sum(s[0] for s in stats_list); D = sum(s[1] for s in stats_list)
    Da = sum(s[1] * s[2] for s in stats_list); Db = sum(s[1] * s[3] for s in stats_list)
    phi = X2 / D
    A = (Da - X2) / (X2 - Db) if X2 > Db else np.inf
    return float(phi), float(A)


def analyse_maldi(out):
    plates, dup = read_maldi_plates()
    meta = []
    for name, s in plates.items():
        fspl = name.split('_')
        meta.append(dict(name=name, exp=fspl[0], atm=fspl[2], variant='-'.join(fspl[3:-5]), temp=fspl[-5], day=fspl[-4],
                         batch=fspl[-3], subsample=fspl[-2], medium=re.sub(r'\d+$', '', fspl[-1]), N=float(s.sum())))
    meta = pd.DataFrame(meta)
    meta = meta[meta.N > 0]
    N = meta.N.values
    res = {'n_plates': int(len(meta)), 'n_duplicate_columns_skipped': dup, 'n_experiments': int(meta.exp.nunique()),
           'N_colonies': {'median': float(np.median(N)), 'mode': int(stats.mode(N, keepdims=False).mode), 'mean': float(N.mean()),
                          'q25': float(np.percentile(N, 25)), 'q75': float(np.percentile(N, 75)), 'min': float(N.min()), 'max': float(N.max()),
                          'frac_le_24': float(np.mean(N <= 24)), 'frac_ge_48': float(np.mean(N >= 48)), 'frac_ge_40': float(np.mean(N >= 40))},
           'N_per_medium': {m: {'n_plates': int(len(g)), 'median': float(g.N.median()), 'q25': float(g.N.quantile(.25)), 'q75': float(g.N.quantile(.75))}
                            for m, g in meta.groupby('medium')}}
    # --- replicate plates of one condition --------------------------------------------------------------------------------
    key = ['exp', 'variant', 'atm', 'temp', 'day', 'medium']
    groups, grp_meta, pstats, points, zero_rows = [], [], [], [], []
    for k, g in meta.groupby(key):
        if len(g) < 2:
            continue
        K = pd.DataFrame({n: plates[n] for n in g.name}).fillna(0.).T.values  # plates x genera
        K = K[:, K.sum(0) > 0]
        if K.shape[1] < 2:
            continue
        Ng = K.sum(1)
        groups.append((K, Ng)); grp_meta.append(dict(zip(key, k), n_plates=len(g)))
        pstats.append(pearson_group(K))
        p = K.sum(0) / K.sum()
        phat = K / Ng[:, None]
        for j in range(K.shape[1]):
            points.append(dict(share=p[j], rel_sd=np.std(phat[:, j], ddof=1) / p[j], n_plates=len(g), N_mean=Ng.mean(), medium=k[-1], day=k[-2]))
            for i in range(len(Ng)):
                zero_rows.append(dict(share=p[j], zero=float(K[i, j] == 0), expected=(1 - p[j])**Ng[i], N=Ng[i]))
    gm = pd.DataFrame(grp_meta)
    phi, A_mom = moment_A(pstats)
    A_ml, A_ci = dm_fit_A(groups)
    N_mean = float(np.mean([N_.mean() for _, N_ in groups]))
    res['replicate_groups'] = {'n_groups': int(len(groups)), 'n_plates': int(gm.n_plates.sum()),
                               'groups_day0': int((gm.day == '00').sum()), 'groups_later_days': int((gm.day != '00').sum()),
                               'per_medium': gm.groupby('medium').size().to_dict(), 'N_mean': N_mean,
                               'pearson_dispersion_phi': phi, 'N_eff_at_48': float(48 / phi), 'A_moment': A_mom,
                               'A_mle': A_ml, 'A_mle_ci95': list(A_ci),
                               'rel_sd_factor_vs_multinomial': float(np.sqrt(phi)),
                               'note': 'groups are plates of one experiment, atmosphere, day and medium that differ in meat subsample '
                                       '(CC01..CC05) or batch; day-0 groups dominate, so the dispersion includes true differences between subsamples'}
    # by day class
    for lab, sel in (('day0', gm.day == '00'), ('later', gm.day != '00')):
        sub = [s for s, u in zip(pstats, sel) if u]
        if sub:
            ph, am = moment_A(sub)
            res['replicate_groups'][f'phi_{lab}'] = ph
            res['replicate_groups'][f'A_moment_{lab}'] = am
    # zeros vs share
    zr = pd.DataFrame(zero_rows)
    bins = [0, 0.01, 0.03, 0.1, 0.3, 1.0]
    zr['bin'] = pd.cut(zr.share, bins)
    ztab = zr.groupby('bin', observed=True).agg(n=('zero', 'size'), observed_zero_frac=('zero', 'mean'), multinomial_expected=('expected', 'mean')).reset_index()
    ztab['bin'] = ztab['bin'].astype(str)
    ztab.to_csv(os.path.join(out, 'maldi_zeros_by_share.csv'), index=False)
    res['zeros_by_share'] = ztab.to_dict(orient='records')
    pts = pd.DataFrame(points)
    pts['bin'] = pd.cut(pts.share, bins)
    rtab = pts.groupby('bin', observed=True).agg(n=('rel_sd', 'size'), rms_rel_sd=('rel_sd', lambda x: np.sqrt(np.mean(x**2))),
                                                 mean_share=('share', 'mean'), N_mean=('N_mean', 'mean')).reset_index()
    rtab['multinomial_rel_sd'] = np.sqrt((1 - rtab.mean_share) / (rtab.N_mean * rtab.mean_share))
    rtab['ratio'] = rtab.rms_rel_sd / rtab.multinomial_rel_sd
    rtab['bin'] = rtab['bin'].astype(str)
    rtab.to_csv(os.path.join(out, 'maldi_relative_sd_by_share.csv'), index=False)
    res['relative_sd_by_share'] = rtab.to_dict(orient='records')
    # --- figure --------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].hist(N, bins=np.arange(0.5, N.max() + 1.5, 2), color=C['blue'], edgecolor='white')
    for lab, n_ in MALDI_N_SCENARIOS.items():
        ax[0].axvline(n_, color=C['orange'], lw=1.2, ls='--')
        ax[0].text(n_, ax[0].get_ylim()[1] * 0.95, f' {lab}\n N={n_}', fontsize=7.5, color=C['orange'], va='top')
    ax[0].set_xlabel('colonies identified per plate N'); ax[0].set_ylabel('plates')
    ax[0].set_title(f'a  Colonies per plate (median {np.median(N):.0f})')
    pp = np.logspace(-2.3, 0, 200)
    ax[1].loglog(pts.share, pts.rel_sd, '.', color=C['blue'], ms=5, alpha=0.5, label='replicate plates (ZL2030)')
    ax[1].loglog(pp, np.sqrt((1 - pp) / (N_mean * pp)), color=C['orange'], lw=2, label=f'multinomial, N = {N_mean:.0f}')
    ax[1].loglog(pp, np.sqrt(phi) * np.sqrt((1 - pp) / (N_mean * pp)), color=C['aqua'], lw=2, ls='--', label=f'x sqrt(phi) = {np.sqrt(phi):.2f}')
    ax[1].axhline(0.1, color=C['grey'], lw=1, ls=':'); ax[1].text(0.012, 0.11, '10 % constant relative noise', fontsize=7.5, color=C['grey'])
    ax[1].set_xlabel('share of the species on the plate (pooled)'); ax[1].set_ylabel('relative SD between replicate plates')
    ax[1].set_title('b  Relative SD between replicate plates'); ax[1].legend(fontsize=7.5, loc='lower left'); ax[1].set_ylim(0.03, 3)
    x = np.arange(len(ztab))
    ax[2].bar(x - 0.2, ztab.observed_zero_frac, 0.4, color=C['blue'], label='observed')
    ax[2].bar(x + 0.2, ztab.multinomial_expected, 0.4, color=C['orange'], label='multinomial (1-p)^N')
    ax[2].set_xticks(x); ax[2].set_xticklabels(short_bins(ztab['bin']), fontsize=7.5); ax[2].set_xlabel('share of the species (pooled)')
    ax[2].set_ylabel('fraction of plates where the species is absent'); ax[2].legend(); ax[2].set_title('c  Species absent from a plate')
    savefig(fig, out, 'zl2030_maldi_noise')
    return res


# ======================================================================================================================
# NGS
# ======================================================================================================================
def read_ngs_table(path):
    """(counts: genus x sample DataFrame, meta: sample x header-label DataFrame) of one ZOTU table, counts summed per genus."""
    with open(path, encoding='ISO-8859-1') as fh:
        rows = [l.rstrip('\n').split('\t') for l in fh]
    hdr = next(i for i, r in enumerate(rows) if any('#taxonomy' in c for c in r))
    hrow = rows[hdr]
    tax_col = next(j for j, c in enumerate(hrow) if '#taxonomy' in c)
    first = 2 if tax_col == 1 else 3
    id_cols = [j for j in range(first, len(hrow)) if hrow[j].strip()]
    ids = [hrow[j].strip() for j in id_cols]
    meta = {}
    for r in rows[:hdr]:
        label = next((c.strip() for c in reversed(r[:min(id_cols)]) if c.strip()), None)
        if label:
            meta[label] = [r[j].strip() if j < len(r) else '' for j in id_cols]
    meta = pd.DataFrame(meta, index=ids)
    data = {}
    for r in rows[hdr + 1:]:
        if len(r) <= tax_col or ';' not in r[tax_col]:
            continue
        tax = r[tax_col].replace('"', '')
        parts = tax.split(';')
        if parts[0] != 'Bacteria':
            continue
        genus = parts[-2] if len(parts) >= 2 else ''
        if not genus:
            continue
        vals = np.array([float(r[j]) if j < len(r) and r[j].strip() else 0. for j in id_cols])
        data[genus] = data.get(genus, 0.) + vals
    counts = pd.DataFrame(data, index=ids).T
    return counts, meta


def analyse_ngs(out):
    tables = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'experiments', 'NGS', 'ZOTU_Table_V*.tsv'))):
        exp = os.path.basename(f)[11:14]
        tables[exp] = read_ngs_table(f)
    depth = pd.concat([c.sum() for c, _ in tables.values()])
    header_tot, calib_ids = [], []
    for exp, (c, m) in tables.items():
        for lab in ('Gesamtreads/Probe',):
            if lab in m.columns:
                header_tot += [float(v) for v in m[lab] if str(v).replace('.', '').isdigit()]
        atm = m['Atmosphäre'] if 'Atmosphäre' in m.columns else m.get('atmosphere', pd.Series('', index=m.index))
        is_o2 = atm.astype(str).str.contains('O2')
        variant = pd.Series(False, index=m.index)
        for lab, bad in (('Kühlkettenunterbrechung', ('6h', '12h')), ('Hersteller', ('SHL1', 'SHL2')), ('Fleischabhang', ('7d',))):
            if lab in m.columns:
                variant |= m[lab].astype(str).str.strip().isin(bad)
        calib_ids += list(m.index[is_o2 & ~variant & ~m.index.str.endswith('b')])
    depth_calib = depth[depth.index.isin(calib_ids)]
    res = {'n_samples': int(len(depth)), 'n_experiments': len(tables),
           'reads_per_sample': {'median': float(depth.median()), 'q25': float(depth.quantile(.25)), 'q75': float(depth.quantile(.75)),
                                'min': float(depth.min()), 'max': float(depth.max()),
                                f'frac_ge_{NGS_DEPTH_SPLIT}': float((depth >= NGS_DEPTH_SPLIT).mean())},
           'reads_per_sample_header_totals_median': float(np.median(header_tot)) if header_tot else None,
           'reads_per_sample_calibration_like': {'n_samples': int(len(depth_calib)), 'median': float(depth_calib.median()),
                                                 f'frac_ge_{NGS_DEPTH_SPLIT}': float((depth_calib >= NGS_DEPTH_SPLIT).mean()),
                                                 'note': 'O2 atmosphere, no chill-chain interruption, no SHL / 7d variants, first of a replicate pair'}}
    # minimum non-zero count per table (read threshold)
    res['min_nonzero_count_per_experiment'] = {e: int(c.values[c.values > 0].min()) for e, (c, _) in tables.items()}
    # --- replicate pairs ---------------------------------------------------------------------------------------------
    pairs = []
    for exp, (c, m) in tables.items():
        for sid in c.columns:
            if sid.endswith('b') and sid[:-1] in c.columns:
                pairs.append((exp, sid[:-1], c[sid[:-1]], c[sid], m.loc[sid]))
    res['n_replicate_pairs'] = len(pairs)
    res['pairs_per_experiment'] = pd.Series([p[0] for p in pairs]).value_counts().to_dict()
    res['pair_metadata_labels'] = list(pairs[0][4].index) if pairs else []

    def model_counts(s):
        others = s.drop([t for t in MODEL_TAXA if t in s.index], errors='ignore').sum()
        return np.array([s.get(t, 0.) for t in MODEL_TAXA] + [others])

    pair_rows, points, zero_rows, groups = [], [], [], {'all': [], 'deep': [], 'shallow': []}
    pstats = {'all': [], 'deep': [], 'shallow': []}
    for exp, sid, s1, s2, m in pairs:
        k1, k2 = model_counts(s1), model_counts(s2)
        K = np.vstack([k1, k2]); Ng = K.sum(1)
        cls = 'deep' if Ng.min() >= NGS_DEPTH_SPLIT else 'shallow'
        X2, dof, a, b = pearson_group(K)
        A_pair = (a - X2 / dof) / (X2 / dof - 1) if X2 / dof > 1 else np.inf
        pair_rows.append(dict(exp=exp, sample=sid, N1=int(Ng[0]), N2=int(Ng[1]), depth_class=cls, X2=X2, dof=dof, phi=X2 / dof, A_pair=A_pair,
                              day=m.get('Tag', m.get('storage day', '')), **{k: m[k] for k in m.index if k not in ('Tag', 'storage day')}))
        for g in ('all', cls):
            groups[g].append((K, Ng)); pstats[g].append((X2, dof, a, b))
        # all genera for the relative-error cloud
        allg = sorted(set(s1.index) | set(s2.index))
        p1 = np.array([s1.get(g, 0.) for g in allg]) / Ng[0]; p2 = np.array([s2.get(g, 0.) for g in allg]) / Ng[1]
        pbar = (p1 * Ng[0] + p2 * Ng[1]) / Ng.sum()
        for j in range(len(allg)):
            if pbar[j] <= 0:
                continue
            both = p1[j] > 0 and p2[j] > 0
            points.append(dict(share=pbar[j], rel_dev=abs(p1[j] - p2[j]) / (np.sqrt(2) * pbar[j]), both=both, depth_class=cls, Nh=2 * Ng[0] * Ng[1] / Ng.sum(),
                               model_taxon=allg[j] in MODEL_TAXA))
            zero_rows.append(dict(share=pbar[j], one_zero=float((p1[j] == 0) != (p2[j] == 0)), N1=Ng[0], N2=Ng[1], depth_class=cls))
    pr = pd.DataFrame(pair_rows)
    pr.to_csv(os.path.join(out, 'ngs_replicate_pairs.csv'), index=False)
    res['pairs'] = {}
    for g in ('all', 'deep', 'shallow'):
        if not groups[g]:
            continue
        phi, A_mom = moment_A(pstats[g])
        A_ml, A_ci = dm_fit_A(groups[g], bounds=(0., 7.))
        # bootstrap over pairs for the moment estimate
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(pstats[g]), len(pstats[g]))
            boots.append(moment_A([pstats[g][i] for i in idx])[1])
        boots = np.array(boots); boots = boots[np.isfinite(boots)]
        Nh = [2 * N_[0] * N_[1] / N_.sum() for _, N_ in groups[g]]
        res['pairs'][g] = {'n_pairs': len(groups[g]), 'harmonic_depth_median': float(np.median(Nh)), 'phi': phi, 'A_moment': A_mom,
                           'A_moment_boot90': [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))] if len(boots) else None,
                           'A_mle': A_ml, 'A_mle_ci95': list(A_ci),
                           'rel_error_at_share_0.02_insilico_depth': float(np.sqrt((1 - 0.02) / 0.02 * (NGS_DEPTH_INSILICO + A_ml) / (NGS_DEPTH_INSILICO * (1 + A_ml))))}
    # shares > 1 % only (deep pairs): exclude rare categories
    deep_sub = []
    for K, Ng in groups['deep']:
        p = K.sum(0) / K.sum(); use = p > 0.01
        if use.sum() >= 2:
            deep_sub.append(pearson_group(K[:, use]))
    if deep_sub:
        res['pairs']['deep_shares_gt_1pct'] = dict(zip(('phi', 'A_moment'), moment_A(deep_sub)))
    # relative deviation by share bin
    pts = pd.DataFrame(points)
    bins = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0]
    pts['bin'] = pd.cut(pts.share, bins)
    rtab = pts[pts.both].groupby(['depth_class', 'bin'], observed=True).agg(n=('rel_dev', 'size'), rms_rel_dev=('rel_dev', lambda x: np.sqrt(np.mean(x**2))),
                                                                              mean_share=('share', 'mean'), Nh=('Nh', 'mean')).reset_index()
    rtab['multinomial_rel_sd'] = np.sqrt((1 - rtab.mean_share) / (rtab.Nh * rtab.mean_share))
    for lab, A in NGS_A_SCENARIOS.items():
        rtab[f'dm_rel_sd_A{int(A)}'] = np.sqrt((1 - rtab.mean_share) / rtab.mean_share * (rtab.Nh + A) / (rtab.Nh * (1 + A)))
    rtab['bin'] = rtab['bin'].astype(str)
    rtab.to_csv(os.path.join(out, 'ngs_relative_error_by_share.csv'), index=False)
    res['relative_error_by_share'] = rtab.to_dict(orient='records')
    # zeros
    zr = pd.DataFrame(zero_rows)
    zr['bin'] = pd.cut(zr.share, bins)

    def p_below(p, N, A, thr):
        """P(k < thr) of the beta-binomial marginal of a Dirichlet-multinomial (thr = 1: P(k = 0))."""
        return stats.betabinom(N.astype(int), A * p, A * (1 - p)).cdf(thr - 1)
    READ_THRESHOLD = 5  # the sequence tables from V08 on contain no count below 5 reads
    cols = {}
    for lab, A in NGS_A_SCENARIOS.items():
        for thr, suffix in ((1, ''), (READ_THRESHOLD, f'_thr{READ_THRESHOLD}')):
            pz1 = p_below(zr.share.values, zr.N1.values, A, thr); pz2 = p_below(zr.share.values, zr.N2.values, A, thr)
            zr[f'expected_A{int(A)}{suffix}'] = pz1 * (1 - pz2) + pz2 * (1 - pz1)
            cols[f'expected_A{int(A)}{suffix}'] = (f'expected_A{int(A)}{suffix}', 'mean')
    ztab = zr.groupby('bin', observed=True).agg(n=('one_zero', 'size'), observed_one_zero_frac=('one_zero', 'mean'), **cols).reset_index()
    res['read_threshold_assumed'] = READ_THRESHOLD
    ztab['bin'] = ztab['bin'].astype(str)
    ztab.to_csv(os.path.join(out, 'ngs_zeros_by_share.csv'), index=False)
    res['zeros_by_share'] = ztab.to_dict(orient='records')
    # --- figure --------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].hist(depth.values / 1000, bins=np.arange(0, depth.max() / 1000 + 5, 5), color=C['blue'], edgecolor='white')
    ax[0].axvline(NGS_DEPTH_INSILICO / 1000, color=C['orange'], lw=1.5, ls='--'); ax[0].text(NGS_DEPTH_INSILICO / 1000 + 1, ax[0].get_ylim()[1] * 0.9, f'in silico\nN_R = {NGS_DEPTH_INSILICO // 1000}k', color=C['orange'], fontsize=8)
    ax[0].axvline(NGS_DEPTH_SPLIT / 1000, color=C['grey'], lw=1, ls=':'); ax[0].text(NGS_DEPTH_SPLIT / 1000 - 1, ax[0].get_ylim()[1] * 0.9, f'{NGS_DEPTH_SPLIT // 1000}k', color=C['grey'], fontsize=8, ha='right')
    ax[0].set_xlabel('reads per sample (thousands)'); ax[0].set_ylabel('samples'); ax[0].set_title(f'a  Read depth (median {depth.median() / 1000:.0f}k, n = {len(depth)})')
    pp = np.logspace(-4, 0, 300)
    sub = pts[pts.both]
    for cls, col, lab in (('deep', C['blue'], f'pairs with >= {NGS_DEPTH_SPLIT // 1000}k reads'), ('shallow', C['magenta'], 'shallower pairs')):
        d = sub[sub.depth_class == cls]
        ax[1].loglog(d.share, d.rel_dev, '.', color=col, ms=5, alpha=0.55, label=lab)
    Nh_deep = res['pairs']['deep']['harmonic_depth_median'] if 'deep' in res['pairs'] else NGS_DEPTH_INSILICO
    ax[1].loglog(pp, np.sqrt((1 - pp) / (Nh_deep * pp)), color=C['grey'], lw=1.2, ls=':', label=f'multinomial only, N = {Nh_deep / 1000:.0f}k')
    for (lab, A), col, ls in zip(NGS_A_SCENARIOS.items(), (C['aqua'], C['orange'], C['violet']), ('-.', '-', '--')):
        ax[1].loglog(pp, np.sqrt((1 - pp) / pp * (Nh_deep + A) / (Nh_deep * (1 + A))), color=col, lw=2, ls=ls, label=f'Dirichlet-multinomial A = {int(A)} ({lab})')
    ax[1].axhline(0.1, color=C['light'], lw=1); ax[1].text(1.2e-4, 0.115, '10 %', fontsize=7.5, color=C['grey'])
    ax[1].set_xlabel('share of the genus (mean of the pair)'); ax[1].set_ylabel('|p1 - p2| / (sqrt(2) p)'); ax[1].set_ylim(0.003, 3); ax[1].set_xlim(1e-4, 1)
    ax[1].set_title('b  Replicate error vs. abundance'); ax[1].legend(fontsize=6.8, loc='lower left')
    x = np.arange(len(ztab)); w = 0.16
    ax[2].bar(x - 2 * w, ztab.observed_one_zero_frac, w, color=C['blue'], label='observed')
    ax[2].bar(x - w, ztab['expected_A2000'], w, color=C['orange'], alpha=0.45, label='DM, A = 2000, no threshold')
    for i, ((lab, A), col) in enumerate(zip(NGS_A_SCENARIOS.items(), (C['aqua'], C['orange'], C['violet']))):
        ax[2].bar(x + i * w, ztab[f'expected_A{int(A)}_thr{READ_THRESHOLD}'], w, color=col, label=f'DM, A = {int(A)}, < {READ_THRESHOLD} reads = 0')
    ax[2].set_xticks(x); ax[2].set_xticklabels(short_bins(ztab['bin']), fontsize=7); ax[2].set_xlabel('share of the genus (mean of the pair)')
    ax[2].set_ylabel('fraction of pairs with exactly one zero'); ax[2].legend(fontsize=6.8); ax[2].set_title('c  Zeros in replicate pairs')
    savefig(fig, out, 'zl2030_ngs_noise')
    # dispersion per pair figure
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    pr_s = pr.sort_values('N1')
    Nh = 2 * pr_s.N1 * pr_s.N2 / (pr_s.N1 + pr_s.N2)
    for cls, col in (('deep', C['blue']), ('shallow', C['magenta'])):
        d = pr_s[pr_s.depth_class == cls]
        ax.semilogx(2 * d.N1 * d.N2 / (d.N1 + d.N2), d.phi, 'o', color=col, ms=7, label=f'{cls} pairs')
    nn = np.logspace(3.8, 5.1, 100)
    for (lab, A), col, ls in zip(NGS_A_SCENARIOS.items(), (C['aqua'], C['orange'], C['violet']), ('-.', '-', '--')):
        ax.semilogx(nn, (nn + A) / (1 + A), color=col, lw=1.8, ls=ls, label=f'A = {int(A)}')
    ax.axhline(1, color=C['grey'], lw=1, ls=':'); ax.set_yscale('log')
    ax.set_xlabel('harmonic mean read depth of the pair'); ax.set_ylabel('Pearson dispersion X2 / dof (model taxa)')
    ax.set_title('NGS: overdispersion of the 15 replicate pairs'); ax.legend(fontsize=7.5, ncol=2)
    savefig(fig, out, 'zl2030_ngs_dispersion_pairs')
    return res


# ======================================================================================================================
# Initial values
# ======================================================================================================================
def x0_stats(x0, n_exp, n_cl, bounds):
    mu, sd = float(x0.mean()), float(x0.std())
    X = x0.reshape(n_exp, n_cl)
    lo, hi = bounds
    return {'n_values': int(len(x0)), 'n_species': n_cl, 'n_experiments': n_exp, 'bounds_of_fit': list(bounds),
            'frac_at_bounds': float(np.mean((x0 <= lo + 1e-4) | (x0 >= hi - 1e-4))),
            'normal_mle': {'mean': mu, 'sd': sd}, 'sd_ddof1': float(x0.std(ddof=1)), 'median': float(np.median(x0)),
            'q05': float(np.percentile(x0, 5)), 'q95': float(np.percentile(x0, 95)), 'min': float(x0.min()), 'max': float(x0.max()),
            'skew': float(stats.skew(x0)), 'kurtosis_excess': float(stats.kurtosis(x0)),
            'shapiro_p': float(stats.shapiro(x0).pvalue), 'ks_p_vs_fitted_normal': float(stats.kstest(x0, stats.norm(mu, sd).cdf).pvalue),
            'anderson_statistic': float(stats.anderson(x0, 'norm').statistic), 'anderson_crit_5pct': float(stats.anderson(x0, 'norm').critical_values[2]),
            'uniform_1_4_ks_p': float(stats.kstest(x0, stats.uniform(1., 3.).cdf).pvalue),
            'per_species_mean': [float(v) for v in X.mean(0)], 'per_species_sd': [float(v) for v in X.std(0, ddof=1)],
            'per_experiment_mean': [float(v) for v in X.mean(1)],
            'sd_between_species_means': float(X.mean(0).std(ddof=1)), 'sd_between_experiment_means': float(X.mean(1).std(ddof=1)),
            'sd_residual_within': float(np.sqrt(np.sum((X - X.mean(0) - X.mean(1)[:, None] + X.mean())**2) / ((n_exp - 1) * (n_cl - 1))))}


X0_SETS = {  # label: (result file, bounds of x0 in that calibration)
    'earlier calibration (in-silico setup until 11 Nov 2026)': ('out/noise_analysis/Result_calibration_zl2030_copy_until_2026-11-11.json', (1., 4.5)),
    'dm_fit2 (in-silico setup since 11 Nov 2026)': (FROZEN_X0_FILE, (0., 5.)),
}


def analyse_x0(out, pc_res):
    n_cl, n_exp = 12, 15
    res = {'sets': {}, 'newer_fits': {}}
    sets = {}
    for label, (f, bounds) in X0_SETS.items():
        f = f if os.path.isabs(f) else os.path.join(ROOT, f)
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            x0 = np.asarray(json.load(fh)['param_ode'], float)[:n_cl * n_exp]
        sets[label] = (x0, bounds)
        res['sets'][label] = dict(x0_stats(x0, n_exp, n_cl, bounds), source=os.path.relpath(f, ROOT))
    for lab, f in NEWER_FITS.items():
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            with open(p) as fh:
                v = np.asarray(json.load(fh)['param_ode'], float)[:n_cl * n_exp]
            res['newer_fits'][lab] = {'mean': float(v.mean()), 'sd': float(v.std()), 'median': float(np.median(v)),
                                      'frac_at_bounds_0_5': float(np.mean((v <= 1e-4) | (v >= 5. - 1e-4))), 'n_below_1': int(np.sum(v < 1.)), 'n_above_4.5': int(np.sum(v > 4.5)),
                                      'q05': float(np.percentile(v, 5)), 'q95': float(np.percentile(v, 95)),
                                      'shapiro_p': float(stats.shapiro(v).pvalue), 'file': f}
    res['insilico_sampling'] = {'distribution': 'lognormal', 'mean': 2.99, 'sigma': 1.14, 'truncated_to': [0., 5.], 'seed': 4698517,
                                'until_2026-11-11': {'mean': 2.9, 'sigma': 0.69, 'truncated_to': [1., 4.5]}}
    d0 = pc_res['day0_totals'].get('PC')
    if d0:
        res['day0_PC_totals'] = dict(d0, note='log10 total CFU/g on plate count agar at day 0; a species x0 is below the total by '
                                              'log10(1 / share) and the recovery on the medium, e.g. about 1.1 log10 for 12 equally abundant species')
    # --- figure --------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    xs = np.linspace(-0.2, 5.2, 400)
    cols = [C['blue'], C['orange']]
    labels = list(sets)
    for (label, (x0, bounds)), col in zip(sets.items(), cols):
        mu, sd = x0.mean(), x0.std()
        short = label.split(' (')[0]
        ax[0].hist(x0, bins=np.arange(0, 5.25, 0.25), density=True, color=col, alpha=0.45, edgecolor='white', label=f'{short}: N({mu:.2f}, {sd:.2f}$^2$)')
        ax[0].plot(xs, stats.norm.pdf(xs, mu, sd), color=col, lw=2)
        (osm, osr), _ = stats.probplot((x0 - mu) / sd, dist='norm')
        ax[1].plot(osm, osr, '.', color=col, ms=5, label=f'{short} (Shapiro p = {stats.shapiro(x0).pvalue:.2g})')
    ax[0].plot(xs, stats.uniform.pdf(xs, 1, 3), color=C['grey'], lw=1.5, ls='--', label='before: log-uniform U(1, 4)')
    ax[0].set_xlabel('log10 x0 (initial count of a species)'); ax[0].set_ylabel('density'); ax[0].legend(fontsize=7); ax[0].set_title('a  Fitted initial values (180 each)')
    ax[1].plot([-3, 3], [-3, 3], color=C['grey'], lw=1.2); ax[1].legend(fontsize=7)
    ax[1].set_xlabel('normal quantile'); ax[1].set_ylabel('standardized log10 x0'); ax[1].set_title('b  Normal Q-Q plots')
    if labels:
        x0, bounds = sets[labels[-1]]
        X = x0.reshape(n_exp, n_cl); mu, sd = x0.mean(), x0.std()
        ax[2].boxplot([X[:, j] for j in range(n_cl)], positions=np.arange(1, n_cl + 1), widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor=cols[len(labels) - 1], alpha=0.6, edgecolor=cols[len(labels) - 1]), medianprops=dict(color='black'), flierprops=dict(marker='.', ms=4))
        ax[2].axhline(mu, color=C['grey'], lw=1.2); ax[2].axhspan(mu - sd, mu + sd, color=C['grey'], alpha=0.12)
        for b in bounds:
            ax[2].axhline(b, color=C['grey'], lw=0.8, ls=':')
        ax[2].set_xlabel('species index'); ax[2].set_ylabel('log10 x0 over 15 experiments'); ax[2].set_title(f'c  {labels[-1].split(" (")[0]}: spread per species')
    savefig(fig, out, 'zl2030_x0_distribution')
    return res


# ======================================================================================================================
def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/noise_analysis/')
    ap.add_argument('--copy-to', default=None, help='folder for the figures and summary.json, e.g. ../fusion_model_paper/report/figures/')
    args = ap.parse_args()
    os.chdir(ROOT)
    out = os.path.join(args.out, '')
    os.makedirs(out, exist_ok=True)
    summary = {}
    print('plate counts ...', flush=True)
    summary['platecount'] = analyse_platecount(out)
    print('MALDI ...', flush=True)
    summary['maldi'] = analyse_maldi(out)
    print('NGS ...', flush=True)
    summary['ngs'] = analyse_ngs(out)
    print('initial values ...', flush=True)
    summary['x0'] = analyse_x0(out, summary['platecount'])
    with open(out + 'summary.json', 'w') as fh:
        json.dump(to_jsonable(summary), fh, indent=1)
    print(json.dumps(to_jsonable(summary), indent=1))
    if args.copy_to:
        os.makedirs(args.copy_to, exist_ok=True)
        for f in glob.glob(out + '*.png') + glob.glob(out + '*.pdf') + glob.glob(out + '*.csv') + [out + 'summary.json']:
            shutil.copy(f, args.copy_to)
        print(f'figures, tables and summary copied to {args.copy_to}')
    print(f'results in {out}')

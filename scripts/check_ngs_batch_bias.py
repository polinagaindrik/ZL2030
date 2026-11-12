#!/usr/bin/env python3
"""
Is the sequencing bias of the ZL2030 NGS data constant across experiments?
==========================================================================

For a genus i measured in the same sample by MALDI on plate count agar (share p_i, s_i = recovery) and by NGS
(share q_i, B_i = sequencing efficiency), log(p_i / q_i) = log(s_i / B_i) + c_sample. The sample term c cancels within
a sample, so the per-genus level a_i = log(s_i / B_i) must be the same in every sample if S and B are constant.
This script fits L_is = a_i + c_s + e_is (alternating means, missing entries allowed), then asks for each genus
whether the residual e_is has a level that differs between experiments (one-way ANOVA by experiment) and in
particular between the two sequencing-table formats (V01-V07 vs V08 onwards): a step there points to a change of B,
a drift with storage day inside experiments to a change of s.

Only aggregated statistics are written. Run from the root of this repository:
    python scripts/check_ngs_batch_bias.py [--out out/noise_analysis/] [--min-colonies 3]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import zl2030_noise_analysis as z  # noqa: E402

C = z.C


def maldi_pc_samples():
    """{(exp, atm, day): genus counts} of the plate count agar plates, subsamples of one condition summed;
    base conditions only (no interruption / storage variants); day-0 M0 plates count for M1."""
    plates, _ = z.read_maldi_plates()
    out = {}
    for name, s in plates.items():
        f = name.split('_')
        medium = re.sub(r'\d+$', '', f[-1])
        variant = '-'.join(f[3:-5])
        if medium != 'PC' or variant:
            continue
        atm = 'M1' if f[2] == 'M0' else f[2]
        key = (f[0], atm, f[-4])
        out[key] = out[key].add(s, fill_value=0.) if key in out else s.copy()
    return out


def ngs_samples():
    """{(exp, atm, day): genus counts} of the NGS samples, base conditions only, replicate pairs summed."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'experiments', 'NGS', 'ZOTU_Table_V*.tsv'))):
        exp = os.path.basename(f)[11:14]
        counts, meta = z.read_ngs_table(f)
        for sid in counts.columns:
            m = meta.loc[sid]
            if 'atmosphere' in m.index:          # V01-V07 format
                atm = 'M1' if 'O2' in m['atmosphere'] else 'M2'
                day = f"{int(m['storage day']):02d}"
            else:
                atm = 'M1' if 'O2' in m['Atmosphäre'] else 'M2'
                day = f"{int(m['Tag'].strip().lstrip('d')):02d}"
                if m.get('Kühlkettenunterbrechung', '').strip() in ('6h', '12h'):
                    continue
                if m.get('Hersteller', '').strip() in ('SHL1', 'SHL2') or m.get('Fleischabhang', '').strip() == '7d':
                    continue
            key = (exp, atm, day)
            out[key] = out[key].add(counts[sid], fill_value=0.) if key in out else counts[sid].copy()
    return out


def fit_two_way(df, n_iter=50):
    """L = a_genus + c_sample + e by alternating means on a long table with columns genus, sample, L."""
    a = df.groupby('genus').L.mean()
    c = pd.Series(0., index=df['sample'].unique())
    for _ in range(n_iter):
        c = (df.L - df.genus.map(a)).groupby(df['sample']).mean()
        a = (df.L - df['sample'].map(c)).groupby(df.genus).mean()
        a -= a.mean(); c += 0.  # fix the gauge: genus levels centred
        c = (df.L - df.genus.map(a)).groupby(df['sample']).mean()
    df = df.assign(a=df.genus.map(a), c=df['sample'].map(c))
    df['e'] = df.L - df.a - df.c
    return df, a, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/noise_analysis/')
    ap.add_argument('--min-colonies', type=int, default=3, help='MALDI colonies of the genus needed on the plate')
    ap.add_argument('--copy-to', default=None)
    args = ap.parse_args()
    os.chdir(ROOT)
    out = os.path.join(args.out, '')
    os.makedirs(out, exist_ok=True)
    maldi, ngs = maldi_pc_samples(), ngs_samples()
    keys = sorted(set(maldi) & set(ngs))
    rows = []
    for k in keys:
        km, kn = maldi[k], ngs[k]
        Nm, Nn = km.sum(), kn.sum()
        if Nm < 20 or Nn < 5000:
            continue
        for g in km.index:
            if km[g] >= args.min_colonies and kn.get(g, 0.) >= 5:
                p, q = km[g] / Nm, kn[g] / Nn
                rows.append(dict(exp=k[0], atm=k[1], day=int(k[2]), sample='_'.join(k), genus=g, k_maldi=km[g], N_maldi=Nm,
                                 r_ngs=kn[g], N_ngs=Nn, L=np.log(p / q),
                                 var_noise=(1 - p) / (Nm * p) + (1 - q) / q * (Nn + 2000.) / (Nn * 2001.)))
    df = pd.DataFrame(rows)
    df = df[df.groupby('sample').genus.transform('size') >= 2]      # a sample needs 2 genera to carry information
    df = df[df.groupby('genus')['sample'].transform('size') >= 10]  # a genus needs enough samples
    df, a, c = fit_two_way(df)
    df['batch'] = np.where(df.exp.str[1:].astype(int) <= 7, 'V01-V07', 'V08+')
    print(f'{len(keys)} matched samples, {df["sample"].nunique()} used, {df.genus.nunique()} genera, {len(df)} entries')
    # per genus: between-experiment vs within-experiment spread of the residual; step between formats
    res = []
    for g, d in df.groupby('genus'):
        groups = [x.e.values for _, x in d.groupby('exp') if len(x) >= 2]
        f_p = stats.f_oneway(*groups).pvalue if len(groups) >= 2 else np.nan
        within = np.sqrt(np.mean([x.var(ddof=1) for x in groups])) if groups else np.nan
        between = d.groupby('exp').e.mean().std(ddof=1) if d.exp.nunique() >= 2 else np.nan
        b_old, b_new = d[d.batch == 'V01-V07'].e, d[d.batch == 'V08+'].e
        step = b_new.mean() - b_old.mean() if len(b_old) >= 3 and len(b_new) >= 3 else np.nan
        step_p = stats.ttest_ind(b_old, b_new, equal_var=False).pvalue if len(b_old) >= 3 and len(b_new) >= 3 else np.nan
        slope = stats.linregress(d.day, d.e).slope if d.day.nunique() >= 3 else np.nan
        res.append(dict(genus=g, n_samples=len(d), n_experiments=d.exp.nunique(), level_log_s_over_B=a[g],
                        sd_within_exp=within, sd_between_exp_means=between, sd_noise_expected=np.sqrt(d.var_noise.mean()),
                        anova_p_experiment=f_p, step_V08_log=step, step_factor=np.exp(step) if np.isfinite(step) else np.nan,
                        step_p=step_p, slope_per_day=slope, n_old=len(b_old), n_new=len(b_new)))
    tab = pd.DataFrame(res).sort_values('n_samples', ascending=False)
    tab.to_csv(out + 'ngs_batch_bias_by_genus.csv', index=False)
    pd.set_option('display.width', 250)
    print(tab.round(3).to_string(index=False))
    # pooled test: does the experiment x genus interaction exceed the within-experiment noise?
    cells = df.groupby(['genus', 'exp']).e.agg(['mean', 'var', 'size'])
    cells = cells[cells['size'] >= 2]
    ss_between = float((cells['size'] * cells['mean']**2).sum()); df_b = len(cells)
    ss_within = float(((cells['size'] - 1) * cells['var']).sum()); df_w = int((cells['size'] - 1).sum())
    F = (ss_between / df_b) / (ss_within / df_w)
    pooled = dict(F=F, p=float(stats.f.sf(F, df_b, df_w)), df_between=df_b, df_within=df_w,
                  sd_within=np.sqrt(ss_within / df_w), sd_between_cells=np.sqrt(ss_between / df_b),
                  sd_noise_expected=float(np.sqrt(df.var_noise.mean())))
    print('pooled experiment x genus effect:', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pooled.items()})
    pd.DataFrame([pooled]).to_csv(out + 'ngs_batch_bias_pooled.csv', index=False)
    # figure: per genus, experiment means of the residual with the within-experiment SD
    genera = tab.genus.tolist()
    fig, axes = plt.subplots(int(np.ceil(len(genera) / 4)), 4, figsize=(12, 2.6 * np.ceil(len(genera) / 4)), sharey=True)
    exps = sorted(df.exp.unique())
    for ax, g in zip(axes.ravel(), genera):
        d = df[df.genus == g]
        m = d.groupby('exp').e.agg(['mean', 'std', 'size']).reindex(exps)
        x = np.arange(len(exps))
        old = np.array([int(e[1:]) <= 7 for e in exps])
        for sel, col, lab in ((old, C['blue'], 'V01-V07'), (~old, C['orange'], 'V08+')):
            ax.errorbar(x[sel], m['mean'].values[sel], yerr=m['std'].values[sel], fmt='o', color=col, ms=4, capsize=2, label=lab)
        ax.axhline(0, color=C['grey'], lw=0.8); ax.set_xticks(x); ax.set_xticklabels([e[1:] for e in exps], fontsize=6)
        ax.set_title(f'{g} (n = {len(d)})', fontsize=9); ax.grid(True, axis='y')
    for ax in axes.ravel()[len(genera):]:
        ax.axis('off')
    axes.ravel()[0].legend(fontsize=7); axes.ravel()[0].set_ylabel('residual log(p_MALDI / q_NGS)')
    fig.suptitle('ZL2030: per-genus level of the MALDI / NGS ratio by experiment (mean +/- SD of the samples)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out + 'zl2030_ngs_batch_bias.png', dpi=200, bbox_inches='tight'); fig.savefig(out + 'zl2030_ngs_batch_bias.pdf', bbox_inches='tight')
    if args.copy_to:
        import shutil
        for f in ('zl2030_ngs_batch_bias.png', 'zl2030_ngs_batch_bias.pdf', 'ngs_batch_bias_by_genus.csv', 'ngs_batch_bias_pooled.csv'):
            shutil.copy(out + f, args.copy_to)
    print('results in', out)


if __name__ == '__main__':
    main()

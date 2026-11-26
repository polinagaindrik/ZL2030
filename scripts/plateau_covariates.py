"""
Why do experiments at the same temperature reach different maximal counts? Noise levels and correlations with covariates
=======================================================================================================================

Constant-temperature series of ZL2030 (names V.._SH_M[01]_<T>_<day>_..., atmosphere M1 = 70 % O2 + 30 % CO2, M0 = day 0).

1. Measurement noise of the plate counts (log10): replicate plates of one package (technical) and packages of one day
   (day 0 has 5 packages per experiment) -> compared with the spread of the plateaus between experiments.
2. Plateau of each experiment and medium: maximum and last-day log10 count (mean of the replicate plates).
3. Covariates per experiment: day-0 counts (PC, MRS, VRBD), pH, TBARS, POZ (day 0, end, change), O2 / CO2 in the
   headspace at the end, NGS shares at the first sampling day, experiment number (order in time).
4. Spearman correlations of the plateau with the covariates: 2 degC only, and all temperatures after centring plateau and
   covariates within each temperature. Few experiments and many covariates: read as hypotheses, not as proof.

Output: out/current/plateau_covariates/ (tables, summary.txt, figures)
    python scripts/plateau_covariates.py
"""
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
os.chdir(ROOT)
OUT = 'out/current/plateau_covariates/'
os.makedirs(OUT, exist_ok=True)
NAME = re.compile(r'^(V\d\d)_SH_(M[01])_(02C|10C|14C|RT)_(\d\d)_MRI\d\d_CC(\d\d)_(PC|MRS|VRBD)\d\d$')
# temperatures from the fit inputs: the raw files label some series wrongly (V03: '02C', sampled as 14 degC)
EXP_T = {k: int(v) for k, v in pd.read_json('inputs_fusionmodel/exp_temps.json', typ='series').items()}
LAB = ['Lactobacillus', 'Carnobacterium', 'Lactococcus', 'Leuconostoc']
MEDIA = ['PC', 'MRS', 'VRBD']
lines = []


def say(s=''):
    print(s)
    lines.append(s)


# ---------------------------------------------------------------------------------------------------------------------
# plate counts, one row per plate
rows = []
for f in sorted(glob.glob('experiments/microbiology/*.xlsx')):
    d = pd.read_excel(f, header=0)
    reps = [c for c in d.columns if str(c).startswith('Replicate')]
    for _, r in d.iterrows():
        m = NAME.match(str(r.iloc[0]).strip())
        if not m or m.group(1) not in EXP_T:
            continue
        exp, atm, T, day, pkg, med = m.groups()
        T = f'{EXP_T[exp]:02d}'
        for c in reps:
            v = pd.to_numeric(r[c], errors='coerce')
            if np.isfinite(v) and v > 0:
                rows.append(dict(exp=exp, atm=atm, T=int(T[:2]), day=int(day), pkg=int(pkg), medium=med,
                                 log=np.log10(v)))
P = pd.DataFrame(rows)
P.to_csv(OUT + 'plates_long.csv', index=False)
say(f'{len(P)} plates, experiments: {sorted(P.exp.unique())}')

# 1. noise ------------------------------------------------------------------------------------------------------------
say('\n1. Plate-count noise (SD of log10)')
pk = P.groupby(['exp', 'T', 'day', 'pkg', 'medium']).log.agg(['mean', 'std', 'count']).reset_index()
for med in MEDIA:
    s = pk[(pk.medium == med) & (pk['count'] >= 2)]
    tech = np.sqrt(np.average(s['std']**2, weights=s['count'] - 1)) if len(s) else np.nan
    d0 = pk[(pk.medium == med) & (pk.day == 0)].groupby('exp')['mean'].agg(['std', 'count'])
    d0 = d0[d0['count'] >= 3]
    pkg_sd = np.sqrt(np.mean(d0['std']**2)) if len(d0) else np.nan
    say(f'  {med:4s}: replicate plates of one package {tech:.3f} (n={len(s)} sets); '
        f'packages of one day (day 0) {pkg_sd:.3f} (n={len(d0)} experiments)')

# 2. plateaus ---------------------------------------------------------------------------------------------------------
day_mean = pk.groupby(['exp', 'T', 'day', 'medium'])['mean'].mean().reset_index()   # mean over packages
E = day_mean.groupby(['exp', 'T']).day.max().rename('last_day').reset_index()
say(f"excluded (last sampling day < 6): {E[E.last_day < 6].exp.tolist()}")
E = E[E.last_day >= 6]
for med in MEDIA:
    s = day_mean[(day_mean.medium == med) & (day_mean.day > 0)]
    E = E.merge(s.groupby('exp')['mean'].max().rename(f'max_{med}'), on='exp', how='left')
    last = s.loc[s.groupby('exp').day.idxmax()].set_index('exp')['mean'].rename(f'last_{med}')
    E = E.merge(last, on='exp', how='left')
    d0 = day_mean[(day_mean.medium == med) & (day_mean.day == 0)].set_index('exp')['mean'].rename(f'day0_{med}')
    E = E.merge(d0, on='exp', how='left')
    E[f'rise_{med}'] = E[f'max_{med}'] - E[f'day0_{med}']

# 3. covariates -------------------------------------------------------------------------------------------------------
B = pd.read_pickle('experiments/biochemistry/biochem_oh_tbars_poz.pkl')
G = pd.read_pickle('experiments/biochemistry/biochem_o2_co2.pkl')


def series(df, quantity, exp):
    """(day, value) of one quantity of one constant-temperature experiment, day 0 from M0 or M1, then M1."""
    out = {}
    for c in df.columns:
        p = c.split('_')
        if p[0] != exp or p[1] not in ('M0', 'M1'):
            continue
        v = pd.to_numeric(df.loc[quantity, c], errors='coerce')
        if np.isfinite(v):
            out.setdefault(int(p[3]), v)
    return pd.Series(out).sort_index()


for i, e in E.iterrows():
    for q in ('pH', 'TBARS', 'POZ'):
        s = series(B, q, e.exp)
        if len(s):
            E.loc[i, f'{q}_0'] = s.iloc[0]
            E.loc[i, f'{q}_end'] = s.iloc[-1]
            E.loc[i, f'{q}_change'] = s.iloc[-1] - s.iloc[0]
            if q == 'pH':
                E.loc[i, 'pH_min'] = s.min()
    for q in ('O2', 'CO2'):
        s = series(G, q, e.exp)
        if len(s):
            E.loc[i, f'{q}_end'] = s.iloc[-1]
            E.loc[i, f'{q}_change'] = s.iloc[-1] - s.iloc[0]
E['exp_no'] = E.exp.str[1:].astype(int)
E['pkg_sd0_PC'] = E.exp.map(pk[(pk.medium == 'PC') & (pk.day == 0)].groupby('exp')['mean'].std())

ngs_file = 'out/current/k_shared_nexp0/dataframe_ngs_zl2030.pkl'
if os.path.exists(ngs_file):
    N = pd.read_pickle(ngs_file)
    N = N / N.sum()
    for i, e in E.iterrows():
        cols = [c for c in N.columns if c.startswith(e.exp + '_M1_') or c.startswith(e.exp + '_M0_')]
        if not cols:
            continue
        first = min(cols, key=lambda c: int(c.split('_')[3]))
        sh = N[first]
        E.loc[i, 'ngs_first_day'] = int(first.split('_')[3])
        for g in ('Brochothrix', 'Pseudomonas'):
            E.loc[i, f'ngs0_{g}'] = sh.get(g, np.nan)
        E.loc[i, 'ngs0_LAB'] = sh.reindex(LAB).sum()

E = E.sort_values(['T', 'exp']).reset_index(drop=True)
E.to_csv(OUT + 'experiments.csv', index=False)
say('\n2. Plateaus (log10, maximum over days > 0; mean over packages)')
say(E[['exp', 'T', 'last_day', 'day0_PC', 'max_PC', 'last_PC', 'max_MRS', 'max_VRBD', 'pH_0', 'pH_min',
       'O2_end', 'CO2_end']].round(2).to_string(index=False))
for T, g in E.groupby('T'):
    say(f'  {T:2d} degC: {len(g)} experiments, SD between experiments of max_PC {g.max_PC.std():.2f}, '
        f'max_MRS {g.max_MRS.std():.2f}, range max_PC {g.max_PC.min():.2f}-{g.max_PC.max():.2f}')

# 4. correlations -----------------------------------------------------------------------------------------------------
targets = ['max_PC', 'max_MRS', 'last_PC', 'rise_PC']
covs = ['day0_PC', 'day0_MRS', 'day0_VRBD', 'pkg_sd0_PC', 'pH_0', 'pH_min', 'pH_change', 'TBARS_0',
        'TBARS_change', 'POZ_0', 'POZ_change', 'O2_end', 'O2_change', 'CO2_end', 'CO2_change', 'ngs0_Brochothrix',
        'ngs0_Pseudomonas', 'ngs0_LAB', 'exp_no', 'max_MRS', 'max_PC', 'max_VRBD']
covs = [c for c in covs if c in E and E[c].notna().sum() >= 5]


def corr_table(D, label):
    res = []
    for t in targets:
        for c in covs:
            if c == t or (t.endswith('_PC') and c == 'max_PC') or (t == 'max_MRS' and c == 'max_MRS'):
                continue
            ok = D[[t, c]].dropna()
            if len(ok) < 5:
                continue
            rho, p = stats.spearmanr(ok[t], ok[c])
            res.append(dict(set=label, target=t, covariate=c, n=len(ok), rho=rho, p=p))
    R = pd.DataFrame(res)
    R['p_BH'] = np.nan
    if len(R):
        o = np.argsort(R.p.values)
        q = R.p.values[o] * len(R) / (np.arange(len(R)) + 1)
        R.loc[R.index[o], 'p_BH'] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1)
    return R.sort_values('p')


C2 = corr_table(E[E['T'] == 2], '2 degC')
Ec = E.copy()
num = [c for c in set(targets + covs) if c in E]
Ec[num] = E.groupby('T')[num].transform(lambda s: s - s.mean())
Ec = Ec[E.groupby('T').exp.transform('count') >= 3]
Call = corr_table(Ec, 'all T, centred within T')
R = pd.concat([C2, Call])
R.to_csv(OUT + 'correlations.csv', index=False)
for lab, Rs in (('2 degC only', C2), ('all temperatures, centred within temperature', Call)):
    say(f'\n3. Spearman correlations, {lab}: strongest 15 (p_BH: Benjamini-Hochberg over {len(Rs)} tests)')
    say(Rs.head(15)[['target', 'covariate', 'n', 'rho', 'p', 'p_BH']].round(3).to_string(index=False))

# figures -------------------------------------------------------------------------------------------------------------
top = Call[Call.target == 'max_PC'].head(6).covariate.tolist()
fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
clr = {2: '#2a78d6', 10: '#3b9a57', 14: '#c8402b'}
for ax, c in zip(axes.ravel(), top):
    for T, g in E.groupby('T'):
        ax.plot(g[c], g.max_PC, 'o', color=clr.get(T, 'k'), mec='white', ms=7, label=f'{T} °C')
        for _, r in g.iterrows():
            ax.annotate(r.exp[1:], (r[c], r.max_PC), fontsize=7, xytext=(3, 3), textcoords='offset points')
    rr = Call[(Call.target == 'max_PC') & (Call.covariate == c)].iloc[0]
    ax.set_title(f'{c}: ρ = {rr.rho:.2f}, p = {rr.p:.3f} (centred)', fontsize=9)
    ax.set_xlabel(c); ax.set_ylabel('max log10 count, PC')
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT + 'max_PC_vs_top_covariates.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4.5))
for T, g in E.groupby('T'):
    ax.plot(g.max_MRS, g.max_PC, 'o', color=clr.get(T, 'k'), mec='white', ms=7, label=f'{T} °C')
    for _, r in g.iterrows():
        ax.annotate(r.exp[1:], (r.max_MRS, r.max_PC), fontsize=7, xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('max log10 count, MRS'); ax.set_ylabel('max log10 count, PC'); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT + 'max_PC_vs_max_MRS.png', dpi=150)
plt.close(fig)

open(OUT + 'summary.txt', 'w').write('\n'.join(lines) + '\n')
print('\nsaved to', OUT)

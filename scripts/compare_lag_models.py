"""
Lag (lambda) and capacity models of the ZL2030 fit with shared inhibition: -2 log L, AIC, BIC and likelihood-ratio tests
================================================================================================================

Runs of fit_dm_concentration.py with --lambda-mode free / const / ratio (--lambda-groups) and --fix-nexp. The number of
free parameters is read from each result (n_free, including the two concentrations A), so the counts follow the bounds
of the run. Nested pairs (reduced model, full model) get a likelihood-ratio test with df = difference of free parameters.
Run from the root of this repository:
    python scripts/compare_lag_models.py [--out out/compare_lag_models.csv]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {
    'free': 'out/k_shared',                        # lambda = 10^l1 T^l_exp, shared inhibition, 12 strengths
    'ratio': 'out/k_shared_lratio',                # lambda_i = c_i mu_i(T)
    'ratio_tax': 'out/k_shared_lratio_tax',        # c: lactic acid bacteria / others
    'ratio_all': 'out/k_shared_lratio_all',        # one c
    'const': 'out/k_shared_lconst',                # lambda_exp = 0
    'nexp0': 'out/k_shared_nexp0',                 # free lambda, capacity independent of T
    'tax_free': 'out/k_shared_taxonomy',           # free lambda, inhibitor strengths by taxonomy
    'tax_ratio': 'out/k_shared_taxonomy_lratio',   # lambda = c mu, inhibitor strengths by taxonomy
}
# (reduced, full): nested pairs
NESTED = [('ratio', 'free'), ('const', 'free'), ('ratio_tax', 'ratio'), ('ratio_all', 'ratio_tax'),
          ('ratio_all', 'ratio'), ('nexp0', 'free'), ('tax_ratio', 'tax_free'), ('tax_ratio', 'ratio'),
          ('tax_free', 'free')]


def read(folder):
    f = os.path.join(ROOT, folder, 'Result_calibration_zl2030_dm.json')
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        d = json.load(fh)
    return dict(neg2logL=float(d['cost']), p=int(d['n_free']), n_data=int(d['n_data']),
                lambda_mode=d.get('lambda_mode') or 'free', lambda_groups=d.get('lambda_groups'),
                n_exp_fixed=bool(d.get('n_exp_fixed')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/compare_lag_models.csv')
    args = ap.parse_args()
    rows = {k: r for k, v in RUNS.items() if (r := read(v)) is not None}
    missing = [k for k in RUNS if k not in rows]
    if missing:
        print('not finished:', missing)
    df = pd.DataFrame.from_dict(rows, orient='index')
    n = int(df.n_data.max())
    df['AIC'] = df.neg2logL + 2 * df.p
    df['BIC'] = df.neg2logL + np.log(n) * df.p
    for c in ('neg2logL', 'AIC', 'BIC'):
        df['d_' + c] = df[c] - df[c].min()
    pd.set_option('display.width', 250)
    print(f'n = {n} data points')
    print(df[['lambda_mode', 'n_exp_fixed', 'p', 'neg2logL', 'd_neg2logL', 'd_AIC', 'd_BIC']].round(2).sort_values('d_AIC').to_string())
    lr = []
    for red, full in NESTED:
        if red in rows and full in rows:
            d = rows[red]['neg2logL'] - rows[full]['neg2logL']
            dfree = rows[full]['p'] - rows[red]['p']
            lr.append(dict(reduced=red, full=full, df=dfree, d_neg2logL=d, p_LR=float(chi2.sf(max(d, 0.), dfree)) if dfree > 0 else np.nan,
                           d_AIC=d - 2 * dfree, d_BIC=d - np.log(n) * dfree))
    if lr:
        print('\nNested comparisons (reduced - full; p_LR < 0.05: the reduced model is rejected; negative d_AIC / d_BIC: reduced preferred)')
        print(pd.DataFrame(lr).round(4).to_string(index=False))
    df.to_csv(os.path.join(ROOT, args.out))
    if lr:
        pd.DataFrame(lr).to_csv(os.path.join(ROOT, args.out.replace('.csv', '_nested.csv')), index=False)


if __name__ == '__main__':
    main()

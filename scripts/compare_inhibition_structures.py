"""
Nested comparison of inhibition structures on the ZL2030 data
=============================================================

Compares the fits of fit_dm_concentration.py with --k-structure full / product / column / none (Dirichlet-multinomial
MALDI and NGS with estimated concentrations, plate counts relative). The structures are nested in 'full', so the
total -2 log L can be compared directly; AIC = -2 log L + 2 p and BIC = -2 log L + p ln(n) weigh it against the number
of free parameters p (incl. the two concentrations) and the n data points.

    python scripts/compare_inhibition_structures.py [--runs full=out/dm_fit2 product=out/k_product column=out/k_column none=out/k_none]
Runs that are not finished are read from their start<r>.json files (best start so far).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = ['full=out/dm_fit2', 'product=out/k_product', 'column=out/k_column', 'none=out/k_none']
N_CL, N_EXPS, N_MEDIA = 12, 15, 2
K_PARAMS = {'full': N_CL * (N_CL - 1), 'product': 2 * N_CL - 1, 'column': N_CL, 'none': 0}


def read_run(folder):
    """(total -2 log L, A_maldi, A_ngs, finished, n starts within 3.84) of a run folder."""
    res = os.path.join(folder, 'Result_calibration_zl2030_dm.json')
    starts = []
    for f in glob.glob(os.path.join(folder, 'start*.json')):
        with open(f) as fh:
            d = json.load(fh)
        starts.append((d['total_neg2loglik'], d['A_maldi'], d['A_ngs'], d['log'][-1]['round']))
    if os.path.exists(res):
        with open(res) as fh:
            d = json.load(fh)
        best = (d['cost'], d['A_maldi'], d['A_ngs'])
        finished, extra = True, {k: d.get(k) for k in ('n_free', 'n_data')}
    elif starts:
        best, finished, extra = min(starts)[:3], False, {}
    else:
        return None
    near = sum(1 for s in starts if s[0] - best[0] <= 3.84)
    return dict(neg2logL=best[0], A_maldi=best[1], A_ngs=best[2], finished=finished, starts=len(starts), starts_near_best=near,
                rounds_max=max((s[3] for s in starts), default=None), **extra)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=DEFAULT, metavar='STRUCTURE=FOLDER')
    args = ap.parse_args()
    os.chdir(ROOT)
    rows = []
    for item in args.runs:
        name, folder = item.split('=')
        r = read_run(folder)
        if r is None:
            print(f'{name}: no results in {folder} yet')
            continue
        rows.append(dict(structure=name, folder=folder, k_parameters=K_PARAMS.get(name), **r))
    if not rows:
        sys.exit()
    df = pd.DataFrame(rows)
    n_free_full = next((r['n_free'] for r in rows if r.get('n_free') and r['structure'] == 'full'), None)
    n_data = next((r['n_data'] for r in rows if r.get('n_data')), None)
    # free parameters: those of the structure's own result, else from the difference in k parameters to a known run
    known = next(((r['n_free'], K_PARAMS[r['structure']]) for r in rows if r.get('n_free')), None)
    if known:
        df['p'] = [r.get('n_free') or known[0] - known[1] + K_PARAMS[r['structure']] for r in rows]
        df['AIC'] = df.neg2logL + 2 * df.p
        if n_data:
            df['BIC'] = df.neg2logL + df.p * np.log(n_data)
    ref = df.neg2logL.min()
    df['d_neg2logL'] = df.neg2logL - ref
    for c in ('AIC', 'BIC'):
        if c in df:
            df['d_' + c] = df[c] - df[c].min()
    cols = [c for c in ('structure', 'k_parameters', 'p', 'neg2logL', 'd_neg2logL', 'd_AIC', 'd_BIC', 'A_maldi', 'A_ngs', 'finished', 'starts',
                        'starts_near_best', 'rounds_max') if c in df]
    pd.set_option('display.width', 250)
    print(df[cols].round(2).to_string(index=False))
    if n_data:
        print(f'n = {n_data} data points')
    df.to_csv('out/inhibition_structures_comparison.csv', index=False)

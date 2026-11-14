"""
Cross-validation of inhibition structures by experiment (leave one experiment out)
==================================================================================

For each structure (a finished run of fit_dm_concentration.py) and each of the 15 calibration experiments e:

  1) training: refit all parameters on the data without experiment e (least squares from the structure's optimum,
     MALDI / NGS Dirichlet-multinomial with the concentrations A of the full fit, plate counts relative);
     the initial values of e have no data and stay where they are;
  2) prediction: with the trained ODE parameters and media coefficients fixed, fit only the initial values of e on its
     data of days <= X0_DAYS (as a prediction of a new experiment would: its first measurements are known);
  3) score: predictive -2 log L of the data of e after day X0_DAYS: Dirichlet-multinomial MALDI and NGS (without the
     terms that depend on the data only, the same for all structures) plus Gaussian log plate counts with the variance
     of the training residuals; also the RMSE of log10 plate counts.

A lower summed score over the folds means better prediction of unseen experiments. --summary prints the table with
paired differences to a reference structure (mean and standard error over the folds).
Run from the root of this repository:
    python scripts/cv_inhibition.py --structures product shared none full --n-jobs 10
    python scripts/cv_inhibition.py --summary
Output: out/cv_inhibition/<structure>_<experiment>.json, summary.csv
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'scripts')]

import numpy as np
import pandas as pd

RUNS = {'full': 'out/dm_fit2', 'product': 'out/k_product2', 'shared': 'out/k_shared', 'shared_c': 'out/k_sharedc',
        'shared_abundance': 'out/k_shared_abundance', 'shared_taxonomy': 'out/k_shared_taxonomy',
        'column': 'out/k_column', 'none': 'out/k_none', 'product_taxonomy': 'out/k_product_taxonomy',
        'product_gmodel': 'out/k_product_gmodel', 'shared_gmodel': 'out/k_shared_gmodel',
        # lag models (lambda = c mu: lratio, by taxonomy / one c: _tax / _all, lambda_exp = 0: lconst) and N_exp = 0
        'shared_lratio': 'out/k_shared_lratio', 'shared_lratio_tax': 'out/k_shared_lratio_tax',
        'shared_lratio_all': 'out/k_shared_lratio_all', 'shared_lconst': 'out/k_shared_lconst',
        'shared_taxonomy_lratio': 'out/k_shared_taxonomy_lratio', 'shared_nexp0': 'out/k_shared_nexp0'}
X0_DAYS = 1.            # days used to fit the initial values of the left-out experiment
TRAIN_MINUTES = 45.
OUT = 'out/cv_inhibition/'

_RAW = {}


def raw_setup():
    """Count setup on all data, once per process."""
    if 'setup' not in _RAW:
        import fit_dm_concentration as fd
        fd._SETUP.clear()
        s = fd.base_setup(OUT, 'full', 'none')
        _RAW.update(setup=s, bact=fd._SETUP['bact'])
    return _RAW['setup']


def structure_setup(result):
    """Setup with the structure and groups of a result and its bounds."""
    import fit_dm_concentration as fd
    from fusion_model.parameter_estimation.cost_function import k_structure_bounds
    s = dict(raw_setup())
    structure = result.get('k_structure') or 'full'
    if structure != 'full':
        s['k_structure'] = structure
        if result.get('k_groups') is not None:
            s['k_groups'] = list(result['k_groups'])
        s['param_bnds'] = k_structure_bounds(s['param_bnds'], s['n_cl'], structure, fd.k_start_of(s), groups=s.get('k_groups'))
    lambda_mode = result.get('lambda_mode') or 'free'
    if lambda_mode != 'free':
        from fusion_model.parameter_estimation.cost_function import lambda_mode_bounds
        s['lambda_mode'] = lambda_mode
        if result.get('lambda_groups') is not None:
            s['lambda_groups'] = list(result['lambda_groups'])
        s['param_bnds'] = lambda_mode_bounds(s['param_bnds'], s['n_cl'], lambda_mode, fd.ode_start_of(s),
                                             groups=s.get('lambda_groups'))
    if result.get('n_exp_fixed'):
        s['param_bnds'] = fd.fix_nexp_bounds(s['param_bnds'], s)
    return s


def masked(s, keep):
    """Setup whose data are NaN outside keep(experiment index, day index)."""
    days, (pc, ma, ng) = s['data_array']
    E, D = len(s['exps']), len(days)
    k = np.array([[keep(e, d) for d in range(D)] for e in range(E)])
    pc2 = np.where(k[:, None, :], pc, np.nan)
    ma2 = np.where(k[:, None, None, :], ma, np.nan)
    ng2 = np.where(k[:, None, :], ng, np.nan)
    return dict(s, data_array=(days, [pc2, ma2, ng2]))


def run_fold(task):
    """One fold; an error is recorded in <structure>_<experiment>_FAILED.json instead of stopping the other folds."""
    structure, e = task[:2]
    if len(task) > 2:
        global OUT, TRAIN_MINUTES
        OUT, TRAIN_MINUTES = task[2], task[3]
    try:
        return _run_fold(structure, e)
    except Exception as ex:
        import traceback
        with open(OUT + f'{structure}_{e}_FAILED.json', 'w') as fh:
            json.dump({'structure': structure, 'experiment': e, 'error': f'{type(ex).__name__}: {ex}',
                       'traceback': traceback.format_exc()}, fh, indent=1)
        print(f'[{structure} {e}] FAILED: {type(ex).__name__}: {ex}', flush=True)
        return None


def _run_fold(structure, e):
    import fusion_model as fm
    import fit_dm_concentration as fd
    from fusion_model.parameter_estimation.cost_function import dm_neg2loglik, residuals_withS
    out_file = OUT + f'{structure}_{e}.json'
    if os.path.exists(out_file):
        return out_file
    with open(os.path.join(RUNS[structure], 'Result_calibration_zl2030_dm.json')) as fh:
        res = json.load(fh)
    p0 = np.asarray(res['param_ode'], dtype=float)
    a_m, a_n = res['A_maldi'], res['A_ngs']
    s = structure_setup(res)
    exps, days = s['exps'], np.asarray(s['data_array'][0], dtype=float)
    ie = exps.index(e)
    n, E = s['n_cl'], len(exps)
    lo, hi = np.array(s['param_bnds'], dtype=float).T
    p0 = np.clip(p0, lo, hi)
    start = time.time()
    # 1) training without experiment e
    s_train = fd.dm_setup(masked(s, lambda ee, d: ee != ie), a_m, a_n)
    rng = np.random.default_rng(abs(hash((structure, e))) % 2**32)
    for attempt in range(3):   # a failed SVD in the least-squares step: retry from a slightly perturbed start
        try:
            t_train = time.time()
            p_train, _ = fm.pest.local_optimization(fm.pest.cost_withS, p0, s_train, method='least_squares', maxiter=500, n_jobs=1,
                                                    backend='jax', max_minutes=TRAIN_MINUTES, history_file=OUT + f'history_{structure}_{e}.csv')
            train_minutes = (time.time() - t_train) / 60.
            break
        except np.linalg.LinAlgError as ex:
            if attempt == 2:
                raise
            print(f'[{structure} {e}] {ex}: retry {attempt + 1} from a perturbed start', flush=True)
            p0 = np.clip(p0 + 1e-3 * (hi - lo) * np.where(hi > lo, rng.standard_normal(len(p0)), 0.), lo, hi)
    p_train = np.asarray(p_train, dtype=float)
    groups, _ = residuals_withS(p_train, s_train)
    r_pc_train = np.asarray(groups[2]); sigma2 = float(np.mean(r_pc_train**2))
    # 2) initial values of e from its first days, everything else fixed
    early = days <= X0_DAYS
    s_x0 = fd.dm_setup(masked(s, lambda ee, d: ee == ie and early[d]), a_m, a_n)
    x0_idx = list(range(ie * n, (ie + 1) * n))
    fixed = {i: float(p_train[i]) for i in range(len(p_train)) if i not in x0_idx and hi[i] > lo[i]}
    p_pred, _ = fm.pest.local_optimization(fm.pest.cost_withS, p_train, s_x0, method='least_squares', maxiter=200, n_jobs=1,
                                           backend='jax', fixed_values=fixed, history_file=OUT + f'history_x0_{structure}_{e}.csv')
    p_pred = np.asarray(p_pred, dtype=float)
    # 3) score on the later data of e
    s_test = fd.dm_setup(masked(s, lambda ee, d: ee == ie and not early[d]), a_m, a_n)
    km, kn = fd.data_counts(s_test)
    m_maldi, m_ngs = fd.model_proportions(p_pred, s_test)
    dm_m = dm_neg2loglik(km, m_maldi, s_test['maldi_counts'], a_m)
    dm_n = dm_neg2loglik(kn, m_ngs, s_test['ngs_counts'], a_n)
    r_pc = np.asarray(residuals_withS(p_pred, s_test)[0][2])
    pc_term = float(np.sum(r_pc**2) / sigma2 + len(r_pc) * np.log(2 * np.pi * sigma2))
    row = dict(structure=structure, experiment=e, score=dm_m + dm_n + pc_term, score_maldi=dm_m, score_ngs=dm_n,
               score_platecount=pc_term, n_pc=len(r_pc), rmse_log10_pc=float(np.sqrt(np.mean(r_pc**2)) / np.log(10)) if len(r_pc) else np.nan,
               n_maldi=int(np.isfinite(km).sum()), n_ngs=int(np.isfinite(kn).sum()), sigma_log10_train=float(np.sqrt(sigma2) / np.log(10)),
               minutes=(time.time() - start) / 60., train_minutes=train_minutes,
               train_hit_time_limit=bool(train_minutes >= 0.98 * TRAIN_MINUTES), train_attempts=attempt + 1)
    with open(out_file, 'w') as fh:
        json.dump(row, fh, indent=1)
    print(f'[{structure} {e}] score {row["score"]:.1f} (MALDI {dm_m:.1f}, NGS {dm_n:.1f}, plate counts {pc_term:.1f}), '
          f'RMSE log10 PC {row["rmse_log10_pc"]:.3f}, {row["minutes"]:.1f} min', flush=True)
    return out_file


def summary(reference='product'):
    rows = [json.load(open(f)) for f in glob.glob(OUT + '*_V*.json') if not f.endswith('_FAILED.json')]
    failed = glob.glob(OUT + '*_FAILED.json')
    if failed:
        print(f'{len(failed)} folds failed:', sorted(os.path.basename(f)[:-12] for f in failed))
    if not rows:
        print('no folds finished yet'); return
    df = pd.DataFrame(rows)
    piv = df.pivot(index='experiment', columns='structure', values='score')
    folds = piv.dropna().index
    if 'train_hit_time_limit' in df:
        print('folds whose training hit the time limit:', df.groupby('structure').train_hit_time_limit.sum().astype(int).to_dict())
    tab = df.groupby('structure').agg(folds=('score', 'size'), score_sum=('score', 'sum'), maldi=('score_maldi', 'sum'),
                                      ngs=('score_ngs', 'sum'), platecount=('score_platecount', 'sum'),
                                      rmse_log10_pc=('rmse_log10_pc', 'mean'))
    if reference in piv and len(folds):
        d = piv.loc[folds].sub(piv.loc[folds, reference], axis=0)
        tab['common_folds'] = len(folds)
        tab['sum_on_common_folds'] = piv.loc[folds].sum()
        tab[f'diff_to_{reference}'] = d.sum()
        tab['diff_se'] = d.std(ddof=1) * np.sqrt(len(folds))
        tab['folds_better_than_ref'] = (d < 0).sum()
    pd.set_option('display.width', 250)
    print(tab.round(2).sort_values('sum_on_common_folds' if 'sum_on_common_folds' in tab else 'score_sum').to_string())
    tab.to_csv(OUT + 'summary.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--structures', nargs='*', default=['product', 'shared', 'none', 'full'])
    ap.add_argument('--n-jobs', type=int, default=10)
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--reference', default='product')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--train-minutes', type=float, default=TRAIN_MINUTES)
    args = ap.parse_args()
    os.chdir(ROOT)
    OUT = os.path.join(args.out, '')
    os.makedirs(OUT, exist_ok=True)
    if args.summary:
        summary(args.reference)
        sys.exit()
    import main_zl2030_multistart as ms
    tasks = [(st, e, OUT, args.train_minutes) for e in ms.EXPS for st in args.structures]
    with mp.get_context('spawn').Pool(processes=args.n_jobs) as pool:
        for f in pool.imap_unordered(run_fold, tasks):
            pass
    summary(args.reference)

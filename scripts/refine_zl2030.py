"""
Local refinement of the best points of a multistart run on the current data
============================================================================

Takes the best point of each of the N_STARTS best starts of a multistart run (out/multistart_logk/ by default) and
refines them with the same model, bounds and objective as main_zl2030_multistart.py, on the data as read now. Use it
after a change of the data reading, e.g. the averaged day-0 subsamples (commit 6c0668e), or to converge a run that
stopped at its time limit.

Output (default out/refine_corrected/):
    Result_calibration_zl2030_refine.json   best parameters
    start_costs.csv                         each refined start: source start, cost before and after
    optimization_history_refine_start*.csv  history of each start
Run from the root of this repository:
    python scripts/refine_zl2030.py [--source out/multistart_logk] [--out out/refine_corrected] [--n-starts 12]
                                    [--max-minutes 360] [--counts]
--counts: MALDI and NGS with count likelihoods (multinomial / Dirichlet-multinomial with the colonies of each plate and
the reads of each sample), as the in-silico error model. Its costs cannot be compared with those of relative noise.
Figures and progress: add --run out/refine_corrected to plot_zl2030_multistart_best.py / show_multistart_progress.py.
"""
import argparse
import glob
import os
import re
import sys
import time
# CPU only, one thread per process (see main_zl2030_multistart.py)
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'scripts')]
import fusion_model as fm
import zl2030 as zl
import main_zl2030_multistart as ms
from fusion_model.parameter_estimation import profile_likelihood as pl
from fusion_model.parameter_estimation.calibration_utils import make_param_names
from fusion_model.parameter_estimation.jax_backend import cost_withS_jax

import numpy as np
import pandas as pd


NGS_CONCENTRATION = 2000.  # --counts: Dirichlet-multinomial concentration A of NGS, as the in-silico baseline (from ZL2030 replicate pairs)


def calibration_setup(out, counts=False):
    """Setup of main_zl2030_multistart.py on the data as read now. counts=True: MALDI multinomial and NGS
    Dirichlet-multinomial with the colonies of each plate and the reads of each sample (as the in-silico error model),
    plate counts with relative noise; otherwise all three with relative noise."""
    dfs, bact, T_x, s_x_predefined = zl.prepare_ZL2030_data(ms.EXPS, ms.EXPS_MALDI, ms.EXPS_NGS, cutoff=ms.CUTOFF,
                                                            cutoff_prop=ms.CUTOFF_PROP, path=out, add_name='_zl2030')
    bact = list(dfs[1].index)
    media = sorted(set(s.split('_')[-1].split('-')[0] for s in dfs[1].columns))
    setup = fm.pest.define_calibr_setup_insilico(
        {'model': fm.mdl.fusion_model2, 'T_x': T_x, 'workers': 1, 'output_path': out, 'n_cl': len(bact),
         'n_media': ms.N_MEDIA, 'dfs': dfs, 'exps': ms.EXPS, 'exp_temps': {e: zl.EXP_TEMPS[e] for e in ms.EXPS},
         'media': media},
        inhib=True, s_x_predefined=s_x_predefined, s_x=None, lambda_exp_max=ms.LAMBDA_EXP_MAX)
    setup['data_array'] = fm.dtf.extract_observables_from_df(dfs)
    maldi_n, ngs_dm = None, None
    if counts:
        maldi_n, n_ngs = zl.counts_per_composition(dfs, zl.sample_totals(ms.EXPS_MALDI, ms.EXPS_NGS))
        ngs_dm = (n_ngs, NGS_CONCENTRATION)
    setup.update(s_max_penalty=ms.S_MAX_PENALTY, noise_model=ms.NOISE_MODEL, maldi_n=maldi_n, ngs_dm=ngs_dm,
                 jax_options={'structured_jacobian': True})
    setup['aggregation_func'] = pl.make_gaussian_neg2loglik(setup['data_array'], ms.NOISE_MODEL, maldi_n=maldi_n,
                                                            ngs_dm=ngs_dm)
    return setup, bact, media, T_x


def best_starts(source, n):
    """[(source start, cost in the source run, parameters)] of the n best starts of a run."""
    found = []
    for f in glob.glob(os.path.join(source, 'optimization_history_multistart_start*.csv')):
        try:
            c, p = ms.best_of_start(f)
        except (ValueError, KeyError, pd.errors.EmptyDataError):
            continue
        found.append((int(re.search(r'_start(\d+)\.csv$', f).group(1)), c, p))
    if not found:
        raise SystemExit(f'No start histories in {source}')
    return sorted(found, key=lambda x: x[1])[:n]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='out/multistart_logk')
    ap.add_argument('--out', default='out/refine_corrected')
    ap.add_argument('--n-starts', type=int, default=12)
    ap.add_argument('--max-minutes', type=float, default=360)
    ap.add_argument('--n-jobs', type=int, default=None)
    ap.add_argument('--counts', action='store_true', help='MALDI and NGS with count likelihoods, see calibration_setup')
    args = ap.parse_args()
    os.chdir(ROOT)
    out = os.path.join(args.out, '')
    os.makedirs(out, exist_ok=True)
    start = time.time()

    setup, bact, media, T_x = calibration_setup(out, counts=args.counts)
    lo, hi = np.array(setup['param_bnds'], dtype=float).T
    sources = best_starts(args.source, args.n_starts)
    points = [np.clip(p, lo, hi) for _, _, p in sources]
    before = [float(cost_withS_jax(p, setup)) for p in points]
    print(f'Refining the {len(points)} best starts of {args.source} on the current data, at most {args.max_minutes:g} min '
          f'each:', flush=True)
    for (r, c, _), b in zip(sources, before):
        print(f'  source start {r}: cost {c:.3f} in the source run, {b:.3f} on the current data', flush=True)

    history = out + 'optimization_history_refine.csv'
    best, cost_best = fm.pest.multistart_optimization(
        fm.pest.cost_withS, setup, n_starts=0, extra_starts=points, method='least_squares', maxiter=ms.MAX_ITER,
        n_jobs=args.n_jobs or min(len(points), os.cpu_count()), history_file=history, backend='jax',
        max_minutes=args.max_minutes)
    n_cl = setup['n_cl']
    fm.output.json_dump({'param_ode': best.astype(list), 's_x': best[-n_cl * ms.N_MEDIA:].reshape((ms.N_MEDIA, n_cl)),
                         'T_x': T_x, 'cost': cost_best, 'objective': 'neg2logL', 'noise_model': ms.NOISE_MODEL,
                         'backend': 'jax', 'counts': args.counts,
                         'ngs_concentration': NGS_CONCENTRATION if args.counts else None, 's_max_penalty': ms.S_MAX_PENALTY,
                         'bact': bact, 'media': media, 'source': args.source,
                         'source_starts': [int(r) for r, _, _ in sources]},
                        'Result_calibration_zl2030_refine.json', dir=out)
    rows, params = [], {}
    for i, ((r, c, _), b) in enumerate(zip(sources, before)):
        f = out + f'optimization_history_refine_start{i}.csv'
        try:
            c_after, p_after = ms.best_of_start(f)
        except (OSError, ValueError, KeyError, pd.errors.EmptyDataError):
            c_after, p_after = b, points[i]
        rows.append({'start': i, 'source_start': r, 'cost_source_run': c, 'cost_current_data_before': b,
                     'cost_after': c_after, 'delta_to_best': c_after - cost_best})
        params[i] = p_after
    table = pd.DataFrame(rows).sort_values('cost_after')
    table.to_csv(out + 'start_costs.csv', index=False)
    print(table.round(3).to_string(index=False), flush=True)
    near = [i for i in table['start'] if table.set_index('start').loc[i, 'delta_to_best'] <= ms.NEAR_BEST]
    names = make_param_names(ms.EXPS, bact, media)
    near_p = np.array([params[i] for i in near])
    pd.DataFrame({'name': names, 'best': best, 'near_best_min': near_p.min(axis=0), 'near_best_max': near_p.max(axis=0),
                  'lower_bound': lo, 'upper_bound': hi}).to_csv(out + 'parameters.csv', index=False)
    print(f'{len(near)} of {len(points)} starts within {ms.NEAR_BEST} of the best cost {cost_best:.3f}; '
          f'{(time.time() - start) / 60.:.1f} min in total, results in {out}', flush=True)

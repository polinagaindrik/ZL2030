"""
Fit with Dirichlet-multinomial MALDI and NGS and estimated concentrations
==========================================================================

MALDI and NGS are fitted as Dirichlet-multinomial counts with the colonies of each plate and the reads of each sample
(zl2030.counts_per_composition), plate counts with relative noise as before. The concentrations A_MALDI and A_NGS set
how much noise there is beyond counting noise (relative error at high counts about 1 / sqrt(1 + A)) and are estimated:
each start alternates between refining the model parameters with A fixed (least squares, as refine_zl2030.py) and
setting each A to its maximum-likelihood value for the current model (cost_function.dm_concentration_mle), until both
change by less than A_TOL (log10) and the total improves by less than TOTAL_TOL, or the rounds are done. The parameter vector is the same as in the other fits.

Total -2 log L (comparable between starts, not with the relative-noise costs):
    -2 log L_DM(MALDI; A_MALDI) + -2 log L_DM(NGS; A_NGS) + n_pc ln(SSR_pc / n_pc) + max-S penalty

Output (default out/dm_fit/): Result_calibration_zl2030_dm.json (best start), start_costs.csv, start<r>.json (each
start: parameters, history of A and costs), history_dm_start<r>_round<q>.csv.
Run from the root of this repository:
    python scripts/fit_dm_concentration.py [--source out/refine_corrected] [--out out/dm_fit] [--n-starts 12]
                                           [--rounds 6] [--minutes 60] [--k-structure full]
--k-structure: reduced inhibition matrix (fusion_model cost_function.K_STRUCTURES): 'column' k_ij = b_j, 'product'
k_ij = a_i b_j, 'none' k_ij = 0. The start points are the closest reduced form of the source points. The models are
nested in 'full', so their total -2 log L can be compared: scripts/compare_inhibition_structures.py.
--k-groups (with --k-structure product / shared / shared_c): species sharing one inhibitor strength, 'abundance' (from
the NGS shares: rare genera -> background, genera with proportional share curves -> one group, fm.inhib_groups), 'model'
(the same rules on the fitted absolute curves x2_j(t) of an earlier fit without groups, --groups-from <its result JSON>;
the media are handled by the model) or 'taxonomy' (lactic acid bacteria versus all others). The groups are written to <out>/inhibition_groups.json and shown in inhibition_groups.png.
--inhibitor: 'algebraic' (default), 'F1' or 'F2' inhibitor pool of the pool model (fusion_model.model.ode_structure);
--k-scale scales the inhibition coefficients of the sources (F1 accumulates the load over time: about 0.1).
--lambda-mode: lag model, 'free' lambda = 10^l1 T^l_exp (default), 'const' l_exp = 0, 'ratio' lambda_i(T) = c_i mu_i(T)
(log10 c in the lambda_1 entries; --lambda-groups taxonomy / all: genera sharing c). Free-lambda sources are converted
with lambda(6 C) and mu(6 C) kept.
--source: a run of another fit (its history files), or an earlier run of this script (its start<r>.json) to continue it.
Figures: python scripts/plot_zl2030_multistart_best.py --result out/dm_fit/Result_calibration_zl2030_dm.json
"""
import argparse
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
import fusion_model as fm
import main_zl2030_multistart as ms
import refine_zl2030 as rz
from fusion_model.parameter_estimation import profile_likelihood as pl
from fusion_model.parameter_estimation.cost_function import (multinomial_counts, dm_neg2loglik, dm_concentration_mle,
                                                             residuals_withS, ode_params_for_model, structured_k,
                                                             k_structure_bounds, k_to_structure, K_STRUCTURES,
                                                             LAMBDA_MODES, lambda_mode_bounds, lambda_to_mode)

import numpy as np
import pandas as pd

A_TOL = 0.005          # stop when both log10 A change by less than this
TOTAL_TOL = 0.5        # ... and the total -2 log L improved by less than this in the round
A_BOUNDS = (1e-1, 1e7)


def dm_setup(setup, a_maldi, a_ngs):
    """The count setup of refine_zl2030.calibration_setup(counts=True) with MALDI and NGS as Dirichlet-multinomial."""
    n_maldi, n_ngs = setup['maldi_counts'], setup['ngs_counts']
    s = dict(setup, maldi_n=None, maldi_dm=(n_maldi, a_maldi), ngs_dm=(n_ngs, a_ngs))
    s['aggregation_func'] = pl.make_gaussian_neg2loglik(s['data_array'], ms.NOISE_MODEL, ngs_dm=s['ngs_dm'],
                                                        maldi_dm=s['maldi_dm'])
    return s


def model_proportions(p, setup):
    """Model MALDI (experiments, media, species, days) and NGS (experiments, species, days) proportions."""
    n, n_media = setup['n_cl'], setup['n_media']
    exps = setup['exps']
    days, (_, ma, ng) = setup['data_array']
    days = np.asarray(days, dtype=float)
    x0 = p[:n * len(exps)].reshape(len(exps), n)
    ode, s_x = ode_params_for_model(p[n * len(exps):-n * n_media], setup), p[-n * n_media:].reshape(n_media, n)
    m_maldi, m_ngs = np.full(np.shape(ma), np.nan), np.full(np.shape(ng), np.nan)
    for i, e in enumerate(exps):
        const = [[setup['exp_temps'][e]], n, setup['media']]
        c0 = fm.mdl.initial_state(setup['model'], 10**x0[i], n, ode, setup['exp_temps'][e])
        c = fm.mdl.model_ODE_solution(setup['model'], days, ode, c0, const)
        counts = fm.mdl.get_bacterial_count(c, n, 2)
        counts0 = fm.mdl.get_bacterial_count(c0.reshape(-1, 1), n, 2)
        m_maldi[i] = fm.mdl.observable_MALDI(days, counts, s_x, counts0, const, days)[1]
        m_ngs[i] = fm.mdl.observable_NGS(days, counts, setup['T_x'], counts0, const, days)[1]
    return m_maldi, m_ngs


def data_counts(setup):
    """Counts of MALDI and NGS (NaN: not measured)."""
    _, (_, ma, ng) = setup['data_array']
    km = np.where(np.isfinite(ma), multinomial_counts(ma, np.nan_to_num(setup['maldi_counts'], nan=1.)), np.nan)
    kn = np.where(np.isfinite(ng), multinomial_counts(ng, np.nan_to_num(setup['ngs_counts'], nan=1.)), np.nan)
    return km, kn


def estimate_concentrations(p, setup):
    """Maximum-likelihood A_MALDI, A_NGS for the parameters p and the -2 log L of both at these values."""
    km, kn = data_counts(setup)
    m_maldi, m_ngs = model_proportions(p, setup)
    a_m, l_m = dm_concentration_mle(km, m_maldi, setup['maldi_counts'], A_BOUNDS)
    a_n, l_n = dm_concentration_mle(kn, m_ngs, setup['ngs_counts'], A_BOUNDS)
    return a_m, a_n, l_m, l_n


def total_neg2loglik(p, setup, a_m, a_n):
    """Total -2 log L (see the module docstring), comparable between starts and rounds."""
    km, kn = data_counts(setup)
    m_maldi, m_ngs = model_proportions(p, setup)
    s = dm_setup(setup, a_m, a_n)
    groups, _ = residuals_withS(p, s)
    n_pc = s['aggregation_func'].n_data[2]
    total = (dm_neg2loglik(km, m_maldi, setup['maldi_counts'], a_m) + dm_neg2loglik(kn, m_ngs, setup['ngs_counts'], a_n)
             + n_pc * np.log(np.nansum(np.asarray(groups[2])**2) / n_pc))
    s_max = np.max(p[-setup['n_cl'] * setup['n_media']:])
    return float(total + sum(s['aggregation_func'].n_data) * ms.S_MAX_PENALTY * (s_max - 1.)**2)


def run_start(task):
    """Alternate least-squares refinement (A fixed) and estimation of A for one start."""
    r, p, source, out, rounds, minutes, structure, groups_method, groups_from, lambda_mode, lambda_groups, fix_nexp, inhibitor = task
    setup = base_setup(out, structure, groups_method, groups_from, lambda_mode, lambda_groups, fix_nexp, inhibitor)
    lo, hi = np.array(setup['param_bnds'], dtype=float).T
    p = np.clip(np.asarray(p, dtype=float), lo, hi)
    a_m, a_n, _, _ = estimate_concentrations(p, setup)
    log = [{'round': 0, 'A_maldi': a_m, 'A_ngs': a_n, 'total': total_neg2loglik(p, setup, a_m, a_n)}]
    print(f'[start {r}] source start {source}: A_MALDI {a_m:.3g}, A_NGS {a_n:.3g}, -2 log L {log[-1]["total"]:.2f}', flush=True)
    for q in range(1, rounds + 1):
        p, c = fm.pest.local_optimization(fm.pest.cost_withS, p, dm_setup(setup, a_m, a_n), method='least_squares',
                                          maxiter=ms.MAX_ITER, n_jobs=1, backend='jax', max_minutes=minutes,
                                          history_file=out + f'history_dm_start{r}_round{q}.csv')
        p = np.asarray(p, dtype=float)
        a_m_new, a_n_new, _, _ = estimate_concentrations(p, setup)
        total = total_neg2loglik(p, setup, a_m_new, a_n_new)
        change = max(abs(np.log10(a_m_new / a_m)), abs(np.log10(a_n_new / a_n)))
        improvement = log[-1]['total'] - total
        a_m, a_n = a_m_new, a_n_new
        log.append({'round': q, 'A_maldi': a_m, 'A_ngs': a_n, 'total': total, 'cost_fixed_A': float(c),
                    'log10_A_change': float(change)})
        print(f'[start {r}] round {q}: A_MALDI {a_m:.4g}, A_NGS {a_n:.4g}, -2 log L {total:.3f} '
              f'(log10 A change {change:.4f})', flush=True)
        with open(out + f'start{r}.json', 'w') as fh:
            json.dump({'start': r, 'source_start': source, 'param_ode': [float(x) for x in p], 'A_maldi': a_m,
                       'A_ngs': a_n, 'total_neg2loglik': total, 'log': log, 'k_structure': structure,
                       'lambda_mode': lambda_mode, 'lambda_groups': setup.get('lambda_groups')}, fh, indent=1)
        if change < A_TOL and improvement < TOTAL_TOL:
            break
    return r, source, p, a_m, a_n, log


_SETUP = {}


def dm_fit_starts(source, n):
    """[(start, -2 log L, parameters)] of the n best starts of an earlier run of this script (its start<r>.json), to
    continue it; None if source has no such files."""
    import glob
    found = []
    for f in glob.glob(os.path.join(source, 'start*.json')):
        with open(f) as fh:
            d = json.load(fh)
        found.append((int(d['start']), float(d['total_neg2loglik']), np.asarray(d['param_ode'], dtype=float)))
    return sorted(found, key=lambda x: x[1])[:n] if found else None


def ode_start_of(setup):
    """Index of the first ODE parameter (lambda_1 of species 0) in the parameter vector."""
    return setup['n_cl'] * len(setup['exps'])


def nexp_index(setup):
    """Index of N_exp (capacity N(T) = 10^N_1 T^N_exp) in the parameter vector."""
    return ode_start_of(setup) + 4 * setup['n_cl'] + 1


def fix_nexp_bounds(param_bnds, setup):
    """Bounds with N_exp fixed at 0."""
    b = [tuple(x) for x in param_bnds]
    b[nexp_index(setup)] = (0., 0.)
    return tuple(b)


def k_start_of(setup):
    """Index of the first inhibition term k_ij in the parameter vector."""
    return setup['n_cl'] * len(setup['exps']) + 4 * setup['n_cl'] + 2


GROUPS_FROM = None  # --groups-from: result JSON of a fit without groups, for --k-groups model


def inhibition_groups(s, bact, method, groups_from=None):
    """Groups of genera sharing one strength (fm.inhib_groups; 'abundance' uses the NGS shares only, 'model' the fitted
    curves of the result groups_from) and the result."""
    if method == 'model':
        with open(groups_from) as fh:
            src = json.load(fh)
        s_src = dict(s)
        s_src.pop('k_groups', None)
        if src.get('k_groups') is not None:
            raise ValueError('--groups-from must be a fit without groups')
        return fm.inhib_groups.inhibition_groups('model', param=np.asarray(src['param_ode'], dtype=float), calibr_setup=s_src)
    return fm.inhib_groups.inhibition_groups(method, data_array=s['data_array'], names=bact)


def lambda_groups_of(method, bact):
    """Species sharing one lag ratio c (--lambda-groups): None (each its own), 'all' (one c), 'taxonomy'."""
    if method in (None, 'none'):
        return None
    if method == 'all':
        return [0] * len(bact)
    return list(fm.inhib_groups.inhibition_groups('taxonomy', names=bact)[0])


def base_setup(out, structure='full', groups_method='none', groups_from=None, lambda_mode='free', lambda_groups='none',
               fix_nexp=False, inhibitor='algebraic'):
    """Count setup on the current data, once per process; structure: reduced inhibition matrix (--k-structure),
    groups_method: species sharing one strength (--k-groups); lambda_mode: lag model (--lambda-mode, see
    fusion_model cost_function.ode_params_for_model), lambda_groups: species sharing c (--lambda-groups)."""
    if 'setup' not in _SETUP:
        s, bact, media, T_x = rz.calibration_setup(out, counts=True)
        s = dict(s, maldi_counts=s['maldi_n'], ngs_counts=s['ngs_dm'][0], jax_options={'structured_jacobian': True},
                 model=fm.mdl.model_for_inhibitor(inhibitor))   # inhibitor pool F1 / F2 or the algebraic load
        if structure != 'full':
            s['k_structure'] = structure
            if groups_method != 'none':
                s['k_groups'] = inhibition_groups(s, bact, groups_method, groups_from)[0]
            s['param_bnds'] = k_structure_bounds(s['param_bnds'], s['n_cl'], structure, k_start_of(s),
                                                 groups=s.get('k_groups'))
        if lambda_mode != 'free':
            s['lambda_mode'] = lambda_mode
            if lambda_mode == 'ratio':
                s['lambda_groups'] = lambda_groups_of(lambda_groups, bact)
            s['param_bnds'] = lambda_mode_bounds(s['param_bnds'], s['n_cl'], lambda_mode, ode_start_of(s),
                                                 groups=s.get('lambda_groups'))
        if fix_nexp:
            s['n_exp_fixed'] = True
            s['param_bnds'] = fix_nexp_bounds(s['param_bnds'], s)
        _SETUP.update(setup=s, bact=bact, media=media, T_x=T_x)
    return _SETUP['setup']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='out/refine_corrected')
    ap.add_argument('--out', default='out/dm_fit')
    ap.add_argument('--n-starts', type=int, default=12)
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--minutes', type=float, default=60., help='time limit of each least-squares refinement')
    ap.add_argument('--k-structure', default='full', choices=K_STRUCTURES, help='reduced inhibition matrix')
    ap.add_argument('--k-groups', default='none', choices=fm.inhib_groups.METHODS,
                    help="species sharing one inhibitor strength (with --k-structure product / shared / shared_c)")
    ap.add_argument('--groups-from', default=None, help='--k-groups model: result JSON of a fit without groups')
    ap.add_argument('--lambda-mode', default='free', choices=LAMBDA_MODES,
                    help="lag model: 'free' lambda = 10^l1 T^l_exp, 'const' l_exp = 0, 'ratio' lambda = c mu(T)")
    ap.add_argument('--inhibitor', default='algebraic', choices=('algebraic', 'F1', 'F2'),
                    help="inhibition: 'algebraic' load = current abundances, inhibitor pool 'F1' (produced by all growing "
                         "cells) or 'F2' (produced by the growing cells times the resource), fusion_model.model.ode_structure")
    ap.add_argument('--k-scale', type=float, default=1.,
                    help='factor on the inhibition coefficients of the sources (start of another inhibitor model)')
    ap.add_argument('--fix-nexp', action='store_true',
                    help='capacity independent of temperature: N_exp = 0 (N_1 of the sources set to keep N at 6 C)')
    ap.add_argument('--lambda-groups', default='none', choices=('none', 'taxonomy', 'all'),
                    help="--lambda-mode ratio: species sharing c ('all': one c for all genera)")
    args = ap.parse_args()
    os.chdir(ROOT)
    out = os.path.join(args.out, '')
    os.makedirs(out, exist_ok=True)
    start = time.time()
    sources = dm_fit_starts(args.source, args.n_starts) or rz.best_starts(args.source, args.n_starts)
    print(f'{len(sources)} starts from {args.source}, at most {args.rounds} rounds of {args.minutes:g} min', flush=True)
    structure = args.k_structure
    if args.k_groups == 'model' and not args.groups_from:
        raise SystemExit('--k-groups model needs --groups-from <Result_calibration_zl2030_dm.json of a fit without groups>')
    setup0 = base_setup(out, structure, args.k_groups, args.groups_from, args.lambda_mode, args.lambda_groups, args.fix_nexp,
                        args.inhibitor)
    n_cl0 = setup0['n_cl']
    if setup0.get('k_groups') is not None:
        groups0, res0 = inhibition_groups(setup0, _SETUP['bact'], args.k_groups, args.groups_from)
        info = {'method': args.k_groups, 'groups': groups0, 'genera': _SETUP['bact'],
                'clusters': [[_SETUP['bact'][j] for j in c] for c in res0['clusters']]}
        if args.k_groups in ('abundance', 'model'):
            info.update(share=[float(v) for v in res0['share']], min_share=res0['min_share'], min_similarity=res0['min_similarity'],
                        groups_from=args.groups_from)
            fm.inhib_groups.plot_inhibition_groups(res0, _SETUP['bact'], out + 'inhibition_groups.png',
                                                   title='ZL2030: genera sharing one inhibitor strength')
        with open(out + 'inhibition_groups.json', 'w') as fh:
            json.dump(info, fh, indent=1)
        print('Inhibition groups:', info['clusters'], flush=True)
    if structure != 'full':  # sources of a full fit: closest reduced form; sources of the same structure: unchanged
        lo0, hi0 = np.array(setup0['param_bnds'], dtype=float).T
        ks0 = k_start_of(setup0)
        kb = slice(ks0, ks0 + n_cl0 * n_cl0)
        def to_structure(p, i):
            if np.all((p[kb] >= lo0[kb] - 1e-9) & (p[kb] <= hi0[kb] + 1e-9)):  # k block already of this structure
                return p
            if setup0.get('k_groups') is not None:  # shared layout without groups -> group strength = mean of members
                ks = k_start_of(setup0)
                k = p[ks:ks + n_cl0 * n_cl0].reshape(n_cl0, n_cl0)
                idx = np.arange(n_cl0)
                b = k[(idx + 1) % n_cl0, idx].copy()
                g = np.array(setup0['k_groups'])
                for gg in set(g):
                    members = np.where(g == gg)[0]
                    k[(members + 1) % n_cl0, members] = 0.
                    k[(members[0] + 1) % n_cl0, members[0]] = b[members].mean()
                q = p.copy(); q[ks:ks + n_cl0 * n_cl0] = k.ravel()
                return np.clip(q, lo0, hi0)
            if structure == 'product' and i % 2:  # every second start: the 'column' form with a_i = 1, which 'product' contains
                q = k_to_structure(p, n_cl0, 'column', k_start_of(setup0))
                ks, idx = k_start_of(setup0), np.arange(n_cl0)
                k = q[ks:ks + n_cl0 * n_cl0].reshape(n_cl0, n_cl0)
                k[idx, (idx + 1) % n_cl0] = 1.
                return q
            return k_to_structure(p, n_cl0, structure, k_start_of(setup0))
        sources = [(r, c, to_structure(np.asarray(p, dtype=float), i)) for i, (r, c, p) in enumerate(sources)]
        print(f'Inhibition structure {structure!r}: {int(np.sum(hi0 > lo0))} free parameters', flush=True)
    if args.lambda_mode != 'free':  # sources with a free lambda (lambda_exp != 0): lambda(6 C) and mu(6 C) kept
        o0 = ode_start_of(setup0)
        lo0, hi0 = np.array(setup0['param_bnds'], dtype=float).T
        def to_lambda(p):
            if np.any(p[o0 + n_cl0:o0 + 2 * n_cl0] != 0.):
                p = lambda_to_mode(p, n_cl0, args.lambda_mode, o0, groups=setup0.get('lambda_groups'))
            return np.clip(p, lo0, hi0)
        sources = [(r, c, to_lambda(np.asarray(p, dtype=float))) for r, c, p in sources]
        print(f'Lag model {args.lambda_mode!r}, groups {setup0.get("lambda_groups")}: {int(np.sum(hi0 > lo0))} free parameters', flush=True)
    if args.fix_nexp:  # capacity at 6 C kept: N_1' = N_1 + N_exp log10(6), N_exp = 0
        i_ne = nexp_index(setup0)
        def to_nexp0(p):
            p = p.copy()
            if p[i_ne] != 0.:
                p[i_ne - 1] += p[i_ne] * np.log10(6.); p[i_ne] = 0.
            return p
        sources = [(r, c, to_nexp0(np.asarray(p, dtype=float))) for r, c, p in sources]
        print('Capacity independent of temperature (N_exp = 0)', flush=True)
    if args.k_scale != 1.:   # start of another inhibitor model: strengths of the sources scaled (k block)
        ks0 = k_start_of(setup0)
        def scale_k(p):
            p = np.array(p, dtype=float); p[ks0:ks0 + n_cl0 * n_cl0] *= args.k_scale
            return np.clip(p, *np.array(setup0['param_bnds'], dtype=float).T)
        sources = [(r, c, scale_k(p)) for r, c, p in sources]
    tasks = [(i, p, int(r), out, args.rounds, args.minutes, structure, args.k_groups, args.groups_from,
              args.lambda_mode, args.lambda_groups, args.fix_nexp, args.inhibitor) for i, (r, _, p) in enumerate(sources)]
    with mp.get_context('spawn').Pool(processes=min(len(tasks), os.cpu_count())) as pool:
        results = pool.map(run_start, tasks, chunksize=1)
    rows = [{'start': r, 'source_start': src, 'A_maldi': a_m, 'A_ngs': a_n, 'total_neg2loglik': log[-1]['total'],
             'rounds': log[-1]['round']} for r, src, p, a_m, a_n, log in results]
    table = pd.DataFrame(rows).sort_values('total_neg2loglik')
    table['delta_to_best'] = table['total_neg2loglik'] - table['total_neg2loglik'].min()
    table.to_csv(out + 'start_costs.csv', index=False)
    print(table.round(4).to_string(index=False), flush=True)
    r_best = int(table.iloc[0]['start'])
    _, src, p, a_m, a_n, log = next(x for x in results if x[0] == r_best)
    n_cl, n_media = _SETUP['setup']['n_cl'], _SETUP['setup']['n_media']
    fm.output.json_dump({'param_ode': np.asarray(p).astype(list), 's_x': np.asarray(p)[-n_cl * n_media:].reshape(n_media, n_cl),
                         'T_x': _SETUP['T_x'], 'cost': float(table.iloc[0]['total_neg2loglik']), 'A_maldi': a_m,
                         'A_ngs': a_n, 'objective': 'Dirichlet-multinomial MALDI and NGS, A estimated; plate counts relative',
                         'bact': _SETUP['bact'], 'media': _SETUP['media'], 'source': args.source, 'source_start': src,
                         'start': r_best, 'log': log, 'k_structure': structure, 'k_groups': setup0.get('k_groups'),
                         'lambda_mode': args.lambda_mode, 'lambda_groups': setup0.get('lambda_groups'),
                         'n_exp_fixed': bool(args.fix_nexp), 'inhibitor': args.inhibitor,
                         'k_full': structured_k(np.asarray(p)[k_start_of(setup0):k_start_of(setup0) + n_cl0 * n_cl0], n_cl0, structure,
                                                groups=setup0.get('k_groups')),
                         'n_free': int(np.sum(np.diff(np.array(setup0['param_bnds'], dtype=float), axis=1) > 0)) + 2,
                         'n_data': int(sum(dm_setup(setup0, a_m, a_n)['aggregation_func'].n_data))},
                        'Result_calibration_zl2030_dm.json', dir=out)
    near = int((table['delta_to_best'] <= ms.NEAR_BEST).sum())
    print(f'{near} of {len(table)} starts within {ms.NEAR_BEST} of the best -2 log L; '
          f'{(time.time() - start) / 60.:.1f} min in total, results in {out}', flush=True)

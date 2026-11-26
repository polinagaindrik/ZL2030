"""Plate counts on VRBD (Enterobacterales) of the constant-temperature ZL2030 series, one panel per temperature, data only
(mean over the plates of a day, connected; bars: SD of the plates). Same experiments, colours and markers as the model
plots (plot_results/plot_model_results.py). With --with-pc the PC counts of the same experiments are drawn faintly.
Reads out/plateau_covariates/plates_long.csv (scripts/plateau_covariates.py).
    python scripts/plot_vrbd_curves.py [--with-pc]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fusion_model.plotting import plotting_templates as pt

WITH_PC = '--with-pc' in sys.argv
MARKERS = ['^', 'o', 'D', 's', 'X', '.', '*', 'P', 'd', 'x', '^', 'o', 'D', 's']
P = pd.read_csv('out/plateau_covariates/plates_long.csv')
P = P[P.exp != 'V10']                        # only day 0 and 1
D = P.groupby(['exp', 'T', 'day', 'medium']).log.agg(['mean', 'std']).reset_index()
temps = [2, 10, 14]
fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True, gridspec_kw=dict(wspace=0.04))
for ax, T, lab in zip(axes, temps, 'abc'):
    d = D[D['T'] == T]
    for i, exp in enumerate(sorted(d.exp.unique())):
        c = pt.exp_clrs.get(exp, 'k')
        if WITH_PC:
            s = d[(d.exp == exp) & (d.medium == 'PC')].sort_values('day')
            ax.plot(s.day, s['mean'], '-', color=c, alpha=.25, lw=1.2)
        s = d[(d.exp == exp) & (d.medium == 'VRBD')].sort_values('day')
        ax.errorbar(s.day, s['mean'], yerr=s['std'].fillna(0), fmt='-' + MARKERS[i % len(MARKERS)], color=c, ms=8,
                    lw=1.5, capsize=2, label=f'Experiment {int(exp[1:])}')
    ax.legend(loc='upper left', fontsize=12, frameon=False)
    ax.text(.97, .05, f'{lab}) {T}°C', transform=ax.transAxes, ha='right', fontsize=20)
    ax.set_xlabel(r'Time, $t$ [day]', fontsize=14)
    ax.set_xlim(-0.3, 17.4 if T == 2 else 10.3)
axes[0].set_ylabel(r'log CFU g$^{-1}$, VRBD', fontsize=14)

name = 'out/plateau_covariates/platecount_VRBD_data' + ('_with_PC' if WITH_PC else '') + '.png'
fig.savefig(name, dpi=150, bbox_inches='tight')
print('saved', name)

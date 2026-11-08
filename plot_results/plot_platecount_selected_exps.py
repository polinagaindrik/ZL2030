"""
Plate counts of selected constant-temperature experiments, data only (as MiBi_PC_2Grad.png of the talks)
========================================================================================================

Output: out/plot_platecount/MiBi_<medium>_<temp>Grad.png (and .pdf)
Run from the root of this repository:
    python plot_results/plot_platecount_selected_exps.py
    python plot_results/plot_platecount_selected_exps.py --csv out/zl2030/refine_corrected/best_so_far/dataframe_platecount_zl2030.csv
--csv: plate counts from a dataframe saved by zl.read_all_data_in_df instead of experiments/microbiology/
"""
import argparse
import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
import zl2030 as zl
from fusion_model.plotting import plotting_templates as plt_templ

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIME_LABEL = r'Time, $t$ [day]'
COUNT_LABEL = r'log CFU mL$^{-1}$'
MARKERS = ['^', 'o', 'D', 's', 'X', '*', 'P', 'd']
PANEL = {2.: 'a)', 10.: 'b)', 14.: 'c)'}


def read_platecount(csv=None, exps=None):
    """Plate count dataframe (rows Average, Replicas, Standard deviation), from experiments/ or a saved csv."""
    if csv is None:
        df = zl.get_all_experiments_dataframe(zl.read_platecount, exps, 'experiments/microbiology/')
        return fm.dtf.drop_column(df, ['M2', 'VRBD'])
    df = pd.read_csv(csv)
    df.index = ['Average', 'Replicas', 'Standard deviation']  # order of read_all_data_in_df; Replicas stay strings
    return df


def plot_selected(df, exps, temp, medium, path, add_name=''):
    """Measured plate counts (mean +- SD, in log10) of the experiments at one temperature, connected by lines."""
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for exp, mrk in zip(exps, MARKERS):
        d = df.filter(regex=f'^{exp}_.*_{int(temp):02d}C_.*_{medium}-')
        days = np.array([float(c.split('_')[3]) for c in d.columns])
        mean = np.array(d.loc['Average'], dtype=float)
        sd = np.array(d.loc['Standard deviation'], dtype=float)
        o = np.argsort(days)
        ax.errorbar(days[o], np.log10(mean[o]), yerr=np.abs(0.43*sd[o]/mean[o]), fmt=mrk, linestyle='solid', linewidth=1.5,
                    markersize=7, color=plt_templ.exp_clrs[exp], label=exp)
    ax.set_xlim(-0.2, None)
    ax.set_xticks(np.arange(0, ax.get_xlim()[1], 4))
    fm.plotting.set_labels(fig, ax, TIME_LABEL, COUNT_LABEL)
    ax.text(0.95, 0.06, f'{PANEL.get(temp, "")} {temp:g}°C', fontsize=18, ha='right', va='bottom', transform=ax.transAxes)
    ax.legend(loc='upper left', fontsize=15, framealpha=0.1)
    for ext in ('png', 'pdf'):
        fig.savefig(path + f'MiBi_{medium}_{temp:g}Grad{add_name}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(path + f'MiBi_{medium}_{temp:g}Grad{add_name}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=None)
    parser.add_argument('--exps', nargs='+', default=['V01', 'V11', 'V19'])
    parser.add_argument('--temp', type=float, default=2.)
    parser.add_argument('--medium', default='PC')
    parser.add_argument('--out', default='out/plot_platecount/')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plot_selected(read_platecount(args.csv, args.exps), args.exps, args.temp, args.medium, args.out)

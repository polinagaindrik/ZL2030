#!/usr/bin/env python3
from functools import partial

import numpy as np
import pandas as pd
from pybis import Openbis
import getpass

from fusion_model.tools.dataframe_functions import drop_column, merge_dfs, calc_proportion
from fusion_model.tools import dataframe_functions as dtf
from fusion_model.data.output import read_from_json

# Storage temperature (degC) of each experiment series. V03 is 14 degC, although its colony count sample names say 02C:
# its sampling days and growth are those of the 14 degC runs and its MALDI / NGS samples are labelled 14C.
EXP_TEMPS = {'V01': 2., 'V02': 10., 'V03': 14., 'V04': 14., 'V05': 10., 'V06': 14., 'V07': 2., 'V08': 2., 'V09': 10.,
             **{f'V{i:02d}': 2. for i in range(10, 29)}}

# NGS genera that are probably reagent / kit contamination, not meat flora (marked only, NOT removed by default):
# listed as contaminants of extraction kits and reagents (Salter et al. 2014, doi:10.1186/s12915-014-0087-z), never
# identified by MALDI among ~31 000 colonies of ZL2030, and their share is highest in samples with a low bacterial load.
# Streptophyta: chloroplast 16S (plant DNA). Remove them with drop_taxa=SUSPECTED_NGS_CONTAMINANTS (prepare_ZL2030_data,
# sample_totals, ...); the default drop_taxa=() keeps all reads, as in the earlier fits. Without blank (negative)
# controls of the sequencing this is circumstantial: a falling share with the load alone also fits real flora that does
# not grow, so culture (MALDI) is the deciding criterion.
SUSPECTED_NGS_CONTAMINANTS = ('Bradyrhizobium', 'Sphingomonas', 'Burkholderia', 'Deinococcus', 'Paracoccus',
                              'Propionibacterium', 'Ralstonia', 'Methylobacterium', 'Streptophyta')

# Order Enterobacterales (all seven families of the current taxonomy: Enterobacteriaceae, Yersiniaceae, Hafniaceae,
# Erwiniaceae, Morganellaceae, Pectobacteriaceae, Budviciaceae). With group_enterobacterales=True they are read as one
# taxon 'Enterobacterales': in NGS all reads of the order (most are classified to family only, 'Enterobacteriales;
# Enterobacteriaceae;;', and are otherwise dropped for lack of a genus name), in MALDI these genera. The selective medium
# of the group is VRBD (media=('PC', 'MRS', 'VRBD')). Default False: the reading of the earlier fits.
ENTEROBACTERALES = 'Enterobacterales'
ENTEROBACTERALES_GENERA = frozenset({
    'Atlantibacter', 'Buttiauxella', 'Citrobacter', 'Cronobacter', 'Enterobacter', 'Escherichia', 'Escherichia/Shigella',
    'Franconibacter', 'Klebsiella', 'Kluyvera', 'Kosakonia', 'Leclercia', 'Lelliottia', 'Pseudescherichia', 'Raoultella',
    'Salmonella', 'Shigella', 'Siccibacter', 'Enterobacteriaceae',                                # Enterobacteriaceae
    'Serratia', 'Yersinia', 'Ewingella', 'Rahnella',                                              # Yersiniaceae
    'Hafnia', 'Obesumbacterium', 'Edwardsiella',                                                  # Hafniaceae
    'Erwinia', 'Pantoea', 'Rosenbergiella',                                                       # Erwiniaceae
    'Morganella', 'Proteus', 'Providencia',                                                       # Morganellaceae
    'Pectobacterium', 'Dickeya', 'Lonsdalea',                                                     # Pectobacteriaceae
    'Budvicia', 'Pragia'})                                                                        # Budviciaceae
DEFAULT_MEDIA = ('PC', 'MRS')
ALL_MEDIA = ('PC', 'MRS', 'VRBD')


def _ngs_genus(tax, group_enterobacterales=False):
    """Taxon name of one NGS taxonomy string ('' = dropped): the genus field; eukaryotic reads are dropped. With
    group_enterobacterales every read of the order Enterobacteriales / Enterobacterales, with or without genus."""
    parts = tax.split(';')
    if parts[0] == 'Eukaryota':
        return ''
    if group_enterobacterales and any(p.strip().strip('"') in ('Enterobacteriales', 'Enterobacterales') for p in parts):
        return ENTEROBACTERALES
    return parts[-2] if len(parts) >= 2 else ''


def _media_to_drop(media):
    """Plate-count / MALDI media not in media, as patterns for drop_column."""
    unknown = set(media) - set(ALL_MEDIA)
    if unknown:
        raise ValueError(f'unknown media {sorted(unknown)}, known: {ALL_MEDIA}')
    return [m for m in ALL_MEDIA if m not in media]


def get_bacteria_for_model(exps_maldi, exps_ngs, path_data, cutoff=0., cutoff_prop=0., drop_taxa=(), group_enterobacterales=False):
    """Load MALDI and NGS experiments and return their shared filtered bacterial set.
    The selection uses MALDI of PC and MRS only (VRBD plates exist only for a few experiments and are selective).
    drop_taxa: NGS taxa removed before the shares are computed (e.g. SUSPECTED_NGS_CONTAMINANTS; default none).
    group_enterobacterales: Enterobacterales as one taxon in NGS and MALDI (default False)."""
    df_ngs = get_all_experiments_dataframe(partial(read_ngs, drop_taxa=drop_taxa, group_enterobacterales=group_enterobacterales),
                                           exps_ngs, path_data+'NGS/')
    df_ngs = dtf.drop_column(df_ngs, ['M2'])
    df_ngs, bact_ngs = dtf.preprocess_dataframe(df_ngs, cutoff=cutoff, cutoff_prop=cutoff_prop, calc_prop=False)

    df_maldi = get_all_experiments_dataframe(partial(read_maldi, group_enterobacterales=group_enterobacterales), exps_maldi,
                                             path_data+'MALDI/')
    df_maldi = dtf.drop_column(df_maldi, ['M2', 'VRBD'])
    df_maldi, bact_maldi = dtf.preprocess_dataframe(df_maldi, cutoff=cutoff, cutoff_prop=cutoff_prop, calc_prop=False)
    bact_all = dtf.intersection(bact_ngs.values, bact_maldi.values)
    return bact_all, df_maldi.T[bact_all].T, df_ngs.T[bact_all].T


def sample_totals(exps_maldi, exps_ngs, path_data='experiments/', drop_taxa=(), group_enterobacterales=False, media=DEFAULT_MEDIA):
    """Colonies identified on each MALDI plate and reads of each NGS sample, as Series indexed like the columns of the
    MALDI and NGS data frames of prepare_ZL2030_data (same reading and the same filters, before proportions; use the
    same drop_taxa, group_enterobacterales and media as there)."""
    df_maldi = get_all_experiments_dataframe(partial(read_maldi, counts=True), exps_maldi, path_data+'MALDI/')
    df_maldi = dtf.drop_column(df_maldi, ['M2'] + _media_to_drop(media))
    df_ngs = get_all_experiments_dataframe(partial(read_ngs, counts=True, drop_taxa=drop_taxa,
                                                   group_enterobacterales=group_enterobacterales), exps_ngs, path_data+'NGS/')
    df_ngs = dtf.drop_column(df_ngs, ['M2'])
    return df_maldi.sum(), df_ngs.sum()


def counts_per_composition(dfs, totals):
    """Arrays of N per composition for calibr_setup['maldi_n'] and the N of calibr_setup['ngs_dm'], aligned with
    fm.dtf.extract_observables_from_df(dfs): MALDI (experiments, media, 1, days), NGS (experiments, 1, days).
    dfs: the data frames of prepare_ZL2030_data; totals: sample_totals(...). Raises if a composition has no total."""
    n_maldi, n_ngs = totals
    for df, n, what in ((dfs[1], n_maldi, 'MALDI'), (dfs[2], n_ngs, 'NGS')):
        missing = [c for c in df.columns if c not in n.index]
        if missing:
            raise ValueError(f'{what}: no total for {missing[:5]}')
    one_row = lambda df, n: pd.DataFrame([n[df.columns].astype(float).values], columns=df.columns)
    _, (_, arr_maldi, arr_ngs) = dtf.extract_observables_from_df([dfs[0], one_row(dfs[1], n_maldi), one_row(dfs[2], n_ngs)])
    return arr_maldi, arr_ngs


def read_all_data_in_df(exps_platecount, exps_maldi, exps_ngs, path_data, path_out, bact_all=None, cutoff=0., cutoff_prop=0., add_name='',
                        drop_taxa=(), group_enterobacterales=False, media=DEFAULT_MEDIA):
    """Load, align, and save calibration dataframes for the selected experiments.
    media: plate-count and MALDI media used (default PC and MRS; add 'VRBD' for the Enterobacterales medium)."""
    df_platecount = get_all_experiments_dataframe(read_platecount, exps_platecount, path_data+'microbiology/')
    df_platecount =dtf.drop_column(df_platecount, ['M2'] + _media_to_drop(media))
    if bact_all is None:
        df_maldi = get_all_experiments_dataframe(read_maldi, exps_maldi, path_data+'MALDI/')
        df_ngs = get_all_experiments_dataframe(partial(read_ngs, drop_taxa=drop_taxa), exps_ngs, path_data+'NGS/')
        # Restore ngs and maldi dataframes on missing bacteria
        #bact_all = sorted(set(list(bact_ngs.values) + list(bact_maldi.values)))
        #df_maldi, df_ngs, T_x, s_x_predefined = fm.dtf.make_df_maldi_ngs_compatible(df_maldi, df_ngs, cutoff=cutoff0)
        df_ngs = df_ngs.T[bact_all].T
        df_maldi = df_maldi.T[bact_all].T
    else:
        bact_all, df_maldi, df_ngs = get_bacteria_for_model(exps_maldi, exps_ngs, path_data, cutoff=cutoff, cutoff_prop=cutoff_prop,
                                                            drop_taxa=drop_taxa, group_enterobacterales=group_enterobacterales)
        if 'VRBD' in media:   # the selection uses PC and MRS only; add the VRBD plates, aligned to the same taxa
            df_vrbd = get_all_experiments_dataframe(partial(read_maldi, group_enterobacterales=group_enterobacterales),
                                                    exps_maldi, path_data+'MALDI/').filter(like='VRBD')
            df_vrbd = dtf.drop_column(df_vrbd, ['M2'])
            df_vrbd = df_vrbd.fillna(0.)
            df_vrbd = df_vrbd.loc[:, df_vrbd.sum() > 0]          # shares of all genera; the rest becomes Others below
            df_maldi = pd.concat([df_maldi, df_vrbd.reindex(df_maldi.index).fillna(0.)], axis=1)

    df_maldi, bact_all = dtf.add_rest_bacteria(df_maldi, bact_all)
    df_ngs, _ = dtf.add_rest_bacteria(df_ngs, bact_all)

    T_x = np.ones((len(bact_all)))
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    s_x_predefined = np.ones((len(media), len(bact_all)))*np.nan
    for i, med in enumerate(media):
        bact_med_null = df_maldi.filter(like=med)[df_maldi.filter(like=med).T.sum()==0].T.columns
        ind_med_null = [bact_all.index(i) for i in list(bact_med_null)]
        s_x_predefined[i, ind_med_null] = 0.

    data = [df_platecount, df_maldi, df_ngs]
    df_platecount.to_pickle(path_out+f'dataframe_platecount{add_name}.pkl')
    df_maldi.to_pickle(path_out+f'dataframe_maldi{add_name}.pkl') 
    df_ngs.to_pickle(path_out+f'dataframe_ngs{add_name}.pkl')

    df_platecount.to_csv(path_out+f'dataframe_platecount{add_name}.csv', index=False)
    df_maldi.to_csv(path_out+f'dataframe_maldi{add_name}.csv')
    df_ngs.to_csv(path_out+f'dataframe_ngs{add_name}.csv')

    print(bact_all)
    data = dtf.filter_dataframe_regex('V.._', data)
    return data, T_x, s_x_predefined


def prepare_ZL2030_data(exps_platecount, exps_maldi, exps_ngs, cutoff=0., cutoff_prop=0., path='', add_name='', drop_taxa=(),
                        group_enterobacterales=False, media=DEFAULT_MEDIA):
    """Prepare the ZL2030 calibration data and return observations and filtering defaults.
    drop_taxa: NGS taxa removed before the shares are computed (e.g. SUSPECTED_NGS_CONTAMINANTS; default none);
    group_enterobacterales: Enterobacterales as one taxon in NGS and MALDI (default False);
    media: plate-count and MALDI media (default PC, MRS; ALL_MEDIA adds VRBD, the medium of the Enterobacterales).
    Pass the same options to sample_totals; the number of media of the model is len(media)."""
    bact_all, df_maldi_all, df_ngs_all = get_bacteria_for_model(exps_maldi, exps_ngs, 'experiments/', cutoff=cutoff, cutoff_prop=cutoff_prop,
                                                                drop_taxa=drop_taxa, group_enterobacterales=group_enterobacterales)
    dfs_calibr, T_x, s_x_predefined = read_all_data_in_df(exps_platecount, exps_maldi, exps_ngs, 'experiments/', path, bact_all=bact_all,
                                                          cutoff=cutoff, cutoff_prop=cutoff_prop, add_name=add_name, drop_taxa=drop_taxa,
                                                          group_enterobacterales=group_enterobacterales, media=media)
    return dfs_calibr, bact_all, T_x, s_x_predefined


def prepare_ZL2030_data_for_prediction(exps, bact_all, path_data, path_out='', add_name='', drop_taxa=(),
                                       group_enterobacterales=False, media=DEFAULT_MEDIA):
    """Load prediction experiments and align their bacteria to the calibration set (same options as the calibration)."""
    df_platecount = get_all_experiments_dataframe(read_platecount, exps, path_data+'microbiology/')
    df_platecount = dtf.drop_column(df_platecount, ['M2'] + _media_to_drop(media))
    df_maldi = get_all_experiments_dataframe(partial(read_maldi, group_enterobacterales=group_enterobacterales), exps, path_data+'MALDI/')
    df_maldi = dtf.drop_column(df_maldi, ['M2'] + _media_to_drop(media))
    df_maldi, _ = dtf.preprocess_dataframe(df_maldi, cutoff=0., cutoff_prop=0., calc_prop=False)

    df_ngs = get_all_experiments_dataframe(partial(read_ngs, drop_taxa=drop_taxa, group_enterobacterales=group_enterobacterales),
                                           exps, path_data+'NGS/')
    df_ngs = dtf.drop_column(df_ngs, ['M2'])
    df_ngs, _ = dtf.preprocess_dataframe(df_ngs, cutoff=0., cutoff_prop=0., calc_prop=False)

    df_maldi = dtf.restore_missing_rows(df_maldi, bact_all)
    df_ngs = dtf.restore_missing_rows(df_ngs, bact_all)
    df_ngs = df_ngs.T[bact_all].T
    df_maldi = df_maldi.T[bact_all].T
    #df_maldi, bact_all = add_rest_bacteria(df_maldi, bact_all)
    #df_ngs, bact_all = add_rest_bacteria(df_ngs, bact_all)

    for df, meas in zip([df_platecount, df_maldi, df_ngs], ['platecount', 'maldi', 'ngs']):
        df.to_pickle(path_out+f'dataframe_{meas}{add_name}.pkl')
    df_platecount.to_csv(path_out+f'dataframe_platecount{add_name}.csv', index=False)
    df_maldi.to_csv(path_out+f'dataframe_maldi{add_name}.csv')
    df_ngs.to_csv(path_out+f'dataframe_ngs{add_name}.csv')
    return [df_platecount, df_maldi, df_ngs]


def _get_exp_filename(exp_name):
    """Convert a short experiment identifier to its source-file naming convention."""
    # Redefine experiment names to read data
    if exp_name[0] == 'V':
        exp_filename = 'Experiment_' + exp_name[1:]
    elif exp_name[0] == 'P':
        exp_filename = 'Pilot_' + exp_name[1:]
    return exp_filename


def _correct_df_for_tempinterruption(df):
    """Add temperature-interruption variants and remove duplicate baseline columns."""
    names_init = set([col.split('_')[0] for col in df.columns])
    for Tshift in ['CCD01', 'CCD02', 'RP']:
        if np.size((df.filter(regex=Tshift))) != 0:
            for day in ['_00_', '_01_']:
                for f in df.filter(like=day).columns:
                    fspl = f.split('_')
                    fspl[0] = '-'.join([fspl[0], Tshift])
                    df["_".join(fspl)] = df[f]
    names_fin = set([col.split('_')[0] for col in df.filter(like='_00_').filter(like='PC').columns])
    df = drop_column(df, [' ']+[value+'_' for value in names_fin if value not in names_init])
    return df


def read_platecount(exp_name, dir_data=''):
    """Read and normalize colony-count measurements for one experiment."""
    exp_temps = EXP_TEMPS
    exp_filename = _get_exp_filename(exp_name)
    # Read the file:
    filename = f'Data on colony counts - {exp_filename}.xlsx'
    if int(exp_name.split('_')[0][1:]) >= 14:
        df0 = pd.read_excel(dir_data + filename, keep_default_na=True, usecols='A:G')
    else:
        df0 = pd.read_excel(dir_data + filename, keep_default_na=True) # , on_bad_lines='skip'
    df0 = df0.dropna(axis=1, how='all')[-np.isnan(df0.Average)]

    # Several rows can belong to one data point: on day 0, the subsamples CC01, CC02, ... of the same meat. The value
    # is the mean of their means (each row: mean of its replicate plates), the standard deviation that of all plates.
    rows = {}
    for f in df0[df0.columns[0]].values:
        fspl = f.split('_')
        if len(fspl) == 8:
            f1 = fspl[0]
        elif len(fspl) == 9:
            f1 = "-".join([fspl[0], fspl[3]])
        elif len(fspl) == 10:
            if fspl[4] == 'REPACK':
                fspl[4] = 'RP'
            f1 = "-".join([fspl[0], fspl[3], fspl[4]])
        elif len(fspl) == 11:            
            if fspl[4] == 'REPACK':
                f1 = "-".join([fspl[0], fspl[3], fspl[5], 'RP'])
            else:
                f1 = "-".join([fspl[0], fspl[3], fspl[4], fspl[5]])
        df_select = df0[df0[df0.columns[0]] == f].iloc[:, 2:]
        df_select = df_select.values[~np.isnan(df_select.values)]

        name = "-".join(["_".join([f1, fspl[2], f"{int(exp_temps[fspl[0][:3]]):02d}C", fspl[-4], fspl[-1][:-2]]), 'platecount'])
        rows.setdefault(name, []).append(df_select)

    d = {'Properties': ['Average', 'Standard deviation', 'Replicas']}
    for name, plates in rows.items():
        all_plates = np.concatenate(plates)
        d[name] = [np.mean([np.mean(x) for x in plates]), np.std(all_plates), all_plates]
    df = _correct_df_for_tempinterruption(pd.DataFrame(data=d))
    return df.groupby('Properties').sum()


def read_maldi(exp_name, dir_data='', counts=False, group_enterobacterales=False):
    """Read MALDI isolate-identification data and return genus proportions by sample (counts=True: identified colonies).
    group_enterobacterales: the genera of ENTEROBACTERALES_GENERA summed into one taxon 'Enterobacterales'."""
    filename =  f'Data on bacteria identified by MALDI-Biotyper - {_get_exp_filename(exp_name)}'
    media_filenames = ['_MRS_Aerob_wachsende_Milchsäurebakterien', '_PC_Aerobe_Keimzahl (Gesamtkeimzahl)', '_VRBD_Enterobacterales']
    if int(exp_name[1:3]) < 15:
        df_maldi = pd.read_excel(dir_data + filename+ '.xlsx', keep_default_na=True)
        df_maldi =  _df_maldi_transform(df_maldi, exp_name, dir_data, counts=counts)
    elif int(exp_name[1:3]) == 15:
        df_maldi = pd.read_excel(dir_data + filename + '.xlsx', keep_default_na=True, skiprows=1)
        df_maldi = _df_maldi_transform(df_maldi, exp_name, dir_data, counts=counts)
    elif int(exp_name[1:3]) >= 16 and int(exp_name[1:3])<=19:
        df_maldi = []
        for media in media_filenames:
            df = pd.read_excel(dir_data + filename+media + '.xlsx', keep_default_na=True)
            df_maldi.append(_df_maldi_transform(df, exp_name, dir_data, counts=counts))
        df_maldi = merge_dfs(df_maldi)
    else:
        df_maldi = pd.read_excel(dir_data + filename+media_filenames[1] + '.xlsx', keep_default_na=True)
        df_maldi = _df_maldi_transform(df_maldi, exp_name, dir_data, counts=counts)
    if group_enterobacterales:
        df_maldi = df_maldi.groupby([ENTEROBACTERALES if g in ENTEROBACTERALES_GENERA else g for g in df_maldi.index]).sum()
        df_maldi.index.name = 'Genus'
    return df_maldi
        

def _df_maldi_transform(df, exp_name, dir_data='', counts=False):
    """Transform a raw MALDI spreadsheet into genus-by-sample proportions (counts=True: numbers of identified colonies)."""
    df=df.rename(columns={"Unnamed: 0": "Species", "Taxa": "Species"})
    df=df.sort_values(by = "Species").reset_index(drop=True)
    df = df[df['Species'].isna()==False]  # noqa: E712
    df.insert(0, "Genus", [sp.strip().split()[0] for sp in df["Species"].values])
    df = df.dropna(axis=1, how='all').drop(columns=['Species'])
    df['Genus'] = ['Leuconostoc' if f=='Leucononstoc' else f for f in df['Genus'].values]
    #Leucononstoc carnosum_MRI_FLM270
    df = df.groupby('Genus').sum()
    # Get undentified isolates:
    #df_unid = read_pkl('MALDI_unidentified.pkl', dir_data).filter(like=exp_name)
    # for now drop Experiments 10-16
    #df = merge_dfs([df, df_unid])

    # Rename samples
    for f in df.__iter__():
        df = df.rename(columns={f:_rename_maldi_samples(f)})
    df = df.T.groupby(by=df.columns).sum().T

    # Get M1/M2 indentificators instead of M0
    for f in df.columns:
        if f.split('_')[3] == '00':
            df["_".join([ff if i!=1 else 'M1' for i, ff in enumerate(f.split('_'))])] = df[f].values
            df = df.rename(columns={f:"_".join([ff if i!=1 else 'M2' for i, ff in enumerate(f.split('_'))])})
    df = _correct_df_for_tempinterruption(df)
    return df if counts else df.apply(calc_proportion)


def _rename_maldi_samples(f):
    """Convert a raw MALDI sample label into the standardized column format."""
    fspl = f.split('_')
    if fspl[-1] == 'IDENExp01':
        fspl = fspl[:-1]
    if len(fspl) == 8:
        return  "-".join(["_".join([fspl[0], fspl[2], fspl[-5], fspl[-4], fspl[-1][:-2]]), 'maldi'])
    elif len(fspl) == 9:
        f1 = "-".join([fspl[0], fspl[3]])
    elif len(fspl) == 10:
        f1 = "-".join([fspl[0], fspl[3], fspl[4]])
    elif len(fspl) == 11:
        f1 = "-".join([fspl[0], fspl[3], fspl[4], fspl[5]])
    return "-".join(["_".join([f1, fspl[2], fspl[-5], fspl[-4], fspl[-1][:-2]]), 'maldi'])


def read_ngs(exp_name, counts=False, drop_taxa=(), **kwargs):
    """Read NGS data using the file format appropriate for the experiment number.
    drop_taxa: taxa whose reads are removed before the proportions are computed (default none);
    group_enterobacterales (keyword, passed on): all Enterobacterales reads as one taxon, see ENTEROBACTERALES_GENERA."""
    reader = read_ngs1 if int(exp_name[1:]) <= 7 else read_ngs2
    df = reader(exp_name, counts=True, **kwargs)
    df = df.drop(index=[t for t in drop_taxa if t in df.index])
    return df if counts else df.apply(calc_proportion)


def read_ngs1(exp_name, dir_data='', counts=False, group_enterobacterales=False):
    """Read the earlier NGS table format and aggregate counts into genus proportions."""
    exp_temps = EXP_TEMPS
    temp = exp_temps[exp_name[:3]]
    filename = f'ZOTU_Table_{exp_name}.tsv'
    df = pd.read_csv(dir_data + filename, sep='\t', header=[0, 1, 2, 3], keep_default_na=True, encoding="ISO-8859-1")
    df = df.dropna(axis=1, how='all')
    df=df.rename(columns={"70% N2 + 30% CO2": "M2", "70% O2 + 30% CO2": "M1"})
    d = {"Genus": [_ngs_genus(bact, group_enterobacterales) for bact in df[df.columns[1]].values]}
    for f in df.filter(like='M').columns:
        d["_".join([exp_name, f[0], f'{int(temp):02d}C', f'{int(f[1]):02d}', 'ngs'])] = df[f].values
    df = pd.DataFrame(data=d)
    df = df[df.Genus!='']
    df = df.groupby('Genus').sum()
    return df if counts else df.apply(calc_proportion)


def read_ngs2(exp_name, dir_data='', counts=False, group_enterobacterales=False):
    """Read the later NGS table format and aggregate counts into genus proportions."""
    filename = f'ZOTU_Table_{exp_name.split("-")[0]}.tsv'
    if int(exp_name[1:])<12 and int(exp_name[1:])>=8:
        df = pd.read_csv(dir_data + filename, sep='\t', header=[0, 1, 2, 3, 4], skiprows=[5, 6],  keep_default_na=True, encoding="ISO-8859-1")
    elif int(exp_name[1:])>=12:
        df = pd.read_csv(dir_data + filename, sep='\t', header=[0, 1, 2, 3, 4, 5, 6], skiprows=[7, 8],  keep_default_na=True, encoding="ISO-8859-1")

    df = df.drop(columns=[df.columns[1]]).dropna(axis=1, how='all')
    d = {"Genus": [_ngs_genus(bact, group_enterobacterales) for bact in df[df.columns[0]].values]}
    for f in df.filter(like='V').__iter__():
        if f[2].strip() == 'O2':
            atm = 'M1'
        elif f[2].strip() == 'N2':
            atm = 'M2'
        if f[4].strip()=='6h':
            exp_name_new = "-".join([exp_name, 'CCD01'])
        elif f[4].strip()=='12h':
            exp_name_new = "-".join([exp_name, 'CCD02'])
        elif f[-2].strip()=='SHL1' or f[-2].strip()=='SHL2':
            exp_name_new = "-".join([exp_name, f[-2].strip()])
        elif f[-1].strip() == '7d':
            exp_name_new = "-".join([exp_name, 'STOR01'])
        else:
            exp_name_new = exp_name    
        d["_".join([exp_name_new, atm, f'{int(f[1].strip()[0]):02d}C', f[3][1:], 'ngs'])] = df[f].values
    df = pd.DataFrame(data=d)
    df = df[df.Genus!='']
    df = _correct_df_for_tempinterruption(df)
    df = df.groupby('Genus').sum()
    return df if counts else df.apply(calc_proportion)


def read_biochem(folder_path, username, password=None):
    """Download and organize biochemical measurements from OpenBIS."""
    o = login_openbis(username, password)
    d = {'Measurement': ['O2', 'CO2']}
    for s in o.get_samples(space='ZL2030', project='WORKPACKAGE_1.2', experiment='*GAS*'):
        d[s.props.sample_name] = [float(s.props.gas_measurement_oxygen), float(s.props.gas_measurement_carbondioxid)]
    df_gas =  pd.DataFrame(data=d).groupby('Measurement').mean()
    df_gas = _rename_df_biochem(df_gas)
    df_gas.to_pickle(folder_path+'biochem_o2_co2.pkl')

    d = {'Measurement': ['TBARS', 'POZ']}
    for s in o.get_samples(space='ZL2030', project='WORKPACKAGE_1.2', experiment='*TBARS_POZ*'):
        if s.props.tbars_measurement_mean is not None:
            d[s.props.sample_name] = [float(s.props.tbars_measurement_mean), float(s.props.poz_measurement_mean)]
    df_tb =  pd.DataFrame(data=d).groupby('Measurement').mean()
    df_tb = _rename_df_biochem(df_tb)

    d = {'Measurement': ['pH']}
    for s in o.get_samples(space='ZL2030', project='WORKPACKAGE_1.2', experiment='*PH_AW*'):
        if s.props.ph_measurement_mean is not None:
            d[s.props.sample_name] = [float(s.props.ph_measurement_mean)]
    df_ph =  pd.DataFrame(data=d).groupby('Measurement').mean()
    df_ph = _rename_df_biochem(df_ph)
    df = merge_dfs([df_tb, df_ph])
    df = df.replace(0, np.nan, inplace=False)
    df.to_pickle(folder_path+'biochem_oh_tbars_poz.pkl')
    return df, df_gas


def read_pkl(name, folder_path):
    """Load a pickled dataframe from the specified folder."""
    return pd.read_pickle(folder_path+name)


def _rename_df_biochem(df):
    """Standardize biochemical sample names and average duplicate columns."""
    for f in df.columns:
        fspl = f.split('_')
        if len(fspl) == 6:
            new_name = "_".join([fspl[0], fspl[2], fspl[-3], fspl[-2], 'biochem'])
        elif len(fspl) == 7:
            f1 = f1 = "-".join([fspl[0], fspl[3]])
            new_name = "_".join([f1, fspl[2], fspl[-3], fspl[-2], 'biochem'])
        df = df.rename(columns={f:new_name})
    return df.T.groupby(by=df.columns).mean().T


def read_color(exp_name, dir_data=''):
    """Read color measurements and return values grouped by property and sample."""
    exp_temps = EXP_TEMPS
    filename = f'Data of color measurement main trial {int(exp_name[1:])} KI.xlsx'
    df0 = pd.read_excel(dir_data + filename, keep_default_na=True, header=[0])

    d = {'Properties': df0.columns[2:]}
    for f in df0[df0.columns[0]].values:
        fspl = f.split('_')
        if len(fspl) == 9 or (len(fspl) == 8 and fspl[3][-1] != 'C'):
            f1 = "-".join([fspl[0], fspl[3]])
            day = fspl[5]
        else:
            f1 = fspl[0]
            day = fspl[4]

        df_select = df0[df0[df0.columns[0]] == f].iloc[:, 2:].mean()
        df_select = df_select.values[~np.isnan(df_select.values)]
    
        if fspl[-2][:3] != 'MRI':
            meas_name = '-'.join([fspl[-2], 'CLR'])
        else:    
            meas_name = 'CLR'
        if fspl[-4] != '00':
            d["_".join([f1, fspl[2], f"{int(exp_temps[fspl[0][:3]]):02d}C", day, meas_name])] = df_select
        else:
            for media in ['M1', 'M2']:
                d["_".join([f1, fspl[2], f"{int(exp_temps[fspl[0][:3]]):02d}C", day, meas_name])] = df_select
    df = _correct_df_for_tempinterruption(pd.DataFrame(data=d))
    return df.groupby('Properties').mean()


def get_all_experiments_dataframe(read_func, exp_names, dir_data):
    """Read multiple experiments and merge their dataframes by row index."""
    return merge_dfs([read_func(exp, dir_data=dir_data) for exp in exp_names])


def login_openbis(username, password=None):
    """Log in to OpenBIS, prompting for a password when one is not supplied."""
    o = Openbis('https://openbis.zl2030.de/openbis/webapp/eln-lims/')
    if password is None:
        password = getpass.getpass()
    o.login(username, password, save_token=True)
    return o


def download_from_openbis(o, folder_path, workpackage, experiment_name, type_data=None):
    """Download matching OpenBIS datasets into the destination folder."""
    for dataset in o.get_datasets(space='ZL2030', project=workpackage, experiment=experiment_name, type=type_data):
        dataset.download(destination=folder_path, create_default_folders=False)

def get_MALDI_metadata(o, folder_path, workpackage, experiment_name):
    """Save the unidentified-isolate counts for matching MALDI samples."""
    d = {'Genus': 'unidentified'}
    for s in  o.get_samples(space='ZL2030', project=workpackage, experiment=experiment_name):#,
                            #where={'experimental_subsample_name':'*V0*'}): ## For now just exp till 9
        if s.props.maldi_isolates_unidentified is None:
            unid = 0.
        else:
            unid = float(s.props.maldi_isolates_unidentified)
        d[s.props.experimental_subsample_name] = [unid] #s.props.maldi_isolates_analysed
    df =  pd.DataFrame(data=d).groupby('Genus').sum()
    df.to_pickle(folder_path+'MALDI_unidentified.pkl')
        

# Download excels with microbiological data from Openbis
def download_microbiology_from_openbis(folder_path, username, password=None):
    """Download colony-count datasets from OpenBIS and close the session."""
    o = login_openbis(username, password)
    download_from_openbis(o, folder_path, 'WORKPACKAGE_1.2', '*COLONY_COUNT*')
    o.logout()

def download_MALDI_from_openbis(folder_path, username, password=None):
    """Download MALDI datasets and sample metadata from OpenBIS."""
    o = login_openbis(username, password)
    download_from_openbis(o, folder_path, 'WORKPACKAGE_1.2', '*MALDI_BIOTYPER*')
    get_MALDI_metadata(o, folder_path, 'WORKPACKAGE_1.2', '*MALDI_BIOTYPER*')
    o.logout()
    
def download_NGS_from_openbis(folder_path, username, password=None):
    """Download NGS datasets from OpenBIS and close the session."""
    o = login_openbis(username, password)
    download_from_openbis(o, folder_path, 'WORKPACKAGE_1.5', '*AMPLICON_SEQ*', 'PROCESSED_DATA')
    o.logout()

def download_colordata_from_openbis(folder_path, username, password=None):
    """Download color-measurement datasets from OpenBIS and close the session."""
    o = login_openbis(username, password)
    download_from_openbis(o, folder_path, 'WORKPACKAGE_1.2', '*COLOR*')
    o.logout()
# ZL2030 data modelling

Modelling of the [Zukunftslabor2030 (ZL2030)](https://zukunftslabor2030.de) meat storage experiments with the core library `fusion_model` from [MDFM](https://github.com/polinagaindrik/MDFM): reading the measurements, calibrating the model, predicting new experiments and plotting the results.

**This repository contains code only.** The measurements and results are not in git and must not be added, see [Data](#data).

## Setup

The repositories `MDFM` and `zl2030data-modeling` are expected in the same folder:

```
fusion_model/
    MDFM/
    zl2030data-modeling/
```

Create the virtual environment from the root of this repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs `fusion_model` from the neighbouring folder, so changes in `MDFM` take effect without reinstalling, and also PyBIS, the JAX backend and numba. It needs Python 3.11 or newer. The figures need a LaTeX installation with `dvipng`.

This installs JAX for the CPU, which is what the multistart fit needs: one process per core would run out of memory on a single GPU.

## Data

| Folder | Content | In git |
|---|---|---|
| `experiments/` | Plate counts, MALDI, NGS, biochemistry and color measurements | No |
| `inputs_fusionmodel/exp_temps.json` | Temperature profiles of the interruption experiments, for the prediction scripts | No |
| `out/` | Results and figures | No |

The measurements are stored in the project's OpenBIS and need a ZL2030 login. Download them to `experiments/` with the `download_*_from_openbis` functions of `zl2030.read_data`, which ask for the password. Never write passwords into the scripts.

`.gitignore` excludes these folders and the usual data formats. Check `git status` before committing.

## Run

Run all scripts from the root of this repository, because the paths are relative to it:

```sh
python scripts/main_ZL2030_withS.py
```

In a script, `import zl2030 as zl` gives the functions of this repository, for example `zl.prepare_ZL2030_data` and `zl.EXP_TEMPS`, and `import fusion_model as fm` the core library.

## Multistart fit

`scripts/main_zl2030_multistart.py` fits the 12-species model and bounds of the in-silico case to the data, to compare with the earlier parameter set. It uses all cores and can take up to two days, so run it as a background service:

```sh
systemd-run --user --unit=zl2030-multistart --remain-after-exit --working-directory=$PWD \
    -p StandardOutput=append:$PWD/out/run_multistart.log -p StandardError=append:$PWD/out/run_multistart.log \
    --setenv=PYTHONUNBUFFERED=1 $PWD/.venv/bin/python scripts/main_zl2030_multistart.py --log-k
```

With `--log-k`, results go to `out/multistart_logk/`: start 0 continues the best point of the first run (`out/multistart/`), and 47 independent starts have the interaction terms `k_ij` sampled on a log scale. Each start has at most 20 h, so a run takes at most about 40 h on 24 cores. Without it, results go to `out/multistart/`: the Latin hypercube samples `k_ij` linearly, start 0 is the earlier parameter set, and each start has at most 4 h. A new run overwrites the histories in its folder.

While it runs or afterwards, add `--log-k` for the second kind of run:

```sh
python scripts/show_multistart_progress.py      # cost of each start
python scripts/plot_zl2030_multistart_best.py   # figures of the best fit so far
touch out/multistart/optimization_history_multistart.stop   # stop, keep the best points and write the results
```

## Contents

| Path | Content |
|---|---|
| `zl2030/read_data.py` | OpenBIS download, reading of the data, temperatures of the experiments (`EXP_TEMPS`) |
| `zl2030/optimize.py` | Parameter bounds and calibration setup of the real data |
| `zl2030/insilico_from_fit.py` | In-silico data generated from a calibration result |
| `zl2030/plotting.py` | Figures of the measurements |
| `scripts/main_ZL2030_withS.py`, `scripts/main_ZL2030_woS.py` | Calibration, with and without estimating the media matrix `S` |
| `scripts/main_ZL2030_prediction*.py` | Prediction of new experiments |
| `scripts/main_zl2030_multistart.py` | Multistart fit, see above |
| `scripts/show_multistart_progress.py`, `scripts/plot_zl2030_multistart_best.py` | Progress and figures of the multistart fit |
| `scripts/create_exp_temps_dict.py` | Creates `inputs_fusionmodel/exp_temps.json` |
| `plot_results/` | Further figures of data and results |
| `old/` | Earlier scripts, not maintained |

## Notes on the data

- **Temperatures.** V03 is 14 °C, although its colony count sample names say `02C`: its sampling days and growth are those of the 14 °C runs and its MALDI and NGS samples are labelled 14C. `EXP_TEMPS` has the correct value, so do not read temperatures from those sample names.
- **Constant temperature.** The calibration uses the constant-temperature series only: the filter `V.._` drops the variants such as `V11-CCD02` (chill-chain interruption) and `V16-STOR01` (stored meat), which remain available for validation.
- **Atmosphere.** Only M1 (70 % O2 + 30 % CO2) is used, M0 is the sample of day 0.

## License

The code is distributed under the MIT License, see [`LICENSE`](LICENSE). The license does not cover the measurements or results.

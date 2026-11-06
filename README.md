# ZL2030 data modelling

Modelling of the ZL2030 meat storage experiments with the core library `fusion_model` ([MDFM](https://github.com/polinagaindrik/MDFM)).

## Setup

The repositories `MDFM` and `zl2030data-modeling` are expected in the same folder. Run everything from the root of this repository.

```sh
pip install -r requirements.txt
```

The data (`experiments/`) and results (`out/`) are not in git. `experiments/` comes from OpenBIS, see `zl2030.read_data`.

## Contents

| Path | Content |
|---|---|
| `zl2030/` | Reading of the data, temperatures of the experiments, parameter bounds, in-silico data from a fit, figures |
| `scripts/main_ZL2030_*.py` | Calibration and prediction |
| `scripts/main_zl2030_multistart.py` | Multistart fit of the 12-species in-silico model; `--log-k`: inhibition terms sampled on a log scale |
| `scripts/show_multistart_progress.py`, `scripts/plot_zl2030_multistart_best.py` | Progress and figures of the multistart fit, also while it runs |
| `plot_results/` | Figures of data and results |
| `old/` | Earlier scripts |

## Temperatures

`zl2030.EXP_TEMPS` has the storage temperature of each experiment. V03 is 14 degC, although its colony count sample names say 02C: its sampling days and growth are those of the 14 degC runs and its MALDI and NGS samples are labelled 14C.

The prediction scripts also need the temperature profiles of the interruption experiments in `inputs_fusionmodel/exp_temps.json`, see `scripts/create_exp_temps_dict.py`.

# ELM Emulator

Cleaned package for ELM canopy-flux emulator training and coupled site/regional replay.

This repo was split from `surface_emulator` to keep the reusable runtime pieces separate from
one-off analyses, footprint experiments, and plotting notebooks/scripts.

## What Is Included

- `elm_emulator/benchmark_replay.py`
  - site forcing replay through `--regional-met-format site`
  - TESSFA/TRENDY regional replay
  - single-cell, synthetic clone-grid, treatment-perturbation, daily/5-day/monthly NetCDF output
- `elm_emulator/train_patch_models.py`
  - patch-specific and universal canopy-flux MLP training
- `elm_emulator/coupled_surface_hydrology_soiltemp.py`
  - coupled surface surrogate, hydrology, soil temperature, snow, and optional carbon support
- `elm_emulator/elm_*`, `elm_emulator/btran_utils.py`
  - physical helper modules required by replay/training
- `artifacts/canopyflux_models/`
  - curated `model.pt` and `metrics.json` artifacts copied from the old tree
- `data/surface_data/`
  - small/medium default surfdata files
- `data/params/`
  - small CLM/ELM parameter files used by regional replay
- `data/site/usmoz/`
  - US-MOz point `surfdata.nc` and `clm_params.nc` used by the site replay recipe

Large local data are intentionally not copied:

- `surface_emulator/20260519_sg256_ICBELMBC_emulator.canopyflux_training.nc`
- `surface_emulator/surface_data/surfdata_hcru_TRENDY.nc`
- external meteorological forcing under `/Users/zdr/models/inputdata`

## Install

Editable install from the repo root:

```bash
cd /Users/zdr/models/ELM_emulator
python -m pip install -e .
```

Direct module use also works without installation:

```bash
cd /Users/zdr/models/ELM_emulator
python -m elm_emulator replay --help
python benchmark_replay.py --help
```

## Commands

Top-level dispatcher:

```bash
elm-emulator replay --help
elm-emulator train --help
elm-emulator coupled-benchmark --help
```

Direct entry points:

```bash
elm-replay --help
elm-train-patch-models --help
elm-coupled-benchmark --help
```

## US-MOz Site / Clone-Grid Replay

The cleaned run recipe is:

```bash
DEVICE=mps GRID_ROWS=20 GRID_COLS=20 CORR_LENGTH_CELLS=5 \
  scripts/run_usmoz_site_clonegrid.sh
```

Use CPU for larger runs when it is faster on Apple Silicon:

```bash
DEVICE=cpu GRID_ROWS=100 GRID_COLS=100 CORR_LENGTH_CELLS=25 \
  OUTPUT_FREQUENCY=5day N_YEARS=2 START_YEAR=2011 \
  scripts/run_usmoz_site_clonegrid.sh
```

The default US-MOz forcing path is:

```text
/Users/zdr/models/inputdata/atm/datm7/CLM1PT_data/1x1pt_US-MOz
```

Override with `MET_DIR=/path/to/site_forcing`.

The US-MOz point `surfdata.nc` and `clm_params.nc` defaults are repo-local under
`data/site/usmoz/`.

## Regional Replay

Minimal regional command:

```bash
python -m elm_emulator replay \
  --regional \
  --universal \
  --device cpu \
  --year 1980 \
  --month 1 \
  --nmonths 1 \
  --regional-ncell 100 \
  --met-dir /path/to/met_forcing \
  --regional-daily-output output/regional_smoke_monthly.nc \
  --regional-output-frequency monthly \
  --no-plot \
  --no-regional-cell-timeseries
```

Or use the smoke-test recipe:

```bash
MET_DIR=/path/to/met_forcing scripts/run_regional_smoke.sh
```

For the local 0.5-degree CRUJRA/TRENDY forcing and HCRU TRENDY surfdata, this
CONUS smoke test is preconfigured:

```bash
scripts/run_conus_trendy_smoke.sh
```

Defaults:

```text
MET_DIR=/Users/zdr/models/inputdata/atm/datm7/atm_forcing.CRUJRA_trendy_2025/cpl_bypass_full
SURFDATA=/Users/zdr/models/surface_emulator/surface_data/surfdata_hcru_TRENDY.nc
REGIONAL_MIN_LON=-125
REGIONAL_MAX_LON=-66
REGIONAL_MIN_LAT=24
REGIONAL_MAX_LAT=50
MAX_HOURS=24
```

For a true bounded regional subset, specify the forcing directory, surfdata, and bounds:

```bash
MET_DIR=/path/to/tessfa_or_trendy_met \
SURFDATA=data/surface_data/surfdata.TESSFA_DOMAIN2.4km.1d.NLCD.c240827.nc4 \
REGIONAL_MIN_LON=-94 REGIONAL_MAX_LON=-90 \
REGIONAL_MIN_LAT=36 REGIONAL_MAX_LAT=40 \
REGIONAL_NCELL=1000 \
scripts/run_regional_smoke.sh
```

Equivalent direct command:

```bash
python -m elm_emulator replay \
  --regional \
  --universal \
  --met-dir /path/to/tessfa_or_trendy_met \
  --regional-met-format auto \
  --regional-surfdata data/surface_data/surfdata.TESSFA_DOMAIN2.4km.1d.NLCD.c240827.nc4 \
  --regional-min-lon -94 \
  --regional-max-lon -90 \
  --regional-min-lat 36 \
  --regional-max-lat 40 \
  --regional-ncell 1000 \
  --year 1980 \
  --month 1 \
  --nmonths 1 \
  --device cpu \
  --regional-daily-output output/regional_subset_monthly.nc \
  --regional-output-frequency monthly \
  --no-plot \
  --no-regional-cell-timeseries
```

Set `--regional-min-lon -180` when you want to disable the historical default
Southeast-only longitude filter.

Regional met prefetch is enabled by default. Use `--no-regional-prefetch` only
when debugging or when memory pressure from concurrent month loading is a problem.

The default model, surfdata, and parameter paths are repo-local:

```text
artifacts/canopyflux_models
data/surface_data/surfdata.TESSFA_DOMAIN2.4km.1d.NLCD.c240827.nc4
data/params/clm_params_SP.nc
```

## Training

Training data are external by default:

```bash
python -m elm_emulator train \
  /Users/zdr/models/surface_emulator/20260519_sg256_ICBELMBC_emulator.canopyflux_training.nc \
  --surfdata-file data/surface_data/surfdata_sg256.nc \
  --output-dir artifacts/canopyflux_models \
  --universal-model
```

Use a separate output directory for experiments to avoid overwriting curated model artifacts.

## Device Notes

- `DEVICE=cpu` is often faster than `mps` for moderate regional/clone-grid runs on Apple Silicon.
- Apple MPS does not support float64 tensors. Use `--dtype float32` and `--mlp-dtype same` or `float32`.
- CUDA runs can use `--mlp-dtype bfloat16` for Tensor Core MLP inference while keeping physics in float32.

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/../envs/torch/bin/python}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/artifacts/canopyflux_models_econus}"
NATIVE_RUN_DIR="${NATIVE_RUN_DIR:-${REPO_ROOT}/../elm-olmt/setup_runs/20260804_CentralTN_trendy_global_ICBELMBC_native_daily/run}"
SURFDATA="${SURFDATA:-${NATIVE_RUN_DIR}/surfdata.nc}"
CLM_PARAMS="${CLM_PARAMS:-${NATIVE_RUN_DIR}/clm_params.nc}"
MET_DIR="${MET_DIR:-${REPO_ROOT}/met_forcing/elmforc.TRENDY.c2025_0.5x0.5}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/output/central_tn_trendy_2004_2005}"

START_YEAR="${START_YEAR:-2004}"
N_YEARS="${N_YEARS:-2}"
NMONTHS="${NMONTHS:-$((N_YEARS * 12))}"
OUTPUT_FREQUENCY="${OUTPUT_FREQUENCY:-daily}"
LAT="${LAT:-35.75}"
LON="${LON:--86.25}"
CO2_FILE="${CO2_FILE:-${REPO_ROOT}/../e3sm_inputdata/atm/datm7/CO2/fco2_datm_rcp4.5_1765-2500_c130312.nc}"
CO2_PPM="${CO2_PPM:-}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/mplconfig" "${OUT_DIR}/tmp"
export MPLCONFIGDIR="${OUT_DIR}/mplconfig"
export TMPDIR="${OUT_DIR}/tmp"

for required in "${SURFDATA}" "${CLM_PARAMS}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: native OLMT setup file is missing: ${required}" >&2
    echo "Run elm-olmt/elm_olmt.py with Central-TN-trendy.local_setup.cfg first." >&2
    exit 2
  fi
done

if [[ -n "${CO2_FILE}" ]]; then
  if [[ ! -f "${CO2_FILE}" ]]; then
    echo "ERROR: CO2 time-series file is missing: ${CO2_FILE}" >&2
    exit 2
  fi
  CO2_ARGS=(--co2-file "${CO2_FILE}")
elif [[ -n "${CO2_PPM}" ]]; then
  CO2_ARGS=(--co2-ppm "${CO2_PPM}")
else
  echo "ERROR: set CO2_FILE to a time-series file or CO2_PPM to a scalar fallback" >&2
  exit 2
fi

"${PYTHON_BIN}" -m elm_emulator.benchmark_replay \
  --regional \
  --universal \
  --disable-snow \
  --hydrology-backend full \
  --soil-temperature-backend torch \
  --device "${DEVICE}" \
  --dtype float32 \
  --model-dir "${MODEL_DIR}" \
  --regional-surfdata "${SURFDATA}" \
  --clm-params "${CLM_PARAMS}" \
  --met-dir "${MET_DIR}" \
  --regional-met-format trendy \
  --regional-calendar noleap \
  --lat "${LAT}" \
  --lon "${LON}" \
  --year "${START_YEAR}" \
  --month 1 \
  --nmonths "${NMONTHS}" \
  "${CO2_ARGS[@]}" \
  --regional-output-frequency "${OUTPUT_FREQUENCY}" \
  --regional-daily-output "${OUT_DIR}/central_tn_trendy_emulator_${START_YEAR}_$((START_YEAR + N_YEARS - 1))_${OUTPUT_FREQUENCY}.nc" \
  --regional-progress-stride 168 \
  --regional-physics-step-hours 3 \
  --regional-prefetch \
  --no-regional-cell-timeseries \
  --no-plot \
  --no-compile \
  --no-compile-physics \
  "$@"

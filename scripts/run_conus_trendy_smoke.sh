#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/artifacts/canopyflux_models}"
SURFDATA="${SURFDATA:-/Users/zdr/models/surface_emulator/surface_data/surfdata_hcru_TRENDY.nc}"
MET_DIR="${MET_DIR:-/Users/zdr/models/inputdata/atm/datm7/atm_forcing.CRUJRA_trendy_2025/cpl_bypass_full}"
CLM_PARAMS="${CLM_PARAMS:-${REPO_ROOT}/data/params/clm_params_SP.nc}"

YEAR="${YEAR:-1980}"
MONTH="${MONTH:-1}"
NMONTHS="${NMONTHS:-1}"
MAX_HOURS="${MAX_HOURS:-24}"
DEVICE="${DEVICE:-cpu}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/output/conus_trendy_smoke}"
OUTPUT_FREQUENCY="${OUTPUT_FREQUENCY:-monthly}"
REGIONAL_MET_CACHE="${REGIONAL_MET_CACHE:-1}"

REGIONAL_MIN_LON="${REGIONAL_MIN_LON:--125}"
REGIONAL_MAX_LON="${REGIONAL_MAX_LON:--66}"
REGIONAL_MIN_LAT="${REGIONAL_MIN_LAT:-24}"
REGIONAL_MAX_LAT="${REGIONAL_MAX_LAT:-50}"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/plots" "${OUT_DIR}/mplconfig"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/mplconfig}"

CACHE_ARGS=()
case "${REGIONAL_MET_CACHE}" in
  1|true|TRUE|yes|YES) CACHE_ARGS+=(--regional-met-cache) ;;
  0|false|FALSE|no|NO) CACHE_ARGS+=(--no-regional-met-cache) ;;
  *) echo "ERROR: REGIONAL_MET_CACHE must be 0/1 or true/false, got: ${REGIONAL_MET_CACHE}" >&2; exit 2 ;;
esac

if [[ ! -f "${SURFDATA}" ]]; then
  echo "ERROR: missing TRENDY surfdata: ${SURFDATA}" >&2
  echo "       Set SURFDATA=/path/to/surfdata_hcru_TRENDY.nc" >&2
  exit 2
fi

if [[ ! -d "${MET_DIR}" ]]; then
  echo "ERROR: missing TRENDY met directory: ${MET_DIR}" >&2
  echo "       Set MET_DIR=/path/to/atm_forcing.CRUJRA_trendy_2025/cpl_bypass_full" >&2
  exit 2
fi

"${PYTHON_BIN}" -m elm_emulator.benchmark_replay \
  --regional \
  --universal \
  --device "${DEVICE}" \
  --dtype float32 \
  --model-dir "${MODEL_DIR}" \
  --regional-surfdata "${SURFDATA}" \
  --met-dir "${MET_DIR}" \
  --regional-met-format trendy \
  --regional-calendar noleap \
  --regional-min-lon "${REGIONAL_MIN_LON}" \
  --regional-max-lon "${REGIONAL_MAX_LON}" \
  --regional-min-lat "${REGIONAL_MIN_LAT}" \
  --regional-max-lat "${REGIONAL_MAX_LAT}" \
  --year "${YEAR}" \
  --month "${MONTH}" \
  --nmonths "${NMONTHS}" \
  --clm-params "${CLM_PARAMS}" \
  --regional-output-frequency "${OUTPUT_FREQUENCY}" \
  --regional-daily-output "${OUT_DIR}/conus_trendy_${YEAR}${MONTH}_h${MAX_HOURS}_${OUTPUT_FREQUENCY}.nc" \
  --regional-plot-dir "${OUT_DIR}/plots" \
  --regional-plot-variable gpp_gc_m2_d_mean eflx_lh_tot_mean qflx_runoff_tot_mean \
  "${CACHE_ARGS[@]}" \
  --regional-max-hours "${MAX_HOURS}" \
  --no-regional-cell-timeseries \
  --no-compile \
  --no-compile-physics \
  --disable-snow-model \
  "$@"

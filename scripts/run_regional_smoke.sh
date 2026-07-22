#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${MET_DIR:-}" ]]; then
  echo "ERROR: set MET_DIR to a directory containing site, TESSFA, or TRENDY forcing." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/artifacts/canopyflux_models}"
SURFDATA="${SURFDATA:-${REPO_ROOT}/data/surface_data/surfdata.TESSFA_DOMAIN2.4km.1d.NLCD.c240827.nc4}"
CLM_PARAMS="${CLM_PARAMS:-${REPO_ROOT}/data/params/clm_params_SP.nc}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/output/regional_smoke}"
DEVICE="${DEVICE:-cpu}"
YEAR="${YEAR:-1980}"
MONTH="${MONTH:-1}"
NMONTHS="${NMONTHS:-1}"
REGIONAL_NCELL="${REGIONAL_NCELL:-100}"
MET_FORMAT="${MET_FORMAT:-auto}"
OUTPUT_FREQUENCY="${OUTPUT_FREQUENCY:-monthly}"
REGIONAL_MIN_LON="${REGIONAL_MIN_LON:-}"
REGIONAL_MAX_LON="${REGIONAL_MAX_LON:-}"
REGIONAL_MIN_LAT="${REGIONAL_MIN_LAT:-}"
REGIONAL_MAX_LAT="${REGIONAL_MAX_LAT:-}"

mkdir -p "${OUT_DIR}"

BBOX_ARGS=()
if [[ -n "${REGIONAL_MIN_LON}" ]]; then
  BBOX_ARGS+=(--regional-min-lon "${REGIONAL_MIN_LON}")
fi
if [[ -n "${REGIONAL_MAX_LON}" ]]; then
  BBOX_ARGS+=(--regional-max-lon "${REGIONAL_MAX_LON}")
fi
if [[ -n "${REGIONAL_MIN_LAT}" ]]; then
  BBOX_ARGS+=(--regional-min-lat "${REGIONAL_MIN_LAT}")
fi
if [[ -n "${REGIONAL_MAX_LAT}" ]]; then
  BBOX_ARGS+=(--regional-max-lat "${REGIONAL_MAX_LAT}")
fi

CMD=(
  "${PYTHON_BIN}" -m elm_emulator.benchmark_replay \
  --regional \
  --universal \
  --device "${DEVICE}" \
  --dtype float32 \
  --model-dir "${MODEL_DIR}" \
  --regional-surfdata "${SURFDATA}" \
  --clm-params "${CLM_PARAMS}" \
  --met-dir "${MET_DIR}" \
  --regional-met-format "${MET_FORMAT}" \
  --year "${YEAR}" \
  --month "${MONTH}" \
  --nmonths "${NMONTHS}" \
)
if ((${#BBOX_ARGS[@]})); then
  CMD+=("${BBOX_ARGS[@]}")
fi
CMD+=(
  --regional-ncell "${REGIONAL_NCELL}" \
  --regional-daily-output "${OUT_DIR}/regional_smoke_${OUTPUT_FREQUENCY}.nc" \
  --regional-output-frequency "${OUTPUT_FREQUENCY}" \
  --no-plot \
  --no-regional-cell-timeseries \
  --no-compile \
  --no-compile-physics \
  "$@"
)

"${CMD[@]}"

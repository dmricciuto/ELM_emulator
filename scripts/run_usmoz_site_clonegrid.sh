#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_MET_DIR="/Users/zdr/models/inputdata/atm/datm7/CLM1PT_data/1x1pt_US-MOz"

MET_DIR="${MET_DIR:-${DEFAULT_MET_DIR}}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/artifacts/canopyflux_models}"
SURFDATA="${SURFDATA:-${REPO_ROOT}/data/site/usmoz/surfdata.nc}"
CLM_PARAMS="${CLM_PARAMS:-${REPO_ROOT}/data/site/usmoz/clm_params.nc}"
START_YEAR="${START_YEAR:-2004}"
N_YEARS="${N_YEARS:-5}"
NMONTHS="${NMONTHS:-$((N_YEARS * 12))}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/output/usmoz_site_clonegrid}"
PREFIX="${PREFIX:-usmoz_site}"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
DEVICE="${DEVICE:-auto}"
LAT="${LAT:-38.7441}"
LON="${LON:--92.2}"
PFT="${PFT:-7}"

GRID_ROWS="${GRID_ROWS:-200}"
GRID_COLS="${GRID_COLS:-200}"
CORR_LENGTH_CELLS="${CORR_LENGTH_CELLS:-20}"
FLNR_REL_SD="${FLNR_REL_SD:-0.20}"
MBBOPT_REL_SD="${MBBOPT_REL_SD:-0.20}"
SAND_ABS_SD="${SAND_ABS_SD:-8.0}"
GRID_SPACING_KM="${GRID_SPACING_KM:-4.0}"
CLONE_SEED="${CLONE_SEED:-42}"

OUTPUT_FREQUENCY="${OUTPUT_FREQUENCY:-monthly}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-${OUTPUT_FREQUENCY}}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_CLONE="${RUN_CLONE:-1}"
DISABLE_SNOW="${DISABLE_SNOW:-1}"
DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-1}"
PROFILE_SECTIONS="${PROFILE_SECTIONS:-0}"
REGIONAL_PREFETCH="${REGIONAL_PREFETCH:-1}"
DAY_NIGHT_FLUX_OUTPUT="${DAY_NIGHT_FLUX_OUTPUT:-0}"
DAY_NIGHT_FSDS_THRESHOLD="${DAY_NIGHT_FSDS_THRESHOLD:-20}"
MLP_DTYPE="${MLP_DTYPE:-same}"
TF32_MATMUL="${TF32_MATMUL:-high}"

PRECIP_TREATMENT_AREA_HA="${PRECIP_TREATMENT_AREA_HA:-0}"
PRECIP_TREATMENT_REDUCTION="${PRECIP_TREATMENT_REDUCTION:-0}"
PRECIP_TREATMENT_CELL_SIZE_M="${PRECIP_TREATMENT_CELL_SIZE_M:-20}"
PRECIP_TREATMENT_CENTER_X_M="${PRECIP_TREATMENT_CENTER_X_M:-0}"
PRECIP_TREATMENT_CENTER_Y_M="${PRECIP_TREATMENT_CENTER_Y_M:-0}"
PRECIP_TREATMENT_START_DATE="${PRECIP_TREATMENT_START_DATE:-}"
PRECIP_TREATMENT_END_DATE="${PRECIP_TREATMENT_END_DATE:-}"

TMPDIR="${TMPDIR:-${OUT_DIR}/tmp}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/mplconfig}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${REPO_ROOT}/cache/torchinductor}"
PLOT_VAR_ARGS=()
read -r -a PLOT_VAR_ARGS <<< "${PLOT_VARS:-gpp_gc_m2_d_mean eflx_lh_tot_mean eflx_sh_tot_mean qflx_runoff_tot_mean}"

case "${OUTPUT_FREQUENCY}" in
  daily|5day|monthly) ;;
  *)
    echo "ERROR: OUTPUT_FREQUENCY must be daily, 5day, or monthly, got: ${OUTPUT_FREQUENCY}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}" "${TORCHINDUCTOR_CACHE_DIR}"
export TMPDIR MPLCONFIGDIR TORCHINDUCTOR_CACHE_DIR

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: cannot find Python interpreter: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ "${DEVICE}" == "mps" ]]; then
  if ! "${PYTHON_BIN}" -c 'import torch; mps=getattr(torch.backends, "mps", None); raise SystemExit(0 if mps is not None and mps.is_available() else 1)' >/dev/null 2>&1; then
    echo "ERROR: DEVICE=mps was requested, but this Python cannot initialize Apple MPS." >&2
    exit 2
  fi
fi

if [[ "${DEVICE}" == "cuda" ]]; then
  if ! "${PYTHON_BIN}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "ERROR: DEVICE=cuda was requested, but this Python cannot initialize CUDA." >&2
    exit 2
  fi
fi

COMMON_ARGS=(
  --regional
  --universal
  --hydrology-backend full
  --soil-temperature-backend torch
  --model-dir "${MODEL_DIR}"
  --regional-surfdata "${SURFDATA}"
  --met-dir "${MET_DIR}"
  --regional-met-format site
  --regional-calendar noleap
  --year "${START_YEAR}"
  --month 1
  --nmonths "${NMONTHS}"
  --clm-params "${CLM_PARAMS}"
  --regional-output-frequency "${OUTPUT_FREQUENCY}"
  --dtype float32
  --mlp-dtype "${MLP_DTYPE}"
  --tf32-matmul "${TF32_MATMUL}"
  --regional-progress-stride 168
  --regional-physics-step-hours 3
  --lat "${LAT}"
  --lon "${LON}"
  --device "${DEVICE}"
)

case "${PFT}" in
  ""|all|ALL|none|NONE) ;;
  *) COMMON_ARGS+=(--pft "${PFT}") ;;
esac

case "${DISABLE_SNOW}" in
  1|true|TRUE|yes|YES) COMMON_ARGS+=(--disable-snow-model) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "ERROR: DISABLE_SNOW must be 0/1 or true/false, got: ${DISABLE_SNOW}" >&2; exit 2 ;;
esac

case "${DISABLE_TORCH_COMPILE}" in
  1|true|TRUE|yes|YES) COMMON_ARGS+=(--no-compile --no-compile-physics) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "ERROR: DISABLE_TORCH_COMPILE must be 0/1 or true/false, got: ${DISABLE_TORCH_COMPILE}" >&2; exit 2 ;;
esac

case "${PROFILE_SECTIONS}" in
  1|true|TRUE|yes|YES) COMMON_ARGS+=(--profile-sections) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "ERROR: PROFILE_SECTIONS must be 0/1 or true/false, got: ${PROFILE_SECTIONS}" >&2; exit 2 ;;
esac

case "${REGIONAL_PREFETCH}" in
  1|true|TRUE|yes|YES) COMMON_ARGS+=(--regional-prefetch) ;;
  0|false|FALSE|no|NO) COMMON_ARGS+=(--no-regional-prefetch) ;;
  *) echo "ERROR: REGIONAL_PREFETCH must be 0/1 or true/false, got: ${REGIONAL_PREFETCH}" >&2; exit 2 ;;
esac

case "${DAY_NIGHT_FLUX_OUTPUT}" in
  1|true|TRUE|yes|YES)
    COMMON_ARGS+=(--regional-day-night-flux-output --regional-day-night-fsds-threshold "${DAY_NIGHT_FSDS_THRESHOLD}")
    ;;
  0|false|FALSE|no|NO) ;;
  *) echo "ERROR: DAY_NIGHT_FLUX_OUTPUT must be 0/1 or true/false, got: ${DAY_NIGHT_FLUX_OUTPUT}" >&2; exit 2 ;;
esac

if [[ "${RUN_BASELINE}" == "1" || "${RUN_BASELINE}" == "true" || "${RUN_BASELINE}" == "TRUE" ]]; then
  echo "Running US-MOz baseline cell for ${NMONTHS} month(s)"
  "${PYTHON_BIN}" -m elm_emulator.benchmark_replay \
    "${COMMON_ARGS[@]}" \
    --regional-daily-output "${OUT_DIR}/${PREFIX}_moflux_cell_${OUTPUT_SUFFIX}.nc" \
    --no-plot \
    --no-regional-cell-timeseries \
    "$@"
fi

if [[ "${RUN_CLONE}" == "1" || "${RUN_CLONE}" == "true" || "${RUN_CLONE}" == "TRUE" ]]; then
  TREATMENT_ARGS=(
    --regional-precip-treatment-area-ha "${PRECIP_TREATMENT_AREA_HA}"
    --regional-precip-treatment-reduction "${PRECIP_TREATMENT_REDUCTION}"
    --regional-precip-treatment-cell-size-m "${PRECIP_TREATMENT_CELL_SIZE_M}"
    --regional-precip-treatment-center-x-m "${PRECIP_TREATMENT_CENTER_X_M}"
    --regional-precip-treatment-center-y-m "${PRECIP_TREATMENT_CENTER_Y_M}"
  )
  if [[ -n "${PRECIP_TREATMENT_START_DATE}" ]]; then
    TREATMENT_ARGS+=(--regional-precip-treatment-start-date "${PRECIP_TREATMENT_START_DATE}")
  fi
  if [[ -n "${PRECIP_TREATMENT_END_DATE}" ]]; then
    TREATMENT_ARGS+=(--regional-precip-treatment-end-date "${PRECIP_TREATMENT_END_DATE}")
  fi

  echo "Running US-MOz ${GRID_ROWS}x${GRID_COLS} cloned grid for ${NMONTHS} month(s)"
  "${PYTHON_BIN}" -m elm_emulator.benchmark_replay \
    "${COMMON_ARGS[@]}" \
    --regional-clone-grid-rows "${GRID_ROWS}" \
    --regional-clone-grid-cols "${GRID_COLS}" \
    --regional-clone-corr-length-cells "${CORR_LENGTH_CELLS}" \
    --regional-clone-flnr-relative-sd "${FLNR_REL_SD}" \
    --regional-clone-mbbopt-relative-sd "${MBBOPT_REL_SD}" \
    --regional-clone-sand-absolute-sd "${SAND_ABS_SD}" \
    --regional-clone-grid-spacing-km "${GRID_SPACING_KM}" \
    --regional-clone-seed "${CLONE_SEED}" \
    --regional-daily-output "${OUT_DIR}/${PREFIX}_clone_${GRID_ROWS}x${GRID_COLS}_${OUTPUT_SUFFIX}.nc" \
    --regional-plot-dir "${OUT_DIR}/plots_clone_${GRID_ROWS}x${GRID_COLS}" \
    --regional-plot-variable "${PLOT_VAR_ARGS[@]}" \
    "${TREATMENT_ARGS[@]}" \
    --no-regional-cell-timeseries \
    "$@"
fi

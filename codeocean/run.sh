#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# GeoPrior-v3 — Code Ocean run target
#
# This runner follows the reviewer workflow described in README.md:
# - root ./config.py is the reviewer-facing config
# - commands use the packaged CLI: `geoprior run ...`
# - outputs are written under ./results/
#
# Examples:
#   ./run.sh
#   ./run.sh "1,3,4"
#   ./run.sh "stage1,stage3"
#   GEOPRIOR_STAGES="1,2,4" ./run.sh
#
# Optional per-stage args via env:
#   STAGE1_ARGS="..."
#   STAGE2_ARGS="--set EPOCHS=5"
#   STAGE3_ARGS="--set EPOCHS=5"
#   STAGE4_ARGS="..."
#   STAGE5_ARGS="--city-a zhongshan --city-b nansha"
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${GEOPRIOR_CONFIG:-${ROOT_DIR}/config.py}"
RESULTS_DIR="${ROOT_DIR}/results"
OUTPUTS_DIR="${ROOT_DIR}/outputs"
LOG_DIR_DEFAULT="${OUTPUTS_DIR}/logs"
PRIMARY_DATA_DEFAULT="${ROOT_DIR}/data/zhongshan_full_city.csv"
SECONDARY_DATA_DEFAULT="${ROOT_DIR}/data/nansha_full_city.csv"

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export GEOPRIOR_LOG_PATH="${GEOPRIOR_LOG_PATH:-${LOG_DIR_DEFAULT}}"

mkdir -p "${RESULTS_DIR}" "${OUTPUTS_DIR}" "${GEOPRIOR_LOG_PATH}" "${ROOT_DIR}/data" "${ROOT_DIR}/nat.com"

show_help() {
  cat <<'EOF'
GeoPrior-v3 Code Ocean runner

Default behavior
  Runs the minimal reviewer workflow from README.md:
    1) Stage-1 preprocessing
    3) Stage-3 tuning with a short demo override (EPOCHS=5)

Usage
  ./run.sh
  ./run.sh "1,3"
  ./run.sh "1,2,4"
  ./run.sh "stage1,stage3"
  GEOPRIOR_STAGES="1,2,3,5" ./run.sh

Accepted stage tokens
  1, stage1, stage1-preprocess
  2, stage2, stage2-train
  3, stage3, stage3-tune
  4, stage4, stage4-infer
  5, stage5, stage5-transfer

Per-stage args via env
  STAGE1_ARGS="..."
  STAGE2_ARGS="..."   # default: --set EPOCHS=5
  STAGE3_ARGS="..."   # default: --set EPOCHS=5
  STAGE4_ARGS="..."
  STAGE5_ARGS="..."   # default: --city-a zhongshan --city-b nansha

Other env knobs
  GEOPRIOR_CONFIG=/path/to/config.py
  STAGE1_MANIFEST=/path/to/manifest.json
  GEOPRIOR_PRIMARY_DATA=data/zhongshan_full_city.csv
  GEOPRIOR_SECONDARY_DATA=data/nansha_full_city.csv
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "============================================================"
  echo "Missing reviewer-facing config file:"
  echo "  ${CONFIG_PATH}"
  echo
  echo "The new capsule README expects a root-level ./config.py."
  echo "============================================================"
  exit 2
fi

if ! command -v geoprior >/dev/null 2>&1; then
  echo "============================================================"
  echo "GeoPrior CLI is not installed or not on PATH."
  echo "Run postInstall.sh first, or install geoprior-v3."
  echo "============================================================"
  exit 2
fi

read_config_value() {
  local key="$1"
  python - "$CONFIG_PATH" "$key" <<'PY'
import pathlib
import sys

cfg_path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
ns = {}
try:
    exec(compile(cfg_path.read_text(encoding="utf-8"), str(cfg_path), "exec"), ns)
except Exception:
    print("")
    raise SystemExit(0)
value = ns.get(key, "")
if value is None:
    print("")
else:
    print(str(value).strip())
PY
}

CITY_NAME="$(read_config_value CITY_NAME)"
MODEL_NAME="$(read_config_value MODEL_NAME)"
CITY_NAME="${CITY_NAME:-zhongshan}"
MODEL_NAME="${MODEL_NAME:-GeoPriorSubsNet}"

PRIMARY_DATA="${GEOPRIOR_PRIMARY_DATA:-${PRIMARY_DATA_DEFAULT}}"
SECONDARY_DATA="${GEOPRIOR_SECONDARY_DATA:-${SECONDARY_DATA_DEFAULT}}"
STAGE1_MANIFEST="${STAGE1_MANIFEST:-${RESULTS_DIR}/${CITY_NAME}_${MODEL_NAME}_stage1/manifest.json}"
STAGES_RAW="${1:-${GEOPRIOR_STAGES:-1,3}}"

normalize_stage() {
  local raw="$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  case "${raw}" in
    1|stage1|s1|stage1-preprocess|preprocess|prepare)
      echo "1"
      ;;
    2|stage2|s2|stage2-train|train|fit)
      echo "2"
      ;;
    3|stage3|s3|stage3-tune|tune|tuning|search)
      echo "3"
      ;;
    4|stage4|s4|stage4-infer|infer|inference|predict|forecast)
      echo "4"
      ;;
    5|stage5|s5|stage5-transfer|transfer|xfer)
      echo "5"
      ;;
    *)
      echo ""
      ;;
  esac
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "============================================================"
    echo "Missing ${label}:"
    echo "  ${path}"
    echo "============================================================"
    exit 2
  fi
}

run_cli() {
  local subcmd="$1"
  local extra="${2:-}"
  echo "==> Running: geoprior run ${subcmd} --config ${CONFIG_PATH} ${extra}"
  if [[ -n "${extra}" ]]; then
    # shellcheck disable=SC2086
    geoprior run "${subcmd}" --config "${CONFIG_PATH}" ${extra}
  else
    geoprior run "${subcmd}" --config "${CONFIG_PATH}"
  fi
}

stage_default_args() {
  local stage="$1"
  case "${stage}" in
    2|3)
      echo "--set EPOCHS=5"
      ;;
    5)
      echo "--city-a zhongshan --city-b nansha"
      ;;
    *)
      echo ""
      ;;
  esac
}

choose_args() {
  local provided="$1"
  local fallback="$2"
  if [[ -n "${provided}" ]]; then
    echo "${provided}"
  else
    echo "${fallback}"
  fi
}

run_stage() {
  local stage="$1"
  local extra=""
  case "${stage}" in
    1)
      require_file "${PRIMARY_DATA}" "primary reviewer dataset"
      extra="${STAGE1_ARGS:-}"
      run_cli "stage1-preprocess" "${extra}"
      ;;
    2)
      require_file "${PRIMARY_DATA}" "primary reviewer dataset"
      require_file "${STAGE1_MANIFEST}" "Stage-1 manifest"
      extra="$(choose_args "${STAGE2_ARGS:-}" "$(stage_default_args 2)")"
      extra="--stage1-manifest ${STAGE1_MANIFEST} ${extra}"
      run_cli "stage2-train" "${extra}"
      ;;
    3)
      require_file "${PRIMARY_DATA}" "primary reviewer dataset"
      require_file "${STAGE1_MANIFEST}" "Stage-1 manifest"
      extra="$(choose_args "${STAGE3_ARGS:-}" "$(stage_default_args 3)")"
      extra="--stage1-manifest ${STAGE1_MANIFEST} ${extra}"
      run_cli "stage3-tune" "${extra}"
      ;;
    4)
      require_file "${PRIMARY_DATA}" "primary reviewer dataset"
      require_file "${STAGE1_MANIFEST}" "Stage-1 manifest"
      extra="--stage1-manifest ${STAGE1_MANIFEST} ${STAGE4_ARGS:-}"
      run_cli "stage4-infer" "${extra}"
      ;;
    5)
      require_file "${PRIMARY_DATA}" "primary reviewer dataset"
      require_file "${SECONDARY_DATA}" "secondary transfer dataset"
      extra="$(choose_args "${STAGE5_ARGS:-}" "$(stage_default_args 5)")"
      run_cli "stage5-transfer" "${extra}"
      ;;
    *)
      echo "!! Unknown stage token: ${stage}"
      show_help
      exit 2
      ;;
  esac
}

echo "============================================================"
echo "GeoPrior-v3 | Code Ocean run"
echo "Root          : ${ROOT_DIR}"
echo "Config        : ${CONFIG_PATH}"
echo "City          : ${CITY_NAME}"
echo "Model         : ${MODEL_NAME}"
echo "Stages        : ${STAGES_RAW}"
echo "Primary data  : ${PRIMARY_DATA}"
echo "Manifest      : ${STAGE1_MANIFEST}"
echo "Results       : ${RESULTS_DIR}"
echo "Logs          : ${GEOPRIOR_LOG_PATH}"
echo "============================================================"

IFS=',' read -r -a STAGE_LIST <<< "${STAGES_RAW}"
for raw in "${STAGE_LIST[@]}"; do
  stage="$(normalize_stage "${raw}")"
  if [[ -z "${stage}" ]]; then
    echo "!! Could not understand stage token: ${raw}"
    show_help
    exit 2
  fi
  run_stage "${stage}"
done

echo "============================================================"
echo "Done."
echo "Results: ${RESULTS_DIR}"
echo "Logs   : ${GEOPRIOR_LOG_PATH}"
echo "============================================================"

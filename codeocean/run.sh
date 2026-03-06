#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# GeoPrior-v3 — Code Ocean Run Target
#
# Usage:
#   ./codeocean/run.sh
#   ./codeocean/run.sh "1,2,3"
#   GEOPRIOR_STAGES="1,2,3,5" ./codeocean/run.sh
#
# Optional per-stage args via env:
#   STAGE1_ARGS="--foo bar" STAGE2_ARGS="--epochs 30" ./codeocean/run.sh
# ------------------------------------------------------------

# ------------------------------------------------------------------
# Data presence check (fail early with a helpful message)
# ------------------------------------------------------------------
EXPECTED_BIG_FN="$(
python - <<'PY'
import pathlib
p = pathlib.Path("nat.com/config.py")
cfg = {}
exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), cfg)
print(cfg.get("BIG_FN", "").strip())
PY
)"

if [[ -z "${EXPECTED_BIG_FN}" ]]; then
  echo "!! Could not resolve BIG_FN from nat.com/config.py"
  exit 2
fi

if [[ ! -f "data/${EXPECTED_BIG_FN}" ]]; then
  echo "============================================================"
  echo "Missing input CSV:"
  echo "  data/${EXPECTED_BIG_FN}"
  echo
  echo "Place the processed city dataset in ./data/ (repo root)."
  echo "Expected file naming follows nat.com/config.py:"
  echo "  BIG_FN_TEMPLATE = {city}_final_main_std.harmonized.cleaned.{variant}.csv"
  echo "  DATASET_VARIANT = with_zsurf"
  echo "============================================================"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Keep logs and outputs deterministic within the capsule.
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"

# For Code Ocean, keep logs inside the capsule by default.
export GEOPRIOR_LOG_PATH="${GEOPRIOR_LOG_PATH:-${ROOT_DIR}/outputs/logs}"
mkdir -p "${GEOPRIOR_LOG_PATH}"
mkdir -p "${ROOT_DIR}/outputs"

show_help() {
  cat <<'EOF'
GeoPrior-v3 Code Ocean runner

Runs pipeline stages in order:
  1) stage1  data prep / packaging
  2) stage2  training
  3) stage3  evaluation + figures (if available)
  4) stage4  optional diagnostics (if available)
  5) stage5  optional transfer (if available)

Examples:
  ./codeocean/run.sh
  ./codeocean/run.sh "1,2,3,5"
  GEOPRIOR_STAGES="1,2,3" ./codeocean/run.sh

Per-stage args (optional):
  STAGE1_ARGS="..." STAGE2_ARGS="..." STAGE3_ARGS="..." ./codeocean/run.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

STAGES="${1:-${GEOPRIOR_STAGES:-1,2,3}}"

run_py() {
  local file_a="$1"
  local file_b="$2"
  local args="${3:-}"

  if [[ -f "${file_a}" ]]; then
    echo "==> Running: python ${file_a} ${args}"
    # shellcheck disable=SC2086
    python "${file_a}" ${args}
    return 0
  fi

  if [[ -f "${file_b}" ]]; then
    echo "==> Running: python ${file_b} ${args}"
    # shellcheck disable=SC2086
    python "${file_b}" ${args}
    return 0
  fi

  echo "!! Skipped (not found): ${file_a} or ${file_b}"
  return 0
}

run_stage() {
  local s="$1"
  case "${s}" in
    1)
      run_py "nat.com/stage1.py" "scripts/stage1_build_dataset.py" \
        "${STAGE1_ARGS:-}"
      ;;
    2)
      run_py "nat.com/stage2.py" "scripts/stage2_train_model.py" \
        "${STAGE2_ARGS:-}"
      ;;
    3)
      run_py "nat.com/stage3.py" "scripts/stage3_eval_and_figures.py" \
        "${STAGE3_ARGS:-}"
      ;;
    4)
      run_py "nat.com/stage4.py" "scripts/stage4_diagnostics.py" \
        "${STAGE4_ARGS:-}"
      ;;
    5)
      run_py "nat.com/stage5.py" "scripts/stage5_transfer.py" \
        "${STAGE5_ARGS:-}"
      ;;
    *)
      echo "!! Unknown stage: ${s}"
      show_help
      exit 2
      ;;
  esac
}

echo "============================================================"
echo "GeoPrior-v3 | Code Ocean Run"
echo "Root: ${ROOT_DIR}"
echo "Stages: ${STAGES}"
echo "Logs: ${GEOPRIOR_LOG_PATH}"
echo "============================================================"

IFS=',' read -r -a STAGE_LIST <<< "${STAGES}"
for s in "${STAGE_LIST[@]}"; do
  s="$(echo "${s}" | tr -d '[:space:]')"
  run_stage "${s}"
done

echo "============================================================"
echo "Done."
echo "Outputs: ${ROOT_DIR}/outputs"
echo "Logs: ${GEOPRIOR_LOG_PATH}"
echo "============================================================"
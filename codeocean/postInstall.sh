#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# GeoPrior-v3 — Code Ocean postInstall
#
# Aligned with the new capsule README:
# - install GeoPrior from PyPI
# - prepare reviewer-facing root folders
# - expect root ./config.py as the source of truth
# - let GeoPrior generate/update nat.com/config.json on first run
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${ROOT_DIR}/config.py"
OUTPUTS_DIR="${ROOT_DIR}/outputs"
RESULTS_DIR="${ROOT_DIR}/results"
LOG_DIR="${OUTPUTS_DIR}/logs"
DATA_DIR="${ROOT_DIR}/data"
NATCOM_DIR="${ROOT_DIR}/nat.com"
PRIMARY_DATA_DEFAULT="${DATA_DIR}/zhongshan_full_city.csv"
SECONDARY_DATA_DEFAULT="${DATA_DIR}/nansha_full_city.csv"

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export GEOPRIOR_LOG_PATH="${GEOPRIOR_LOG_PATH:-${LOG_DIR}}"

echo "==> postInstall: GeoPrior-v3"
echo "Root: ${ROOT_DIR}"

python -V
python -m pip -V

echo "==> Upgrading pip"
python -m pip install -U pip

echo "==> Installing GeoPrior-v3 from PyPI"
if ! python -m pip install geoprior-v3 joblib pandas scikit-learn matplotlib tensorflow; then
  echo "!! PyPI installation failed."
  if [[ -f "${ROOT_DIR}/pyproject.toml" || -f "${ROOT_DIR}/setup.py" ]]; then
    echo "==> Falling back to local source installation"
    python -m pip install -e .
  else
    exit 2
  fi
fi

echo "==> Preparing capsule directories"
mkdir -p "${OUTPUTS_DIR}" "${RESULTS_DIR}" "${LOG_DIR}" "${DATA_DIR}" "${NATCOM_DIR}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "============================================================"
  echo "Missing reviewer-facing config file:"
  echo "  ${CONFIG_PATH}"
  echo
  echo "The new README expects a root-level ./config.py shipped with the capsule."
  echo "============================================================"
  exit 2
fi

echo "==> Root config detected"
echo "    ${CONFIG_PATH}"

echo "==> Quick import test"
python - <<'PY'
import geoprior
print("GeoPrior import OK")
print("GeoPrior version:", getattr(geoprior, "__version__", "unknown"))
PY

echo "==> Quick CLI test"
geoprior --help >/dev/null

echo "==> Reminder: expected reviewer datasets"
echo "    Primary  : ${GEOPRIOR_PRIMARY_DATA:-${PRIMARY_DATA_DEFAULT}}"
echo "    Transfer : ${GEOPRIOR_SECONDARY_DATA:-${SECONDARY_DATA_DEFAULT}}"

echo "==> Note"
echo "    nat.com/config.json will be created or updated automatically"
echo "    when GeoPrior runs with --config ./config.py"

echo "==> postInstall complete"
echo "Results         : ${RESULTS_DIR}"
echo "Outputs         : ${OUTPUTS_DIR}"
echo "Capsule logs    : ${GEOPRIOR_LOG_PATH}"
echo "Reviewer config : ${CONFIG_PATH}"

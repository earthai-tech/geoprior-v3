#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# GeoPrior-v3 — Code Ocean postInstall
# - installs the package
# - prepares default log/output dirs
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "==> postInstall: GeoPrior-v3"
echo "Root: ${ROOT_DIR}"

python -V
python -m pip -V

echo "==> Upgrading pip"
python -m pip install -U pip

# Install the package (editable is convenient for capsules).
echo "==> Installing GeoPrior-v3 (editable)"
python -m pip install -e .

# Default logs: user preference is ~/.geoprior/logs
DEFAULT_USER_LOG_DIR="${HOME}/.geoprior/logs"
mkdir -p "${DEFAULT_USER_LOG_DIR}"

# Capsule-contained logs (recommended for reproducible artifacts)
CAPSULE_LOG_DIR="${ROOT_DIR}/outputs/logs"
mkdir -p "${CAPSULE_LOG_DIR}"

# If user didn't set it, default to capsule-contained logs
if [[ -z "${GEOPRIOR_LOG_PATH:-}" ]]; then
  export GEOPRIOR_LOG_PATH="${CAPSULE_LOG_DIR}"
fi

mkdir -p "${ROOT_DIR}/outputs"

echo "==> Preparing data folder"
mkdir -p "${ROOT_DIR}/data"

EXPECTED_BIG_FN="$(
python - <<'PY'
import pathlib
p = pathlib.Path("nat.com/config.py")
cfg = {}
exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), cfg)
print(cfg.get("BIG_FN", "").strip())
PY
)"

echo "==> Expected input CSV (for current CITY_NAME):"
echo "    ${ROOT_DIR}/data/${EXPECTED_BIG_FN}"

echo "==> Quick import test"
python - <<'PY'
import geoprior
print("GeoPrior import OK")
print("GeoPrior version:", getattr(geoprior, "__version__", "unknown"))
PY

echo "==> postInstall complete"
echo "User logs: ${DEFAULT_USER_LOG_DIR}"
echo "Capsule logs: ${CAPSULE_LOG_DIR}"
echo "GEOPRIOR_LOG_PATH: ${GEOPRIOR_LOG_PATH}"



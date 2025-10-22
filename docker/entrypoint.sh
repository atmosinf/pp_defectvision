#!/bin/bash
set -euo pipefail

# Optional path to a dotenv file that should be sourced before booting.
if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
  echo "Loading environment from ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

PORT="${PORT:-8000}"
APP_MODULE="${APP_MODULE:-inference.api:app}"
APP_DIR="${APP_DIR:-/app/src}"

# Default checkpoint location inside the container so the API can start with baked-in models.
if [[ -z "${DEFECTVISION_MODEL_PATH:-}" ]]; then
  export DEFECTVISION_MODEL_PATH="/app/models/model.pth"
fi

if [[ -z "${DEFECTVISION_CLASS_NAMES_PATH:-}" ]]; then
  export DEFECTVISION_CLASS_NAMES_PATH="/app/models/class_names.json"
fi

if [[ ! -f "${DEFECTVISION_MODEL_PATH}" ]]; then
  echo "Warning: model checkpoint not found at ${DEFECTVISION_MODEL_PATH}" >&2
fi

exec uvicorn "${APP_MODULE}" --host "0.0.0.0" --port "${PORT}" --app-dir "${APP_DIR}"

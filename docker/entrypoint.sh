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

echo "[Entrypoint] Starting at $(date -Is)"
echo "[Entrypoint] AWS_LAMBDA_RUNTIME_API=${AWS_LAMBDA_RUNTIME_API:-unset}"
echo "[Entrypoint] DEFECTVISION_MODEL_S3_URI=${DEFECTVISION_MODEL_S3_URI:-unset}"
echo "[Entrypoint] DEFECTVISION_CLASS_NAMES_S3_URI=${DEFECTVISION_CLASS_NAMES_S3_URI:-unset}"

# Default checkpoint location inside the container so the API can start with baked-in models.
set_default_paths() {
  local baked_model="$1"
  local baked_class="$2"

  if [[ -z "${DEFECTVISION_MODEL_PATH:-}" ]]; then
    if [[ -n "${AWS_LAMBDA_RUNTIME_API:-}" && -n "${DEFECTVISION_MODEL_S3_URI:-}" ]]; then
      export DEFECTVISION_MODEL_PATH="/tmp/model.pth"
    else
      export DEFECTVISION_MODEL_PATH="$baked_model"
    fi
  fi

  if [[ -z "${DEFECTVISION_CLASS_NAMES_PATH:-}" ]]; then
    if [[ -n "${AWS_LAMBDA_RUNTIME_API:-}" && -n "${DEFECTVISION_CLASS_NAMES_S3_URI:-}" ]]; then
      export DEFECTVISION_CLASS_NAMES_PATH="/tmp/class_names.json"
    else
      export DEFECTVISION_CLASS_NAMES_PATH="$baked_class"
    fi
  fi
}

if [[ -n "${AWS_LAMBDA_RUNTIME_API:-}" ]]; then
  set_default_paths "/var/task/models/model.pth" "/var/task/models/class_names.json"
else
  set_default_paths "/app/models/model.pth" "/app/models/class_names.json"
fi

ensure_parent_dir() {
  local target="$1"
  mkdir -p "$(dirname "$target")"
}

download_s3_object() {
  local uri="$1"
  local destination="$2"
  echo "[Entrypoint] Downloading ${uri} -> ${destination}"
  S3_URI="$uri" DEST_PATH="$destination" python - <<'PY'
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

s3_uri = os.environ["S3_URI"]
dest_path = Path(os.environ["DEST_PATH"])

parsed = urlparse(s3_uri)
if parsed.scheme != "s3":
    print(f"Invalid S3 URI: {s3_uri}", file=sys.stderr)
    sys.exit(1)

bucket = parsed.netloc
key = parsed.path.lstrip("/")

dest_path.parent.mkdir(parents=True, exist_ok=True)

if dest_path.exists():
    print(f"S3 object already downloaded at {dest_path}, skipping.")
    sys.exit(0)

s3 = boto3.client("s3")
try:
    s3.download_file(bucket, key, str(dest_path))
    print(f"Downloaded {s3_uri} -> {dest_path}")
except ClientError as exc:
    print(f"Failed to download {s3_uri}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

ensure_parent_dir "${DEFECTVISION_MODEL_PATH}"
ensure_parent_dir "${DEFECTVISION_CLASS_NAMES_PATH}"

if [[ -n "${DEFECTVISION_MODEL_S3_URI:-}" ]]; then
  download_s3_object "${DEFECTVISION_MODEL_S3_URI}" "${DEFECTVISION_MODEL_PATH}"
fi

if [[ -n "${DEFECTVISION_CLASS_NAMES_S3_URI:-}" ]]; then
  download_s3_object "${DEFECTVISION_CLASS_NAMES_S3_URI}" "${DEFECTVISION_CLASS_NAMES_PATH}"
fi

echo "Model checkpoint: ${DEFECTVISION_MODEL_PATH} ($(ls -lh "${DEFECTVISION_MODEL_PATH}" 2>/dev/null || echo 'missing'))"
echo "Class names: ${DEFECTVISION_CLASS_NAMES_PATH} ($(ls -lh "${DEFECTVISION_CLASS_NAMES_PATH}" 2>/dev/null || echo 'missing'))"

if [[ ! -f "${DEFECTVISION_MODEL_PATH}" ]]; then
  echo "Error: model checkpoint not found at ${DEFECTVISION_MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${DEFECTVISION_CLASS_NAMES_PATH}" ]]; then
  echo "Error: class names file not found at ${DEFECTVISION_CLASS_NAMES_PATH}" >&2
  exit 1
fi

exec uvicorn "${APP_MODULE}" --host "0.0.0.0" --port "${PORT}" --app-dir "${APP_DIR}"

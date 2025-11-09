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
  if [[ -n "${AWS_LAMBDA_RUNTIME_API:-}" ]]; then
    export DEFECTVISION_MODEL_PATH="/tmp/model.pth"
  else
    export DEFECTVISION_MODEL_PATH="/app/models/model.pth"
  fi
fi

if [[ -z "${DEFECTVISION_CLASS_NAMES_PATH:-}" ]]; then
  if [[ -n "${AWS_LAMBDA_RUNTIME_API:-}" ]]; then
    export DEFECTVISION_CLASS_NAMES_PATH="/tmp/class_names.json"
  else
    export DEFECTVISION_CLASS_NAMES_PATH="/app/models/class_names.json"
  fi
fi

ensure_parent_dir() {
  local target="$1"
  mkdir -p "$(dirname "$target")"
}

download_s3_object() {
  local uri="$1"
  local destination="$2"
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

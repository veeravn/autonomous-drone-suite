#!/usr/bin/env bash
set -euo pipefail

# Where the repo lives on Jetson (must match install.sh REPO_DIR)
REPO_DIR="${REPO_DIR:-/opt/autonomous-drone-suite}"

cd "${REPO_DIR}"

# Activate venv
if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "[JETSON] .venv not found in ${REPO_DIR}, aborting."
  exit 1
fi

# Default runtime flags for Jetson hardware mode
# Adjust camera index, takeoff alt, etc. as needed.
python -m src.main \
  --use-sitl 0 \
  --hardware 1 \
  --camera 0 \
  --use-gestures 1 \
  --semantic-nbv 1 \
  --safety 1 \
  --takeoff 3.0

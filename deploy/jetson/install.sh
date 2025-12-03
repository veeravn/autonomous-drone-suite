#!/usr/bin/env bash
set -euo pipefail

###############################################
# Dual-Mode Jetson Installer for Autonomous Drone Suite
#
# Usage:
#   deploy/jetson/install.sh              # default: dev mode
#   deploy/jetson/install.sh dev          # explicit dev mode
#   sudo deploy/jetson/install.sh prod    # production mode
#
# Modes:
#   dev  - operate in-place in current repo (no rsync, no sudo)
#   prod - rsync code into REPO_DIR (default: /opt/autonomous-drone-suite)
#
# Env vars you can tweak:
#   REPO_DIR           - target repo dir
#                        dev: default = current working directory
#                        prod: default = /opt/autonomous-drone-suite
#   TORCH_WHL          - full URL/path to NVIDIA PyTorch wheel (for Jetson only)
#   TORCHVISION_WHL    - full URL/path to NVIDIA TorchVision wheel (for Jetson only)
###############################################

MODE="${1:-dev}"   # dev | prod

echo "[INSTALL] Mode: ${MODE}"

if [[ "${MODE}" != "dev" && "${MODE}" != "prod" ]]; then
  echo "ERROR: MODE must be 'dev' or 'prod'"
  echo "Usage: $0 [dev|prod]"
  exit 1
fi

###############################################
# Detect Jetson
###############################################
ARCH="$(uname -m)"
JETSON=0
if [[ "${ARCH}" == "aarch64" && -f /etc/nv_tegra_release ]]; then
  JETSON=1
  echo "[INSTALL] Detected Jetson (aarch64 + /etc/nv_tegra_release)"
else
  echo "[INSTALL] Not a Jetson (arch=${ARCH})"
fi

###############################################
# Determine REPO_DIR
###############################################
if [[ "${MODE}" == "dev" ]]; then
  # In dev mode we assume you're running inside the repo you cloned
  REPO_DIR="${REPO_DIR:-$(pwd)}"
  echo "[INSTALL] Dev mode: using in-place repo at ${REPO_DIR}"
else
  # Production install under /opt (override with REPO_DIR if you want)
  REPO_DIR="${REPO_DIR:-/opt/autonomous-drone-suite}"
  echo "[INSTALL] Prod mode: target repo dir ${REPO_DIR}"

  # Ensure target dir exists and is owned by current user
  if [[ ! -d "${REPO_DIR}" ]]; then
    echo "[INSTALL] Creating ${REPO_DIR} (requires sudo)"
    sudo mkdir -p "${REPO_DIR}"
    sudo chown "$(id -u):$(id -g)" "${REPO_DIR}"
  fi

  # Rsync current repo into REPO_DIR (excluding venv & git metadata)
  echo "[INSTALL] Syncing code to ${REPO_DIR}"
  rsync -a --delete \
    --exclude ".venv" \
    --exclude ".git" \
    ./ "${REPO_DIR}/"
fi

cd "${REPO_DIR}"

###############################################
# Python virtualenv
###############################################
if [[ ! -d ".venv" ]]; then
  echo "[INSTALL] Creating virtualenv at ${REPO_DIR}/.venv"
  python3 -m venv .venv
else
  echo "[INSTALL] Reusing existing virtualenv at ${REPO_DIR}/.venv"
fi

# shellcheck source=/dev/null
source .venv/bin/activate

echo "[INSTALL] Upgrading pip/setuptools/wheel"
pip install --upgrade pip setuptools wheel

###############################################
# Requirements selection
###############################################
REQ_FILE=""
if [[ -f "requirements_jetson.txt" && "${JETSON}" -eq 1 ]]; then
  REQ_FILE="requirements_jetson.txt"
else
  if [[ -f "requirements.txt" ]]; then
    REQ_FILE="requirements.txt"
  fi
fi

if [[ -n "${REQ_FILE}" ]]; then
  echo "[INSTALL] Installing dependencies from ${REQ_FILE}"
  pip install -r "${REQ_FILE}"
else
  echo "[INSTALL] WARNING: No requirements.txt or requirements_jetson.txt found."
fi

###############################################
# Optional: install NVIDIA PyTorch / TorchVision wheels
###############################################
if [[ "${JETSON}" -eq 1 ]]; then
  if [[ -n "${TORCH_WHL:-}" ]]; then
    echo "[INSTALL] Installing PyTorch from ${TORCH_WHL}"
    pip install "${TORCH_WHL}"
  else
    echo "[INSTALL] Skipping PyTorch wheel (set TORCH_WHL=... to install)"
  fi

  if [[ -n "${TORCHVISION_WHL:-}" ]]; then
    echo "[INSTALL] Installing TorchVision from ${TORCHVISION_WHL}"
    pip install "${TORCHVISION_WHL}"
  else
    echo "[INSTALL] Skipping TorchVision wheel (set TORCHVISION_WHL=... to install)"
  fi
else
  echo "[INSTALL] Non-Jetson: not installing Jetson-specific torch wheels."
fi

###############################################
# Make start script executable
###############################################
if [[ -f "deploy/jetson/start_agent.sh" ]]; then
  chmod +x deploy/jetson/start_agent.sh || true
fi

echo
echo "[INSTALL] Done."
echo

if [[ "${MODE}" == "dev" ]]; then
  cat <<EOF
[INSTALL] Dev mode instructions:

  cd ${REPO_DIR}
  source .venv/bin/activate
  # Example: local loop
  python -m src.main --use-sitl 0 --camera 0

  # Example: mock-drone SITL-style loop
  python -m src.main --use-sitl 1 --mock-drone 1 --camera -1

EOF
else
  cat <<EOF
[INSTALL] Prod mode instructions:

  cd ${REPO_DIR}
  source .venv/bin/activate
  deploy/jetson/start_agent.sh

You can also hook this into systemd, e.g.:

  sudo cp deploy/jetson/drone_agent.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now drone_agent

EOF
fi

if [[ "${JETSON}" -eq 1 ]]; then
  cat <<EOF
[INSTALL] NOTE: On Jetson, for full NVIDIA torch support, rerun with:

  TORCH_WHL=<pytorch_wheel_url> TORCHVISION_WHL=<torchvision_wheel_url> \\
    deploy/jetson/install.sh ${MODE}

EOF
fi

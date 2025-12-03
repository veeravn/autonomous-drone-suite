#!/usr/bin/env bash
set -euo pipefail

###############################################
# Smart Jetson Installer for Autonomous Drone Suite
#
# - Detects Jetson (aarch64 + /etc/nv_tegra_release)
# - Creates venv
# - Installs requirements_jetson.txt if present
# - Optionally installs NVIDIA PyTorch / TorchVision wheels
#
# Env vars you can tweak:
#   REPO_DIR           - target repo dir (default: /opt/autonomous-drone-suite)
#   TORCH_WHL          - full URL/path to NVIDIA PyTorch wheel (for Jetson only)
#   TORCHVISION_WHL    - full URL/path to NVIDIA TorchVision wheel (for Jetson only)
###############################################

# REPO_DIR="${REPO_DIR:-/opt/autonomous-drone-suite}"
REPO_DIR="${REPO_DIR:-$HOME/autonomous-drone-suite}"

echo "[INSTALL] Target repo directory: ${REPO_DIR}"

# --- Detect Jetson / architecture ---
ARCH="$(uname -m || echo unknown)"
JETSON=0
if [[ "${ARCH}" == "aarch64" ]] && [[ -f /etc/nv_tegra_release ]]; then
  JETSON=1
  echo "[INSTALL] Detected Jetson platform (aarch64 + nv_tegra)."
else
  echo "[INSTALL] Non-Jetson platform detected (arch=${ARCH})."
fi

# --- Prepare repo directory ---
sudo mkdir -p "${REPO_DIR}"
sudo chown -R "$USER":"$USER" "${REPO_DIR}"

# If we're running this from within a clone, copy the contents over
if [ -d .git ] || [ -f "pyproject.toml" ] || [ -f "setup.cfg" ]; then
  echo "[INSTALL] Copying current repo contents to ${REPO_DIR}"
  rsync -a --exclude=".venv" --exclude=".git" ./ "${REPO_DIR}/"
fi

cd "${REPO_DIR}"

# --- Create virtualenv ---
if [ ! -d ".venv" ]; then
  echo "[INSTALL] Creating virtualenv at ${REPO_DIR}/.venv"
  python3 -m venv .venv
else
  echo "[INSTALL] Reusing existing virtualenv at ${REPO_DIR}/.venv"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[INSTALL] Python: $(python --version)"
echo "[INSTALL] Pip:    $(pip --version)"

echo "[INSTALL] Upgrading pip / wheel / setuptools"
pip install --upgrade pip wheel setuptools

# --- Choose requirements file ---
REQ_FILE="requirements_jetson.txt"
if [ ! -f "${REQ_FILE}" ]; then
  echo "[INSTALL] ${REQ_FILE} not found, falling back to requirements.txt"
  REQ_FILE="requirements.txt"
fi

echo "[INSTALL] Using requirements file: ${REQ_FILE}"

# --- Install base dependencies ---
echo "[INSTALL] Installing base dependencies from ${REQ_FILE}"
pip install -r "${REQ_FILE}"

# --- Optional: Jetson-specific PyTorch / TorchVision ---
if [[ "${JETSON}" -eq 1 ]]; then
  echo "[INSTALL] Jetson mode: You can optionally install NVIDIA PyTorch / TorchVision wheels."

  if [[ -n "${TORCH_WHL:-}" ]]; then
    echo "[INSTALL] Installing PyTorch from: ${TORCH_WHL}"
    pip install "${TORCH_WHL}"
  else
    echo "[INSTALL] Skipping PyTorch (TORCH_WHL not set)."
    echo "          Set TORCH_WHL to NVIDIA's wheel URL if you need torch on Jetson."
  fi

  if [[ -n "${TORCHVISION_WHL:-}" ]]; then
    echo "[INSTALL] Installing TorchVision from: ${TORCHVISION_WHL}"
    pip install "${TORCHVISION_WHL}"
  else
    echo "[INSTALL] Skipping TorchVision (TORCHVISION_WHL not set)."
    echo "          Set TORCHVISION_WHL to NVIDIA's wheel URL if you need torchvision on Jetson."
  fi
else
  echo "[INSTALL] Non-Jetson platform: PyTorch / TorchVision, if needed, will be managed by ${REQ_FILE}."
fi

echo "[INSTALL] Making start script executable"
chmod +x deploy/jetson/start_agent.sh || true

echo "[INSTALL] Done."
echo
echo "To run the agent manually:"
echo "  cd ${REPO_DIR}"
echo "  source .venv/bin/activate"
echo "  deploy/jetson/start_agent.sh"
echo
if [[ "${JETSON}" -eq 1 ]]; then
  echo "[INSTALL] NOTE: On Jetson, for full torch support, rerun with:"
  echo "  TORCH_WHL=<pytorch_wheel_url> TORCHVISION_WHL=<torchvision_wheel_url> \\"
  echo "    deploy/jetson/install.sh"
fi

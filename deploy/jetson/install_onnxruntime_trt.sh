#!/usr/bin/env bash
set -euo pipefail

###########################################################
# Jetson ONNX Runtime + TensorRT EP Installer
#
# This script:
#   - Detects Jetson (aarch64 + /etc/nv_tegra_release)
#   - Optionally installs a TensorRT-enabled ONNXRuntime wheel
#     if ORT_TRT_WHL is set.
#
# Usage on Jetson:
#   export ORT_TRT_WHL="https://<your_onnxruntime_tensorrt_wheel>.whl"
#   bash deploy/jetson/install_onnxruntime_trt.sh
###########################################################

ARCH="$(uname -m || echo unknown)"
JETSON=0
if [[ "${ARCH}" == "aarch64" ]] && [[ -f /etc/nv_tegra_release ]]; then
  JETSON=1
  echo "[ORT-TRT] Detected Jetson platform (aarch64 + nv_tegra)."
else
  echo "[ORT-TRT] Non-Jetson platform detected (arch=${ARCH})."
fi

if [[ "${JETSON}" -ne 1 ]]; then
  echo "[ORT-TRT] This script is intended for Jetson only. Exiting."
  exit 0
fi

if [[ -z "${ORT_TRT_WHL:-}" ]]; then
  echo "[ORT-TRT] ORT_TRT_WHL is not set. Skipping TensorRT ORT install."
  echo "          Set ORT_TRT_WHL to a TensorRT-enabled onnxruntime wheel URL to use this."
  exit 0
fi

if [ ! -d ".venv" ]; then
  echo "[ORT-TRT] .venv not found in current directory. Run from repo root after install.sh"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[ORT-TRT] Installing TensorRT-enabled ONNXRuntime from:"
echo "          ${ORT_TRT_WHL}"

pip install --upgrade "${ORT_TRT_WHL}"

echo "[ORT-TRT] Done. Available ORT providers now:"
python - << 'EOF'
import onnxruntime as ort
print(ort.get_available_providers())
EOF
#!/usr/bin/env bash
set -euo pipefail


# Create venv (prefer Python 3.10 for MediaPipe). If pyenv exists, try it.
if command -v python3.10 >/dev/null 2>&1; then
PY=python3.10
else
PY=python3
fi


$PY -m venv .venv
source .venv/bin/activate


pip install --upgrade pip wheel setuptools


# Some Jetson/arm64 environments may require specific torch builds; this is a generic CPU/GPU install.
pip install -r requirements.txt || {
echo "\nIf MediaPipe fails on Python > 3.10, it's optional. Continue without gestures:";
sed -i.bak '/mediapipe/d' requirements.txt || true
pip install -r requirements.txt
}


mkdir -p data/images data/logs


echo "\n✅ Setup complete. Activate with: source .venv/bin/activate"
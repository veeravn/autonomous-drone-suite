#!/usr/bin/env bash
set -euo pipefail


# Example PX4 SITL launcher; adjust to your PX4 path
PX4_DIR=${PX4_DIR:-$HOME/PX4-Autopilot}


if [ ! -d "$PX4_DIR" ]; then
echo "PX4 not found at $PX4_DIR"; exit 1
fi


cd "$PX4_DIR"
make px4_sitl gz_x500
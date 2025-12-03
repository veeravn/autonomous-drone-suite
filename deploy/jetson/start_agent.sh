#!/usr/bin/env bash
set -euo pipefail

###############################################
# Start script for Autonomous Drone Suite
#
# Profiles:
#   dev  - Jetson-only / laptop dev profile:
#          - MockDrone backend (no PX4 required)
#          - SITL-style loop (run_sitl) but with --mock-drone
#          - Headless camera by default (can override)
#
#   prod - Flight profile:
#          - Real Pixhawk over MAVSDK
#          - Hardware mode enabled
#          - Camera enabled
#
# Usage:
#   deploy/jetson/start_agent.sh           # defaults to dev
#   deploy/jetson/start_agent.sh dev
#   deploy/jetson/start_agent.sh prod
#
# Env overrides:
#   PROFILE           - dev | prod
#   CAMERA_INDEX      - camera index (default dev:-1, prod:0)
#   TAKEOFF_ALT_M     - default takeoff altitude (m)
#   LOOP_HZ           - main loop frequency
###############################################

PROFILE="${1:-${PROFILE:-dev}}"   # dev | prod

case "${PROFILE}" in
  dev|DEV)
    PROFILE="dev"
    ;;
  prod|PROD|production|flight)
    PROFILE="prod"
    ;;
  *)
    echo "[START] Unknown profile: ${PROFILE}"
    echo "  Use: dev | prod"
    exit 1
    ;;
esac

echo "[START] Profile: ${PROFILE}"

# Ensure we're in repo root (this script lives in deploy/jetson/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

# Activate virtualenv
if [[ -d ".venv" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
else
  echo "[START] ERROR: .venv not found in ${REPO_DIR}"
  echo "        Did you run deploy/jetson/install.sh ?"
  exit 1
fi

# Common tunables with sensible defaults
TAKEOFF_ALT_M="${TAKEOFF_ALT_M:-3.0}"
LOOP_HZ="${LOOP_HZ:-15}"

if [[ "${PROFILE}" == "dev" ]]; then
  ###############################################
  # DEV PROFILE
  #
  # - MockDrone backend
  # - No hardware or PX4 required
  # - Headless camera by default, override with CAMERA_INDEX
  ###############################################
  CAMERA_INDEX="${CAMERA_INDEX:--1}"

  echo "[START][DEV] Running with MockDrone, no hardware."
  echo "[START][DEV] Camera index: ${CAMERA_INDEX}"
  echo "[START][DEV] Takeoff alt (mock): ${TAKEOFF_ALT_M} m, loop: ${LOOP_HZ} Hz"

  exec python -m src.main \
    --use-sitl 1 \
    --mock-drone 1 \
    --hardware 0 \
    --camera "${CAMERA_INDEX}" \
    --takeoff "${TAKEOFF_ALT_M}" \
    --semantic-nbv 1 \
    --safety 1 \
    --min-rel-alt 1.0 \
    --max-rel-alt 15.0 \
    --rtl-battery 20.0 \
    --loop-hz "${LOOP_HZ}"

else
  ###############################################
  # PROD PROFILE
  #
  # - Real Pixhawk over MAVSDK
  # - Hardware mode = 1
  # - Camera enabled by default (0)
  # - Keep safety layer *always* on
  ###############################################
  CAMERA_INDEX="${CAMERA_INDEX:-0}"

  echo "[START][PROD] Running with real Pixhawk (MAVSDK)."
  echo "[START][PROD] Camera index: ${CAMERA_INDEX}"
  echo "[START][PROD] Takeoff alt: ${TAKEOFF_ALT_M} m, loop: ${LOOP_HZ} Hz"

  exec python -m src.main \
    --use-sitl 1 \
    --mock-drone 0 \
    --hardware 1 \
    --camera "${CAMERA_INDEX}" \
    --takeoff "${TAKEOFF_ALT_M}" \
    --semantic-nbv 1 \
    --safety 1 \
    --min-rel-alt 1.0 \
    --max-rel-alt 15.0 \
    --rtl-battery 20.0 \
    --loop-hz "${LOOP_HZ}"
fi

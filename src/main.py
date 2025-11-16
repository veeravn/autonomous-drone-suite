# src/main.py
from __future__ import annotations

import argparse
import asyncio
import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv

from .config import Config
from .planner_agent import PlannerAgent
from .gesture_control import GestureController
from .shot_dedupe import ShotDeduper
from .telemetry_stub import Telemetry as StubTelemetry
from .utils.video import open_camera

from .gestures.mapper import GestureMapper, GestureActionType, FlightState
from .flight.mavsdk_client import MavsdkClient

load_dotenv()

# Default connection URL; you can override with MAVSDK_URL env var if needed
DEF_MAVSDK = "udp://:14540"

# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-gestures", type=int, default=1,
                    help="Enable gesture control (1) or disable (0)")
    ap.add_argument("--use-sitl", type=int, default=0,
                    help="Use PX4 SITL + MAVSDK (1) or local-only mode (0)")
    ap.add_argument("--camera", type=int, default=0,
                    help="Webcam index; -1 = headless dummy frame")
    ap.add_argument("--takeoff", type=float, default=3.0,
                    help="Takeoff altitude (m) in SITL")
    return ap.parse_args()


# ---------- Helpers for gesture actions ----------

async def apply_gesture_action(
    mav: MavsdkClient,
    state: FlightState,
    action,
    tel,
    takeoff_alt: float,
) -> FlightState:
    """
    Map a high-level gesture action into MAVSDK commands + state transitions.
    Altitude changes use discrete climbs only; main loop does yaw-only setpoints.
    """

    kind = action.kind

    # TAKEOFF (rarely used if we auto-takeoff at start)
    if kind == GestureActionType.TAKEOFF:
        if state == FlightState.IDLE:
            await mav.arm_and_takeoff(rel_alt_m=takeoff_alt)
            await mav.start_offboard()
            run_sitl.alt_target = float(takeoff_alt)
            await mav.climb_to_alt(run_sitl.alt_target, vz_m_s=-0.8, max_seconds=12.0)
            return FlightState.CAPTURING
        return state

    # ALTITUDE OFFSET
    if kind == GestureActionType.ALT_OFFSET:
        # Discrete altitude change: update target, then single climb
        new_target = max(0.5, float(tel.alt) + float(action.dz))
        if abs(new_target - float(tel.alt)) < 0.10:  # ignore tiny changes
            return state
        run_sitl.alt_target = new_target
        print(f"[ALT CMD] → {run_sitl.alt_target:.2f} m")
        await mav.climb_to_alt(
            run_sitl.alt_target,
            vz_m_s=(-0.7 if action.dz > 0 else +0.5),
            max_seconds=8.0,
        )
        return state

    # HOLD (pause offboard)
    if kind == GestureActionType.HOLD:
        await mav.stop_offboard()
        return FlightState.PAUSE

    # RESUME (restart offboard)
    if kind == GestureActionType.RESUME:
        await mav.start_offboard()
        return FlightState.CAPTURING

    # LAND
    if kind == GestureActionType.LAND:
        await mav.stop_offboard()
        await mav.sys.action.land()
        return FlightState.IDLE

    # RTL
    if kind == GestureActionType.RTL:
        await mav.stop_offboard()
        await mav.sys.action.return_to_launch()
        return FlightState.RTL

    return state


# ---------- Local (no SITL) loop ----------

def run_local(cfg: Config, args):
    """
    Local loop without MAVSDK: simulates telemetry and runs the planner/gestures.
    Good for quickly testing ONNX gestures, NBV, and dedupe without PX4.
    """
    cap = None
    try:
        # Camera
        if int(args.camera) == -1:
            cap = None  # headless
        else:
            try:
                cap = open_camera(args.camera)
            except Exception:
                # macOS AVFoundation fallback
                cap = cv2.VideoCapture(int(args.camera), cv2.CAP_AVFOUNDATION)
                if not cap.isOpened():
                    raise RuntimeError(
                        f"Cannot open camera {args.camera}. "
                        "Try --camera 1 (or -1 for headless)."
                    )

        planner = PlannerAgent()
        gestures = GestureController(enable=bool(args.use_gestures))  # ONNX-based inside
        dedupe = ShotDeduper()
        mapper = GestureMapper()
        tel_stream = StubTelemetry.fake_stream()

        t_last_capture = 0.0
        NOVELTY_CAPTURE_MIN = 0.14

        while True:
            # Frame
            if cap is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                ok = True
            else:
                ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                if cap is None:
                    ok = True
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    ok, frame = cap.read()
                if not ok:
                    print("[WARN] Camera read failed; exiting.")
                    break

            # Telemetry (stubbed)
            tel = next(tel_stream)

            # Gestures (ONNX)
            raw_gesture = gestures.detect(frame)
            if hasattr(gestures, "overlay_status"):
                gestures.overlay_status(frame)
            action = mapper.map(raw_gesture)

            # NBV decision
            emb = planner.embed(frame)
            decision = planner.decide(emb, tel.heading_deg, tel.battery)
            planner.update_history(emb)

            # OSD
            cv2.putText(
                frame,
                f"state:LOCAL nov:{decision.novelty_score:.2f}",
                (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Novelty-gated capture + dedupe
            now = time.time()
            if now - t_last_capture > cfg.capture_period_s and decision.novelty_score >= NOVELTY_CAPTURE_MIN:
                pos = (tel.lat, tel.lon, tel.alt)
                if not dedupe.is_duplicate(frame, pos):
                    dedupe.add(frame, pos)
                    cv2.imwrite(f"data/images/cap_{int(now)}.jpg", frame)
                    print("Saved unique shot")
                else:
                    print("Duplicate shot skipped")
                t_last_capture = now

            # Show
            cv2.imshow("MVP Feed (Local)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

    except KeyboardInterrupt:
        print("[INTERRUPT] Ctrl+C (local).")
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ---------- SITL / MAVSDK loop ----------

async def run_sitl(cfg: Config, args):
    """
    Full SITL loop:
      - Connects to PX4 via MAVSDK
      - Auto takeoff to args.takeoff altitude
      - Runs ONNX-based gestures via GestureController
      - Uses NBV for yaw-only offboard control
      - Captures unique shots via dedupe
    """
    cap = None
    mav: MavsdkClient | None = None

    try:
        # Camera
        if int(args.camera) == -1:
            cap = None
        else:
            try:
                cap = open_camera(args.camera)
            except Exception:
                cap = cv2.VideoCapture(int(args.camera), cv2.CAP_AVFOUNDATION)
                if not cap.isOpened():
                    raise RuntimeError(
                        f"Cannot open camera {args.camera}. "
                        "Try --camera 1 (or -1 for headless)."
                    )

        planner = PlannerAgent()
        gestures = GestureController(enable=bool(args.use_gestures))  # ONNX-based inside
        dedupe = ShotDeduper()
        mapper = GestureMapper()

        mav = MavsdkClient(DEF_MAVSDK)
        print(f"[MAVSDK] Connecting to {DEF_MAVSDK}")
        await mav.connect()

        # Auto takeoff → start offboard → discrete climb to altitude
        await mav.arm_and_takeoff(rel_alt_m=float(args.takeoff))
        await mav.start_offboard()
        print("[OFFBOARD] started")

        # This is the altitude target used by discrete climbs
        run_sitl.alt_target = float(args.takeoff)
        await mav.climb_to_alt(run_sitl.alt_target, vz_m_s=-0.8, max_seconds=12.0)
        print(f"[OFFBOARD CLIMB] commanded to ~{run_sitl.alt_target:.1f} m")

        mapper.set_state(FlightState.CAPTURING)

        t_last_capture = 0.0
        NOVELTY_CAPTURE_MIN = 0.14

        from mavsdk.offboard import VelocityNedYaw

        while True:
            # Frame
            if cap is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                ok = True
            else:
                ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.05)
                if cap is None:
                    ok = True
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    ok, frame = cap.read()
                if not ok:
                    print("[WARN] Camera read failed; exiting loop.")
                    break

            # Telemetry
            tel = await mav.telemetry_latest()
            if tel is None:
                await asyncio.sleep(0.05)
                continue

            # Gestures (ONNX)
            raw_gesture = gestures.detect(frame)
            if hasattr(gestures, "overlay_status"):
                gestures.overlay_status(frame)
            action = mapper.map(raw_gesture)

            if action.kind != GestureActionType.NONE:
                before = mapper.state
                mapper.set_state(
                    await apply_gesture_action(
                        mav, mapper.state, action, tel, takeoff_alt=float(args.takeoff)
                    )
                )
                print(f"[GESTURE] {raw_gesture} → {action.kind.name} | {before.name} → {mapper.state.name}")

            # NBV decision
            emb = planner.embed(frame)
            decision = planner.decide(emb, tel.heading_deg, tel.battery)
            planner.update_history(emb)

            # OSD
            cv2.putText(
                frame,
                f"state:{mapper.state.name} nov:{decision.novelty_score:.2f}",
                (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # --- Yaw-only offboard control; let PX4 hold altitude ---
            if mapper.state == FlightState.CAPTURING:
                delta = decision.heading_delta_deg
                # enforce minimum step so we don't get stuck on 0
                if abs(delta) < 3.0:
                    delta = 3.0 if delta >= 0 else -3.0

                target_yaw = tel.heading_deg + max(
                    -cfg.max_yaw_delta_deg, min(cfg.max_yaw_delta_deg, delta)
                )
                # wrap to [-180, 180]
                target_yaw = (target_yaw + 180.0) % 360.0 - 180.0

                if not hasattr(run_sitl, "_yaw_lp"):
                    run_sitl._yaw_lp = target_yaw
                alpha = 0.2
                run_sitl._yaw_lp = (1 - alpha) * run_sitl._yaw_lp + alpha * target_yaw

                cv2.putText(
                    frame,
                    f"hdg:{tel.heading_deg:+.1f} tgt:{run_sitl._yaw_lp:+.1f} alt:{tel.alt:.2f}",
                    (12, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                try:
                    await mav.sys.offboard.set_velocity_ned(
                        VelocityNedYaw(0.0, 0.0, 0.0, run_sitl._yaw_lp)
                    )
                except Exception as e:
                    print(f"[SEND] set_velocity failed: {e}. Exiting loop.")
                    break

            # Novelty-gated capture + dedupe
            now = time.time()
            if (
                mapper.state == FlightState.CAPTURING
                and now - t_last_capture > cfg.capture_period_s
                and decision.novelty_score >= NOVELTY_CAPTURE_MIN
            ):
                pos = (tel.lat, tel.lon, tel.alt)
                if not dedupe.is_duplicate(frame, pos):
                    dedupe.add(frame, pos)
                    cv2.imwrite(f"data/images/cap_{int(now)}.jpg", frame)
                    print("Saved unique shot")
                else:
                    print("Duplicate shot skipped")
                t_last_capture = now

            # Window + keyboard fallbacks
            cv2.imshow("MVP Feed (SITL)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("p"):  # pause/hold
                mapper.set_state(
                    await apply_gesture_action(
                        mav,
                        mapper.state,
                        mapper.type_to_action(GestureActionType.HOLD),
                        tel,
                        float(args.takeoff),
                    )
                )
            elif k == ord("r"):  # resume
                mapper.set_state(
                    await apply_gesture_action(
                        mav,
                        mapper.state,
                        mapper.type_to_action(GestureActionType.RESUME),
                        tel,
                        float(args.takeoff),
                    )
                )
            elif k == ord("u"):  # ascend
                mapper.set_state(
                    await apply_gesture_action(
                        mav,
                        mapper.state,
                        mapper.type_to_action(GestureActionType.ALT_OFFSET, dz=+0.8),
                        tel,
                        float(args.takeoff),
                    )
                )
            elif k == ord("d"):  # descend
                mapper.set_state(
                    await apply_gesture_action(
                        mav,
                        mapper.state,
                        mapper.type_to_action(GestureActionType.ALT_OFFSET, dz=-0.6),
                        tel,
                        float(args.takeoff),
                    )
                )
            elif k == ord("l"):  # land
                mapper.set_state(
                    await apply_gesture_action(
                        mav,
                        mapper.state,
                        mapper.type_to_action(GestureActionType.LAND),
                        tel,
                        float(args.takeoff),
                    )
                )

            await asyncio.sleep(0.05)

    except KeyboardInterrupt:
        print("[INTERRUPT] Ctrl+C (SITL).")
    finally:
        try:
            if mav is not None:
                await mav.stop_offboard()
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ---------- Entrypoint ----------

def main():
    args = parse_args()
    cfg = Config()
    if int(args.use_sitl) == 1:
        asyncio.run(run_sitl(cfg, args))
    else:
        run_local(cfg, args)


if __name__ == "__main__":
    main()

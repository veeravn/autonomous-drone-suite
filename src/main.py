# src/main.py
from __future__ import annotations

import argparse
import asyncio
import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv
from safety import SafetyConfig, SafetyManager

from .config import Config
from .hw_logging import HardwareLogger, FrameLog
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
    ap.add_argument("--semantic-nbv", type=int, default=1, 
                    help="Enable semantic NBV (CLIP+YOLO). 1=on, 0=off"),
    ap.add_argument( "--safety", type=int, default=1,
                    help="Enable safety constraints (1=on, 0=off).",
    )
    ap.add_argument( "--min-rel-alt", type=float, default=1.0,
                    help="Minimum relative altitude (m) in offboard.",
    )
    ap.add_argument( "--max-rel-alt", type=float, default=15.0,
                    help="Maximum relative altitude (m) in offboard.",
    )
    ap.add_argument("--rtl-battery", type=float, default=20.0,
                    help="Battery percentage threshold for RTL.",
    )
    ap.add_argument("--hardware", type=int, default=0,
                    help="Use hardware (Pixhawk over serial) instead of SITL. 0=SITL, 1=hardware.",
    )
    return ap.parse_args()


# ---------- Helpers for gesture actions ----------
async def apply_gesture_action(
    mav: MavsdkClient,
    state: FlightState,
    action,
    tel,
    takeoff_alt: float,
    safety=None
) -> FlightState:
    """
    Safety-aware gesture→flight action mapper.
    Handles:
      - Altitude clamps
      - Battery-based RTL (optional)
      - Gesture-based flight state transitions
    """

    kind = action.kind

    # ------------------------------
    # SAFETY: Battery-based RTL
    # ------------------------------
    if safety is not None:
        if safety.should_rtl(float(tel.battery), state):
            print(f"[SAFETY] Battery low ({tel.battery:.1f}%), initiating RTL.")
            try:
                await mav.rtl()
            except Exception:
                await mav.sys.action.return_to_launch()
            return FlightState.RTL

    # ------------------------------
    # TAKEOFF (THUMB_UP from IDLE)
    # ------------------------------
    if kind == "TAKEOFF":
        if state == FlightState.IDLE:
            print("[GESTURE] TAKEOFF triggered")

            target = takeoff_alt
            if safety is not None:
                # convert to relative, clamp, convert back
                rel = target - takeoff_alt
                rel = safety.clamp_altitude(rel)
                target = takeoff_alt + rel

            await mav.takeoff(target)
            return FlightState.CAPTURING
        return state

    # ------------------------------
    # LAND (FIST)
    # ------------------------------
    if kind == "LAND":
        print("[GESTURE] LAND requested")
        try:
            await mav.land()
        except AttributeError:
            await mav.sys.action.land()
        return FlightState.IDLE

    # ------------------------------
    # ALTITUDE OFFSET (+ / -)
    # ------------------------------
    if kind == "ALT_OFFSET":
        if state != FlightState.CAPTURING:
            return state
        # request new altitude (absolute)
        requested = float(tel.alt) + float(action.dz)
        rel = requested - takeoff_alt

        # convert to RELATIVE (for safety)
        rel = requested - takeoff_alt

        if safety is not None:
            rel = safety.clamp_altitude(rel)

        # convert back to absolute altitude
        final_alt = takeoff_alt + rel

        if abs(final_alt - float(tel.alt)) < 0.10:
            return state

        print(f"[ALT CMD] → {final_alt:.2f} m")

        try:
            await mav.climb_to_alt(
                final_alt,
                vz_m_s=(-0.7 if action.dz > 0 else 0.5),
                max_seconds=8.0
            )
        except RuntimeError as e:
            print(f"[ALT] climb_to_alt error: {e}")

        return state

    # ------------------------------
    # YAW OFFSET (POINT_LEFT/RIGHT)
    # ------------------------------
    if kind == "YAW_OFFSET":
        if state != FlightState.CAPTURING:
            return state

        dyaw = action.dyaw_deg
        # You already have yaw_rate in your mavsdk client
        await mav.yaw_offset(dyaw)
        return state

    # ------------------------------
    # HOLD / PAUSE (OPEN PALM)
    # ------------------------------
    if kind == "HOLD":
        print("[GESTURE] HOLD")
        return FlightState.PAUSE

    # RESUME (maybe PEACE gesture mapped earlier)
    if kind == "RESUME":
        print("[GESTURE] RESUME")
        return FlightState.CAPTURING

    return state


# ---------- Local (no SITL) loop ----------

def run_local(cfg: Config, args):
    """
    Local loop without MAVSDK: simulates telemetry and runs the planner/gestures.
    Good for quickly testing ONNX gestures, NBV, and dedupe without PX4.
    """
    cap = None
    try:
        print("[LOCAL] Starting local NBV + gesture loop")

        # Camera
        if int(args.camera) == -1:
            cap = None  # headless
            print("[LOCAL] Running headless (no camera)")
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

        planner = PlannerAgent(semantic_enabled=cfg.semantic_nbv)
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

            # Optional debug print
            if raw_gesture is not None:
                print(f"[LOCAL][GESTURE] raw={raw_gesture}")

            action = mapper.map(raw_gesture)

            # NBV decision
            sf = planner.analyze_frame(frame)
            decision = planner.decide(sf, tel.heading_deg, tel.battery)
            planner.update_history(sf, tel.heading_deg)

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
            # --- YOLO subject bounding box visualization ---
            subj = getattr(decision, "subject", None)
            if subj is not None and getattr(subj, "bbox_xyxy", None) is not None:
                try:
                    x1, y1, x2, y2 = subj.bbox_xyxy  # normalized [0,1]
                    h, w = frame.shape[:2]
                    x1_i = int(max(0, min(w - 1, x1 * w)))
                    x2_i = int(max(0, min(w - 1, x2 * w)))
                    y1_i = int(max(0, min(h - 1, y1 * h)))
                    y2_i = int(max(0, min(h - 1, y2 * h)))

                    # Draw rectangle around subject
                    cv2.rectangle(
                        frame,
                        (x1_i, y1_i),
                        (x2_i, y2_i),
                        (0, 255, 0),  # green box
                        2,
                    )

                    # Label text: class name + (optional) area %
                    label = getattr(subj, "label", "obj")
                    area = getattr(subj, "area", None)
                    if area is not None:
                        text = f"{label} {area*100:.1f}%"
                    else:
                        text = str(label)

                    cv2.putText(
                        frame,
                        text,
                        (x1_i, max(0, y1_i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                except Exception as e:
                    # Don't let drawing bugs kill the loop
                    print(f"[YOLO-OSD] Failed to draw subject bbox: {e}")

            # Novelty-gated capture + dedupe
            now = time.time()
            if (
                now - t_last_capture > cfg.capture_period_s
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
        print("[LOCAL] Cleaned up camera + windows")


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

        # ---- Connection URL selection (SITL vs hardware) ----
        if int(getattr(args, "hardware", 0)) == 1:
            # Hardware mode: Pixhawk via USB serial (adjust device/baud to your setup)
            conn_url = "serial:///dev/ttyACM0:57600"
        else:
            # Default: SITL
            conn_url = DEF_MAVSDK

        mav = MavsdkClient(conn_url)
        print(f"[MAVSDK] Connecting to {conn_url}")
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

        safety = SafetyManager(
            SafetyConfig(
                min_rel_alt_m=cfg.min_rel_alt_m,
                max_rel_alt_m=cfg.max_rel_alt_m,
                rtl_battery_pct=cfg.rtl_battery_pct,
                enabled=cfg.safety_enabled,
            )
        )
        logger = HardwareLogger(enabled=True)

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
            
            safety.maybe_set_home(tel)

            # ---- Safety: Battery-based RTL ----
            if safety.cfg.enabled and safety.should_rtl(float(tel.battery), mapper.state):
                print(
                    f"[SAFETY] Battery {tel.battery:.1f}% "
                    f"<= {safety.cfg.rtl_battery_pct}%, initiating RTL."
                )
                try:
                    await mav.rtl()
                except Exception:
                    await mav.sys.action.return_to_launch()

                mapper.set_state(FlightState.RTL)

                # Skip NBV + gestures while RTL is active
                await asyncio.sleep(0.5)
                continue
                # Option: break out of loop once RTL is requested
                # break

            # Gestures (ONNX)
            raw_gesture = gestures.detect(frame)
            if hasattr(gestures, "overlay_status"):
                gestures.overlay_status(frame)
            action = mapper.map(raw_gesture)

            if action.kind != GestureActionType.NONE:
                before = mapper.state
                mapper.set_state(
                    await apply_gesture_action(
                        mav, mapper.state, action, tel, takeoff_alt=float(args.takeoff), safety=safety
                    )
                )
                print(f"[GESTURE] {raw_gesture} → {action.kind.name} | {before.name} → {mapper.state.name}")

            # NBV decision
            sf = planner.analyze_frame(frame)
            decision = planner.decide(sf, tel.heading_deg, tel.battery)
            planner.update_history(sf, tel.heading_deg)

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
                now = time.time()

                # Base yaw delta from NBV
                delta = decision.heading_delta_deg

                # If a gesture yaw override is active, use it instead
                yaw_until = getattr(run_sitl, "yaw_until", 0.0)
                if now < yaw_until:
                    delta = float(getattr(run_sitl, "dyaw", 0.0))

                # enforce minimum step so we don't get stuck on 0
                if abs(delta) < 3.0:
                    delta = 3.0 if delta >= 0 else -3.0

                # SAFETY: Damp yaw near minimum altitude
                rel_alt = float(tel.alt) - float(args.takeoff)
                if safety.cfg.enabled and rel_alt < safety.cfg.min_rel_alt_m + 0.5:
                    delta *= 0.4  # reduce yaw aggressiveness when too low

                target_yaw = tel.heading_deg + max(
                    -cfg.max_yaw_delta_deg, min(cfg.max_yaw_delta_deg, delta)
                )
                # wrap to [-180, 180]
                target_yaw = (target_yaw + 180.0) % 360.0 - 180.0

                if not hasattr(run_sitl, "_yaw_lp"):
                    run_sitl._yaw_lp = target_yaw
                alpha = 0.2
                run_sitl._yaw_lp = (1 - alpha) * run_sitl._yaw_lp + alpha * target_yaw

                # Lateral velocity from gestures (default 0)
                vy_cmd = 0.0
                vy_until = getattr(run_sitl, "vy_until", 0.0)
                if now < vy_until:
                    vy_cmd = float(getattr(run_sitl, "vy_cmd", 0.0))
                else:
                    # ensure we don't keep drifting after the burst
                    setattr(run_sitl, "vy_cmd", 0.0)

                cv2.putText(
                    frame,
                    f"hdg:{tel.heading_deg:+.1f} tgt:{run_sitl._yaw_lp:+.1f} "
                    f"alt:{tel.alt:.2f} vy:{vy_cmd:+.2f}",
                    (12, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                # --- Safety overlay (always draw when safety available) ---
                if safety is not None:
                    rel_alt = float(tel.alt) - float(args.takeoff)

                    cv2.putText(
                        frame,
                        f"SAFETY ALT rel:{rel_alt:.1f}m "
                        f"[{safety.cfg.min_rel_alt_m:.1f}-{safety.cfg.max_rel_alt_m:.1f}]",
                        (12, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

                    cv2.putText(
                        frame,
                        f"SAFETY BAT {tel.battery:.1f}% "
                        f"(RTL @{safety.cfg.rtl_battery_pct:.1f}%)",
                        (12, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

                    if not safety.cfg.enabled:
                        cv2.putText(
                            frame,
                            "SAFETY: DISABLED",
                            (12, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )

                try:
                    await mav.sys.offboard.set_velocity_ned(
                        VelocityNedYaw(0.0, vy_cmd, 0.0, run_sitl._yaw_lp)
                    )
                except Exception as e:
                    print(f"[SEND] set_velocity failed: {e}. Exiting loop.")
                    break

            # Novelty-gated capture + dedupe
            now = time.time()
            capture_triggered = False
            dedupe_skipped = False
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
                    capture_triggered = True
                else:
                    print("Duplicate shot skipped")
                    capture_triggered = True
                    dedupe_skipped = True
                t_last_capture = now

            # ---- Hardware / SITL logging ----
            if logger is not None:
                subj = decision.subject  # depending on how you named it
                subj_label = getattr(subj, "label", None) if subj is not None else None
                subj_center = getattr(subj, "center", None) if subj is not None else None
                subj_area = getattr(subj, "area", None) if subj is not None else None

                rec = FrameLog(
                    t_sec=time.time(),
                    alt=float(tel.alt),
                    lat=float(tel.lat),
                    lon=float(tel.lon),
                    heading_deg=float(tel.heading_deg),
                    battery_pct=float(tel.battery),
                    state=mapper.state.name,
                    hardware=bool(getattr(args, "hardware", 0)),
                    gesture_raw=str(raw_gesture.name if hasattr(raw_gesture, "name") else raw_gesture),
                    gesture_action=str(action.kind.name if hasattr(action, "kind") else None),
                    novelty_score=float(decision.novelty_score),
                    heading_delta_deg=float(decision.heading_delta_deg),
                    subject_label=subj_label,
                    subject_center=subj_center,
                    subject_area=subj_area,
                    capture_triggered=capture_triggered,
                    dedupe_skipped=dedupe_skipped,
                )
                logger.log_frame(rec)

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
                        safety=safety
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
                        safety=safety
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
                        safety=safety
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
                        safety=safety
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
                        safety=safety
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
        try:
            if logger is not None:
                logger.close()
        except Exception:
            pass


# ---------- Entrypoint ----------

def main():
    args = parse_args()
    cfg = Config()
    cfg.semantic_nbv = bool(args.semantic_nbv)
    cfg.safety_enabled = bool(args.safety)
    cfg.min_rel_alt_m = float(args.min_rel_alt)
    cfg.max_rel_alt_m = float(args.max_rel_alt)
    cfg.rtl_battery_pct = float(args.rtl_battery)
    if int(args.use_sitl) == 1:
        asyncio.run(run_sitl(cfg, args))
    else:
        run_local(cfg, args)


if __name__ == "__main__":
    main()

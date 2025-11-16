# src/flight/mavsdk_client.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
from mavsdk.action import ActionError


@dataclass
class TelemetrySample:
    lat: float
    lon: float
    alt: float        # relative altitude (m)
    heading_deg: float
    battery: float    # 0.0–1.0


class MavsdkClient:
    def __init__(self, url: str):
        self.url = url
        self.sys = System()
        self._tel_task: Optional[asyncio.Task] = None
        self._telemetry_latest: Optional[TelemetrySample] = None
        self._offboard_started: bool = False
        self._climb_lock = asyncio.Lock()

    # ---------- Connection / setup ----------

    async def connect(self):
        await self.sys.connect(system_address=self.url)

        # Wait until connected
        async for state in self.sys.core.connection_state():
            if state.is_connected:
                break

        # Relax arming checks for SIM ONLY (safe in SITL, NOT on real drones)
        await self.relax_sitl_arming_checks()

        # Throttle telemetry rates hard so the user callback queue doesn't flood
        try:
            await self.sys.telemetry.set_rate_position(0.5)   # 0.5 Hz position
        except Exception:
            pass
        try:
            await self.sys.telemetry.set_rate_battery(0.5)    # 0.5 Hz battery
        except Exception:
            pass
        try:
            await self.sys.telemetry.set_rate_heading(0.5)    # 0.5 Hz heading
        except Exception:
            pass

        # Start background telemetry pull (single loop)
        self._tel_task = asyncio.create_task(self._pull_telemetry_loop())

    async def relax_sitl_arming_checks(self):
        """
        Loosen arming checks for SIMULATION ONLY.
        Do not use on a real vehicle.
        """
        # Allow arming without GPS (common need in SITL)
        try:
            await self.sys.param.set_param_int("COM_ARM_WO_GPS", 1)
        except Exception:
            pass
        # COM_PREARM_MODE may not exist in all PX4 builds; ignore if missing
        try:
            await self.sys.param.set_param_int("COM_PREARM_MODE", 0)
        except Exception:
            pass
        # NOTE: We no longer touch CBRK_GPSFAIL / CBRK_SUPPLY_CHK etc.,
        # because PX4 didn't like those and spammed retries.

    # ---------- Telemetry ----------

    async def _pull_telemetry_loop(self):
        """
        Continuously updates self._telemetry_latest with normalized values.
        Runs in a background task at the rates configured in connect().
        """
        pos_a = self.sys.telemetry.position()
        batt_a = self.sys.telemetry.battery()
        hdg_a = self.sys.telemetry.heading()

        while True:
            try:
                pos = await pos_a.__anext__()
                batt = await batt_a.__anext__()
                hdg = await hdg_a.__anext__()
            except Exception:
                await asyncio.sleep(0.1)
                continue

            # Normalize heading
            try:
                heading_deg = float(getattr(hdg, "heading_deg", hdg))
            except Exception:
                try:
                    heading_deg = float(hdg)
                except Exception:
                    heading_deg = 0.0

            # Normalize battery to [0..1]
            try:
                battery = float(getattr(batt, "remaining_percent", batt))
            except Exception:
                try:
                    battery = float(batt)
                except Exception:
                    battery = 1.0

            # Normalize relative altitude
            try:
                alt = float(getattr(pos, "relative_altitude_m", pos.relative_altitude_m))
            except Exception:
                alt = 0.0

            self._telemetry_latest = TelemetrySample(
                lat=float(getattr(pos, "latitude_deg", 0.0)),
                lon=float(getattr(pos, "longitude_deg", 0.0)),
                alt=alt,
                heading_deg=heading_deg,
                battery=battery,
            )
            # No sleep here: telemetry rates govern how often this runs

    async def telemetry_latest(self) -> Optional[TelemetrySample]:
        return self._telemetry_latest

    async def is_link_alive(self) -> bool:
        """True if PX4 is currently connected."""
        try:
            async for st in self.sys.core.connection_state():
                return bool(st.is_connected)
        except Exception:
            return False

    # ---------- Arm / takeoff ----------

    async def arm_and_takeoff(self, rel_alt_m: float = 3.0, timeout_s: float = 10.0):
        """
        Simple arming + takeoff for SITL.
        We rely on Offboard + climb_to_alt() for actual altitude control.
        """
        # Ensure connected
        async for state in self.sys.core.connection_state():
            if state.is_connected:
                break

        # Give SITL a moment for EKF to settle; we already allow arm w/o GPS.
        await asyncio.sleep(2.0)

        await self.sys.action.set_takeoff_altitude(rel_alt_m)

        # Retry arming a few times
        delay = 1.0
        for attempt in range(1, 6):
            try:
                await self.sys.action.arm()
                break
            except ActionError as e:
                if attempt == 5:
                    raise RuntimeError(
                        "Arming denied by PX4 after retries.\n"
                        f"Original: {e}"
                    )
                print(f"[ARM RETRY] attempt {attempt} failed, retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay *= 1.6

        # Issue takeoff (may be a no-op in SITL; Offboard climb will handle)
        await self.sys.action.takeoff()

    # ---------- Offboard helpers ----------

    async def start_offboard(self):
        """
        Start Offboard mode, seeding an initial velocity setpoint.
        """
        if self._offboard_started:
            return
        try:
            await self.sys.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
            await self.sys.offboard.start()
            self._offboard_started = True
        except OffboardError as e:
            print(f"[OFFBOARD] Failed to start: {e}")
            self._offboard_started = False
            raise

    async def stop_offboard(self):
        if not self._offboard_started:
            return
        try:
            await self.sys.offboard.stop()
        except OffboardError as e:
            print(f"[OFFBOARD] Failed to stop: {e}")
        finally:
            self._offboard_started = False

    async def climb_to_alt(
        self,
        target_rel_alt_m: float,
        vz_m_s: float = -0.8,
        max_seconds: float = 10.0,
    ):
        """
        Climb (or descend) to target relative altitude under Offboard control.
        Uses a lock to avoid overlapping climbs, and ignores trivial (<0.1m) changes.
        """
        if not self._offboard_started:
            raise RuntimeError("Offboard not started; call start_offboard() first.")

        # Check current altitude; ignore trivial requests
        pos_a = self.sys.telemetry.position()
        try:
            pos = await pos_a.__anext__()
            cur = float(getattr(pos, "relative_altitude_m", 0.0))
        except Exception:
            cur = 0.0

        if abs(target_rel_alt_m - cur) < 0.10:
            return

        async with self._climb_lock:
            print(f"[OFFBOARD CLIMB] target={target_rel_alt_m:.2f}m (cur={cur:.2f}m)")
            t0 = asyncio.get_event_loop().time()
            pos_a = self.sys.telemetry.position()  # fresh generator

            while True:
                try:
                    pos = await pos_a.__anext__()
                except Exception:
                    await asyncio.sleep(0.05)
                    continue

                rel_alt = float(getattr(pos, "relative_altitude_m", 0.0))

                # Within tolerance band → stop vertical motion
                if abs(rel_alt - target_rel_alt_m) <= 0.25:
                    try:
                        await self.sys.offboard.set_velocity_ned(
                            VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                        )
                    except Exception:
                        pass
                    print(f"[OFFBOARD CLIMB] Reached ~{rel_alt:.1f} m")
                    break

                # Decide climb/descend velocity (NED: up is negative z)
                if rel_alt < target_rel_alt_m - 0.25:
                    vz = vz_m_s  # negative to climb
                else:
                    vz = +0.5   # gentle descent

                try:
                    await self.sys.offboard.set_velocity_ned(
                        VelocityNedYaw(0.0, 0.0, vz, 0.0)
                    )
                except Exception as e:
                    print(f"[OFFBOARD CLIMB] set_velocity failed: {e}")
                    break

                if asyncio.get_event_loop().time() - t0 > max_seconds:
                    print(f"[OFFBOARD CLIMB] Timeout at {rel_alt:.1f} m; proceeding anyway.")
                    try:
                        await self.sys.offboard.set_velocity_ned(
                            VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                        )
                    except Exception:
                        pass
                    break

                await asyncio.sleep(0.05)

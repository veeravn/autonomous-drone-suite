# src/flight/mock_drone.py

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass


@dataclass
class MockTelemetry:
    alt: float = 3.0
    lat: float = 37.0
    lon: float = -122.0
    heading_deg: float = 0.0
    battery: float = 80.0  # percent


class _MockAction:
    def __init__(self, drone: "MockDrone"):
        self._drone = drone

    async def return_to_launch(self):
        # Simple RTL: just "land"
        await self._drone.land()

    async def land(self):
        await self._drone.land()


class _MockOffboard:
    def __init__(self, drone: "MockDrone"):
        self._drone = drone

    async def set_velocity_ned(self, vel_ned_yaw):
        """
        vel_ned_yaw: VelocityNedYaw( north_m_s, east_m_s, down_m_s, yaw_deg )
        We just integrate altitude and yaw crudely for testing.
        """
        now = time.time()
        dt = now - self._drone._last_cmd_ts
        self._drone._last_cmd_ts = now

        # vel_ned_yaw has attributes: north_m_s, east_m_s, down_m_s, yaw_deg
        down = getattr(vel_ned_yaw, "down_m_s", 0.0)
        yaw = getattr(vel_ned_yaw, "yaw_deg", 0.0)

        # NED: down is positive, so subtract to increase altitude
        self._drone._tel.alt -= down * dt
        if self._drone._tel.alt < 0.0:
            self._drone._tel.alt = 0.0

        # Yaw: just move heading toward commanded yaw
        # In a real system, yaw is yaw rate; here we just snap a bit toward it.
        # Keep [-180, 180]
        curr = self._drone._tel.heading_deg
        target = yaw
        alpha = 0.2
        new_yaw = (1 - alpha) * curr + alpha * target
        new_yaw = (new_yaw + 180.0) % 360.0 - 180.0
        self._drone._tel.heading_deg = new_yaw


class _MockSys:
    def __init__(self, drone: "MockDrone"):
        self.action = _MockAction(drone)
        self.offboard = _MockOffboard(drone)


class MockDrone:
    """
    Drop-in stand-in for MavsdkClient used in SITL loop when --mock-drone=1.

    It implements the subset of methods used in main.py:

      - connect()
      - arm_and_takeoff()
      - start_offboard()
      - climb_to_alt()
      - telemetry_latest()
      - rtl()
      - stop_offboard()
      - takeoff()
      - land()
      - yaw_offset()
      - sys.action.return_to_launch()
      - sys.action.land()
      - sys.offboard.set_velocity_ned(...)
    """

    def __init__(self):
        self._connected = False
        self._armed = False
        self._offboard = False
        self._tel = MockTelemetry()
        self._last_cmd_ts = time.time()
        self.sys = _MockSys(self)

    # ----------------- Connection lifecycle -----------------

    async def connect(self):
        # Simulate a short connection delay
        await asyncio.sleep(0.1)
        self._connected = True
        print("[MOCK] Connected")

    async def arm_and_takeoff(self, rel_alt_m: float):
        """
        Equivalent to MavsdkClient.arm_and_takeoff(rel_alt_m=...)
        We'll just set alt to the requested value.
        """
        self._armed = True
        self._tel.alt = float(rel_alt_m)
        print(f"[MOCK] Arm + takeoff to ~{rel_alt_m:.1f} m")
        await asyncio.sleep(0.1)

    async def start_offboard(self):
        self._offboard = True
        print("[MOCK] Offboard started")
        await asyncio.sleep(0.05)

    async def stop_offboard(self):
        self._offboard = False
        print("[MOCK] Offboard stopped")
        await asyncio.sleep(0.05)

    # ----------------- Basic actions -----------------

    async def takeoff(self, target_alt_m: float):
        self._armed = True
        self._tel.alt = float(target_alt_m)
        print(f"[MOCK] TAKEOFF → {target_alt_m:.1f} m")
        await asyncio.sleep(0.1)

    async def land(self):
        print("[MOCK] LAND requested")
        self._tel.alt = 0.0
        self._armed = False
        await asyncio.sleep(0.1)

    async def rtl(self):
        print("[MOCK] RTL requested")
        # For mock, we just call land()
        await self.land()

    async def climb_to_alt(self, target_alt_m: float, vz_m_s: float = -0.8, max_seconds: float = 12.0):
        """
        Very crude altitude animation.
        """
        print(f"[MOCK] climb_to_alt → {target_alt_m:.2f} m")
        self._tel.alt = float(target_alt_m)
        await asyncio.sleep(0.1)

    async def yaw_offset(self, dyaw_deg: float):
        """
        Add a yaw offset to the current heading.
        """
        self._tel.heading_deg += float(dyaw_deg)
        self._tel.heading_deg = (self._tel.heading_deg + 180.0) % 360.0 - 180.0
        print(f"[MOCK] Yaw offset {dyaw_deg:+.1f}° → heading {self._tel.heading_deg:+.1f}°")
        await asyncio.sleep(0.01)

    # ----------------- Telemetry -----------------

    async def telemetry_latest(self):
        """
        Match main.py expectations:
          tel.alt
          tel.lat
          tel.lon
          tel.heading_deg
          tel.battery
        """
        # Slowly drain battery
        self._tel.battery = max(0.0, self._tel.battery - 0.005)
        await asyncio.sleep(0.01)
        return self._tel

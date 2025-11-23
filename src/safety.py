from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math


@dataclass
class SafetyConfig:
    """
    Global safety limits for the drone.

    All altitudes are *relative* to takeoff altitude in meters.
    """
    min_rel_alt_m: float = 1.0      # don't go lower than this in offboard
    max_rel_alt_m: float = 15.0     # don't climb above this
    rtl_battery_pct: float = 20.0   # trigger RTL below this battery %
    enabled: bool = True


class SafetyManager:
    """
    Safety layer sitting between NBV / gestures and MAVSDK.

    For now this handles:
      - altitude clamping (min/max)
      - low-battery RTL trigger

    Later you can extend this with:
      - horizontal geofence
      - collision avoidance (rangefinder/depth)
    """

    def __init__(self, cfg: SafetyConfig) -> None:
        self.cfg = cfg
        self.home_lat: Optional[float] = None
        self.home_lon: Optional[float] = None

    # ---- Home position (for future geofence) ----

    def maybe_set_home(self, tel) -> None:
        """
        Store home lat/lon once when telemetry is valid.
        """
        if self.home_lat is not None:
            return
        lat = getattr(tel, "lat", None)
        lon = getattr(tel, "lon", None)
        if lat is None or lon is None:
            return
        self.home_lat = float(lat)
        self.home_lon = float(lon)

    # ---- Altitude constraints ----

    def clamp_altitude(self, target_rel_alt_m: float) -> float:
        """
        Clamp a requested relative altitude to [min_rel_alt_m, max_rel_alt_m].
        """
        if not self.cfg.enabled:
            return max(0.0, target_rel_alt_m)

        t = max(self.cfg.min_rel_alt_m, min(self.cfg.max_rel_alt_m, target_rel_alt_m))
        return max(0.0, t)

    # ---- Battery-based RTL ----

    def should_rtl(self, battery_pct: float, state) -> bool:
        """
        Decide if we should trigger RTL based on battery level and current state.
        """
        if not self.cfg.enabled:
            return False

        try:
            from .gestures.mapper import FlightState  # local import to avoid cycles
        except Exception:
            FlightState = None

        # don't spam RTL if already idle/RTL
        if FlightState is not None and state in (FlightState.IDLE, FlightState.RTL):
            return False

        return battery_pct <= self.cfg.rtl_battery_pct

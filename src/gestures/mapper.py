# src/gestures/mapper.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


# ---------- Enums ----------


class RawGestureType(Enum):
    """
    Low-level gesture labels coming from the vision pipeline.

    These are what GestureController.detect(...) returns.
    """
    NONE = auto()
    FIST = auto()
    THUMB_UP = auto()
    THUMB_DOWN = auto()
    OPEN_PALM = auto()
    POINT_LEFT = auto()
    POINT_RIGHT = auto()
    PALM_LEFT = auto()
    PALM_RIGHT = auto()
    # Add more as needed (e.g., V_SIGN, POINT_LEFT, etc.)


class GestureActionType(Enum):
    """
    High-level actions understood by the flight layer.
    These are what apply_gesture_action(...) in main.py handles.
    """
    NONE = auto()
    TAKEOFF = auto()
    ALT_OFFSET = auto()
    HOLD = auto()
    RESUME = auto()
    LAND = auto()
    RTL = auto()
    YAW_OFFSET = auto()   # new: short yaw nudge left/right
    STRAFE = auto()       # new: short lateral strafe left/right

class FlightState(Enum):
    """
    Simple flight mode/state used in the gesture logic.
    """
    IDLE = auto()
    CAPTURING = auto()
    PAUSE = auto()
    RTL = auto()


# ---------- Dataclasses ----------


@dataclass
class GestureAction:
    """
    A mapped action coming out of GestureMapper.

    kind:
      - One of GestureActionType (e.g., TAKEOFF, ALT_OFFSET, HOLD, STRAFE, etc.)

    dz:
      - For ALT_OFFSET, the desired altitude change in meters.
        Positive = climb, negative = descend.

    vy:
      - For STRAFE, lateral velocity (m/s) in NED (Y) axis.
        Positive ≈ "right", negative ≈ "left" (we'll interpret consistently).

    dyaw:
      - For YAW_OFFSET, yaw offset in degrees (short override).
        Positive = yaw right, negative = yaw left.
    """    
    kind: GestureActionType
    dz: float = 0.0
    vy: float = 0.0
    dyaw: float = 0.0


# ---------- Mapper ----------


class GestureMapper:
    """
    Maps RawGestureType (from the vision/ONNX pipeline) to GestureAction,
    with awareness of the current FlightState.

    Typical mapping (designed to be easy to remember):

      - THUMB_UP:
          * If IDLE        -> TAKEOFF
          * Else           -> ALT_OFFSET +0.8 m

      - THUMB_DOWN:
          * If CAPTURING or PAUSE -> ALT_OFFSET -0.6 m
          * Else                  -> NONE

      - OPEN_PALM:
          * If CAPTURING   -> HOLD (pause offboard)
          * If PAUSE       -> RESUME (restart offboard)
          * Else           -> NONE

      - FIST:
          * If not IDLE    -> LAND
          * Else           -> NONE

      - NONE / no gesture:
          * Always -> NONE

    A small “edge-trigger” debouncing is applied so that
    you don’t fire the same action every frame while holding a gesture.
    """

    def __init__(self):
        self.state: FlightState = FlightState.IDLE

        # For simple debouncing: only trigger when the raw gesture changes
        self._last_raw: RawGestureType = RawGestureType.NONE

    # ---- state helpers ----

    def set_state(self, new_state: FlightState):
        self.state = new_state

    # ---- utility for keyboard shortcuts ----

    def type_to_action(self, kind: GestureActionType, dz: float = 0.0) -> GestureAction:
        """
        Helper used by keyboard controls in main.py to synthesize
        GestureAction instances directly.
        """
        return GestureAction(kind=kind, dz=dz)

    # ---- main mapping function ----

    def map(self, raw: Optional[RawGestureType]) -> GestureAction:
        """
        Map a raw gesture into a high-level GestureAction, taking
        current FlightState into account.

        This is called once per frame:

            raw_gesture = gestures.detect(frame)
            action = mapper.map(raw_gesture)
        """

        # Normalize None -> RawGestureType.NONE
        if raw is None:
            raw = RawGestureType.NONE

        # Simple edge-trigger: only fire when the gesture CHANGES.
        # If user holds the same gesture across frames, we only trigger once.
        if raw == self._last_raw:
            # Repeat of same gesture: usually do nothing.
            self._last_raw = raw
            return GestureAction(kind=GestureActionType.NONE)

        self._last_raw = raw

        # Now interpret the new raw gesture
        if raw == RawGestureType.NONE:
            return GestureAction(GestureActionType.NONE)

        # ---------- THUMB_UP ----------
        if raw == RawGestureType.THUMB_UP:
            if self.state == FlightState.IDLE:
                # From idle, thumb up = TAKEOFF
                return GestureAction(GestureActionType.TAKEOFF)
            else:
                # In the air, thumb up = climb a bit
                return GestureAction(GestureActionType.ALT_OFFSET, dz=+0.8)

        # ---------- THUMB_DOWN ----------
        if raw == RawGestureType.THUMB_DOWN:
            if self.state in (FlightState.CAPTURING, FlightState.PAUSE):
                # Thumb down = descend a bit
                return GestureAction(GestureActionType.ALT_OFFSET, dz=-0.6)
            else:
                return GestureAction(GestureActionType.NONE)

        # ---------- OPEN_PALM ----------
        if raw == RawGestureType.OPEN_PALM:
            if self.state == FlightState.CAPTURING:
                # Open hand while capturing = pause
                return GestureAction(GestureActionType.HOLD)
            if self.state == FlightState.PAUSE:
                # Open hand while paused = resume
                return GestureAction(GestureActionType.RESUME)
            return GestureAction(GestureActionType.NONE)

        # ---------- FIST ----------
        if raw == RawGestureType.FIST:
            if self.state != FlightState.IDLE:
                # Fist in the air = land
                return GestureAction(GestureActionType.LAND)
            return GestureAction(GestureActionType.NONE)
        
        # ---------- POINT_LEFT / POINT_RIGHT ----------
        if raw == RawGestureType.POINT_LEFT:
            if self.state == FlightState.CAPTURING:
                # Short yaw nudge left (~20 degrees)
                return GestureAction(GestureActionType.YAW_OFFSET, dyaw=-20.0)
            return GestureAction(GestureActionType.NONE)

        if raw == RawGestureType.POINT_RIGHT:
            if self.state == FlightState.CAPTURING:
                # Short yaw nudge right (~20 degrees)
                return GestureAction(GestureActionType.YAW_OFFSET, dyaw=+20.0)
            return GestureAction(GestureActionType.NONE)

        # Fallback
        return GestureAction(GestureActionType.NONE)

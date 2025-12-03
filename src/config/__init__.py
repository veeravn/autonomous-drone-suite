# src/config/__init__.py

from __future__ import annotations

from .models import MODEL_PATHS, get as get_model  # if/when you add models.py
from .thresholds import THRESHOLDS
from .camera import CAMERA
from .jetson import JETSON_CONFIG
from .agent import AGENT_CONFIG


class Config:
    """
    Backwards-compatible configuration object used by main.py.

    It is built on top of the new module-based config:
      - AGENT_CONFIG
      - THRESHOLDS
      - CAMERA
    """

    def __init__(self):
        # Semantic NBV default
        self.semantic_nbv: bool = bool(
            AGENT_CONFIG.get("semantic_nbv_default", True)
        )

        # Safety defaults (these will be overridden in main() from CLI)
        self.safety_enabled: bool = bool(
            AGENT_CONFIG.get("safety_enabled_default", True)
        )
        self.min_rel_alt_m: float = float(
            THRESHOLDS.get("min_rel_alt_m", 1.0)
        )
        self.max_rel_alt_m: float = float(
            THRESHOLDS.get("max_rel_alt_m", 15.0)
        )
        self.rtl_battery_pct: float = float(
            THRESHOLDS.get("rtl_battery_pct", 20.0)
        )

        # Capture & NBV
        self.capture_period_s: float = float(
            AGENT_CONFIG.get("capture_period_s", 3.0)
        )
        self.max_yaw_delta_deg: float = float(
            AGENT_CONFIG.get("max_yaw_delta_deg", 45.0)
        )


__all__ = [
    "MODEL_PATHS",
    "get_model",
    "THRESHOLDS",
    "CAMERA",
    "JETSON_CONFIG",
    "AGENT_CONFIG",
    "Config",
]

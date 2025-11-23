from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


@dataclass
class FrameLog:
    # Time
    t_sec: float

    # Telemetry
    alt: float
    lat: float
    lon: float
    heading_deg: float
    battery_pct: float

    # Flight state / mode
    state: str
    hardware: bool

    # Gestures
    gesture_raw: Optional[str]
    gesture_action: Optional[str]

    # NBV / semantic info
    novelty_score: float
    heading_delta_deg: float
    subject_label: Optional[str]
    subject_center: Optional[Tuple[float, float]]
    subject_area: Optional[float]

    # Capture / dedupe
    capture_triggered: bool
    dedupe_skipped: bool


class HardwareLogger:
    """
    Simple JSONL logger for hardware/SITL flights.
    Writes one JSON object per frame to logs/flight_*.jsonl.
    """

    def __init__(self, enabled: bool = True, log_dir: str = "logs") -> None:
        self.enabled = enabled
        self.log_dir = log_dir
        self._fh = None

        if not self.enabled:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.log_dir, f"flight_{stamp}.jsonl")
        self._fh = open(path, "a", buffering=1)
        print(f"[LOG] HardwareLogger writing to {path}")

    def log_frame(self, record: FrameLog) -> None:
        if not self.enabled or self._fh is None:
            return
        try:
            self._fh.write(json.dumps(asdict(record)) + "\n")
        except Exception as e:
            print(f"[LOG] Failed to write frame log: {e}")

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

from __future__ import annotations
import time
from dataclasses import dataclass


@dataclass
class Telemetry:
    lat: float
    lon: float
    alt: float
    heading_deg: float
    battery: float
    
    @staticmethod
    def fake_stream():
        heading = 0.0
        while True:
            yield Telemetry(lat=45.0, lon=-93.0, alt=30.0, heading_deg=heading, battery=0.95)
            heading = (heading + 5.0) % 360.0
            time.sleep(0.1)
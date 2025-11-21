from dataclasses import dataclass
from typing import List, Tuple, Optional

LatLon = Tuple[float, float]

@dataclass
class Config:
    # Video / capture
    camera_index: int = 0
    capture_period_s: float = 1.0

    # NBV weights
    w_novelty: float = 2.0
    w_turn_cost: float = 1.0
    w_energy_cost: float = 1.0
    w_geo_penalty: float = 3.0
    semantic_nbv: bool = True

    # Dedupe
    phash_thresh: float = 0.12
    pos_meters: float = 5.0

    # Offboard / yaw limits
    max_yaw_rate_deg_s: float = 45.0
    max_yaw_delta_deg: float = 90.0

    # Geofence polygon (lat, lon); replace with your test field
    geofence_poly: Optional[List[LatLon]] = None

    def __post_init__(self):
        if self.geofence_poly is None:
            # ~100m box (example)
            self.geofence_poly = [
                (45.0005, -93.0005),
                (45.0005, -92.9995),
                (44.9995, -92.9995),
                (44.9995, -93.0005),
            ]

# src/config/thresholds.py
from typing import List, Tuple

LatLon = Tuple[float, float]

# Default ~100m test box (copied from your old Config.__post_init__)
DEFAULT_GEOFENCE_POLY: List[LatLon] = [
    (45.0005, -93.0005),
    (45.0005, -92.9995),
    (44.9995, -92.9995),
    (44.9995, -93.0005),
]

THRESHOLDS = {
    # YOLO
    "yolo_conf": 0.35,
    "yolo_iou": 0.45,

    # CLIP
    "clip_top_k": 5,
    "clip_similarity_threshold": 0.20,

    # Gestures
    "gesture_smoothing": 5,        # frames
    "gesture_confidence": 0.70,
    "gesture_debounce_ms": 300,

    # NBV / semantic
    "novelty_threshold": 0.14,     # used as NOVELTY_CAPTURE_MIN in main
    "yaw_step_deg": 15.0,

    # Deduplication
    "pos_meters": 5.0,
    "phash_threshold": 9,

    # Safety (defaults, can be overridden by CLI/main)
    "safety_enabled_default": True,
    "min_rel_alt_m": 1.0,
    "max_rel_alt_m": 15.0,
    "rtl_battery_pct": 20.0,

    # Geofence
    "geofence_poly_default": DEFAULT_GEOFENCE_POLY,
}

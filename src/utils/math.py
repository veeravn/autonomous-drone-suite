from __future__ import annotations
import numpy as np


def wrap_angle_deg(angle: float) -> float:
    """Wrap angle to [-180, 180)."""
    a = (angle + 180.0) % 360.0 - 180.0
    return a


def l2(p, q):
    p = np.asarray(p); q = np.asarray(q)
    return float(np.linalg.norm(p - q))
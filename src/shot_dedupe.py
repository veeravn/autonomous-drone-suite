from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


import numpy as np
import cv2
from PIL import Image
import imagehash
from scipy.spatial.distance import hamming as hamming_dist

@dataclass
class Shot:
    phash: imagehash.ImageHash
    pos: Tuple[float, float, float] # lat, lon, alt (or x,y,z)

class ShotDeduper:
    def __init__(self, dist_thresh: float = 0.12, pos_m: float = 5.0):
        self.dist_thresh = dist_thresh
        self.pos_m = pos_m
        self.db: List[Shot] = []
    @staticmethod
    def _phash(img_bgr) -> imagehash.ImageHash:
        return imagehash.phash(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))

    def _pos_close(self, p, q) -> bool:
        # Plain L2 in meters if you pass (x,y,z); for GPS plug in a proper haversine if needed
        p = np.asarray(p); q = np.asarray(q)
        return float(np.linalg.norm(p - q)) < self.pos_m
    def is_duplicate(self, img_bgr, pos) -> bool:
        h = self._phash(img_bgr)
        for s in self.db:
            # Hamming on raw boolean array
            d = hamming_dist(h.hash.flatten(), s.phash.hash.flatten())
            if d < self.dist_thresh and self._pos_close(pos, s.pos):
                return True
            return False
    def add(self, img_bgr, pos):
        self.db.append(Shot(phash=self._phash(img_bgr), pos=pos))
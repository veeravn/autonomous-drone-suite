# src/gesture_control.py
from __future__ import annotations

from dataclasses import dataclass
import time
import collections
from typing import Optional

import cv2
import numpy as np

from .gestures.mapper import RawGestureType
from .onnx_gesture import HandLandmarks, ONNXHandRuntime


@dataclass
class DetectedGesture:
    kind: RawGestureType
    score: float
    landmarks: Optional[HandLandmarks] = None


class GestureController:
    """
    GestureController that uses ONNX Runtime hand models instead of MediaPipe.

    Public API:
      - detect(frame_bgr) -> RawGestureType | None
      - overlay_status(frame_bgr) -> draws status text + landmarks

    This is used by main.py like:
        raw_gesture = gestures.detect(frame)
        action = mapper.map(raw_gesture)
    """

    def __init__(self, enable: bool = True):
        self.enabled: bool = False
        self.last_status: str = "DISABLED"
        self.runtime: Optional[ONNXHandRuntime] = None
        self.last_detected: Optional[DetectedGesture] = None
        self._last_emit_time = 0.0
        self.cooldown_s = 10  # minimum seconds between non-NONE gestures
        self._history = collections.deque(maxlen=5)  # last 5 raw gesture kinds

        if not enable:
            print("[GESTURES] Disabled by config (--use-gestures 0).")
            return

        try:
            self.runtime = ONNXHandRuntime(
                det_model_path="models/hand_det.onnx",
                lm_model_path="models/hand_landmarks.onnx",
                device="cpu",
                detection_mode="cv",
            )
            self.enabled = True
            self.last_status = "ENABLED"
            print("[GESTURES] ONNX hand runtime initialized.")
        except Exception as e:
            print(f"[GESTURES] Failed to initialize ONNX hand runtime: {e}")
            print("[GESTURES] Gestures will be OFF for this run.")
            self.enabled = False
            self.runtime = None
            self.last_status = "INIT_FAIL"

    # ---------- Main hook used by main.py ----------

    def detect(self, frame_bgr) -> RawGestureType:
        """
        Run ONNX hand detection + landmarks, then classify into a RawGestureType.

        Returns:
            RawGestureType value, e.g.
            - RawGestureType.THUMB_UP
            - RawGestureType.THUMB_DOWN
            - RawGestureType.OPEN_PALM
            - RawGestureType.FIST
            - RawGestureType.NONE
        """
        # If gestures are disabled, always output NONE
        if not getattr(self, "enabled", False):
            return RawGestureType.NONE

        if frame_bgr is None:
            return RawGestureType.NONE

        if self.runtime is None:
            # Failed initialization earlier
            return RawGestureType.NONE

        # 1) Run detector + landmarks
        lm = self.runtime.detect_and_landmarks(frame_bgr)
        if lm is None:
            # No hand found / landmarks failed
            self.last_detected = None
            return RawGestureType.NONE

        # Optional: if you want to use lm.score as a confidence gate
        # (set something like self.min_conf = 0.3 in __init__)
        min_conf = getattr(self, "min_conf", 0.0)
        if lm.score < min_conf:
            self.last_detected = None
            return RawGestureType.NONE

        # 2) Classify based on landmark geometry
        raw = self._classify_raw_gesture(lm)

        # Store for overlay/debug
        self.last_detected = DetectedGesture(
            kind=raw,
            score=float(lm.score),
            landmarks=lm,
        )

        # 3) Optional: smoothing / cooldown
        # If you already have a smoother or cooldown logic, plug it in here
        # instead of returning `raw` directly.
        #
        # Example:
        #   raw_smoothed = self._smoother.update(raw)
        #   return raw_smoothed

        return raw


    def overlay_status(self, frame_bgr):
        """
        Draw small status text + landmarks overlay on the frame.
        """
        h, w, _ = frame_bgr.shape
        status = "ONNX:OFF" if not self.enabled else "ONNX:ON"
        if self.last_detected is not None:
            status += f" last:{self.last_detected.kind.name} ({self.last_detected.score:.2f})"
        else:
            status += " last:NONE"

        cv2.putText(
            frame_bgr,
            status,
            (12, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Optionally visualize landmarks
        if self.last_detected and self.last_detected.landmarks is not None:
            pts = self.last_detected.landmarks.points
            for (x, y, _) in pts:
                cx = int(x * w)
                cy = int(y * h)
                cv2.circle(frame_bgr, (cx, cy), 2, (0, 255, 0), -1)

    # ---------- Gesture heuristics ----------

    def _classify_from_landmarks(self, lm: HandLandmarks) -> tuple[RawGestureType, float]:
        """
        Very simple gesture classification from 21 landmarks.

        Assumes landmark indexing compatible with Mediapipe-style hands:
          0  - wrist
          4  - thumb tip
          8  - index tip
          12 - middle tip
          16 - ring tip
          20 - pinky tip

        We'll compute:
          - "open_score": how spread fingers are from the wrist
          - thumb direction: left/right relative to index tip
        Then map to RawGestureType:
          - FIST
          - THUMB_UP
          - THUMB_DOWN
          - OPEN_PALM
          - NONE
        """

        pts = lm.points  # (21, 3) normalized [0,1]

        # Safety: if wrong shape, bail
        if pts.shape != (21, 3):
            return RawGestureType.NONE, 0.0

        wrist = pts[0]
        # fingertips: thumb, index, middle, ring, pinky
        tips = pts[[4, 8, 12, 16, 20]]

        # distance of fingertips from wrist in x-y space
        dists = np.linalg.norm(tips[:, :2] - wrist[:2], axis=1)
        open_score = float(dists.mean())

        thumb_tip = pts[4]
        index_tip = pts[8]

        # thumb left/right relative to index on x-axis
        thumb_left = thumb_tip[0] < index_tip[0]
        thumb_right = thumb_tip[0] > index_tip[0]

        # Bring in the enum
        from .gestures.mapper import RawGestureType as RGT

        # Heuristic thresholds – tweak to taste:
        #  - small open_score => fist
        #  - large open_score with thumb right => thumbs up
        #  - large open_score with thumb left => thumbs down
        #  - moderately open => open palm

        if open_score < 0.05:
            return RGT.FIST, open_score
        if open_score > 0.12 and thumb_right:
            return RGT.THUMB_UP, open_score
        if open_score > 0.12 and thumb_left:
            return RGT.THUMB_DOWN, open_score
        if open_score > 0.08:
            return RGT.OPEN_PALM, open_score

        return RGT.NONE, open_score

    def _finger_extended(self, pts: np.ndarray, tip_idx: int, pip_idx: int, thresh: float = 0.04) -> bool:
        """
        Returns True if a finger is extended (tip above PIP in image coords).

        pts: (21, 3) array
        tip_idx: landmark index of the fingertip
        pip_idx: landmark index of the PIP joint (second joint from hand)
        """
        tip_y = pts[tip_idx, 1]
        pip_y = pts[pip_idx, 1]
        # In image coords, smaller y = higher (more "up")
        return tip_y < pip_y - thresh

    def _classify_raw_gesture(self, lm: HandLandmarks) -> RawGestureType:
        """
        Classify a raw gesture from hand landmarks.

        Returns one of:
          - RawGestureType.FIST
          - RawGestureType.OPEN_PALM
          - RawGestureType.THUMB_UP
          - RawGestureType.THUMB_DOWN
          - RawGestureType.NONE
        """
        pts = lm.points  # (21, 3) normalized [0,1] in full-frame coords

        if pts.shape != (21, 3):
            return RawGestureType.NONE

        wrist = pts[0]
        wrist_y = wrist[1]

        # --- Finger extended flags (index, middle, ring, pinky) ---
        index_ext = self._finger_extended(pts, tip_idx=8, pip_idx=6)
        middle_ext = self._finger_extended(pts, tip_idx=12, pip_idx=10)
        ring_ext = self._finger_extended(pts, tip_idx=16, pip_idx=14)
        pinky_ext = self._finger_extended(pts, tip_idx=20, pip_idx=18)

        non_thumb_extended = sum([index_ext, middle_ext, ring_ext, pinky_ext])

        # --- Thumb "extended" & direction ---
        thumb_tip = pts[4]
        thumb_base = pts[1]
        thumb_vec = thumb_tip[:2] - thumb_base[:2]
        thumb_len = float(np.linalg.norm(thumb_vec))

        # Approximate average length of index, middle, ring fingers (MCP -> tip)
        index_len = float(np.linalg.norm(pts[8, :2] - pts[5, :2]))
        middle_len = float(np.linalg.norm(pts[12, :2] - pts[9, :2]))
        ring_len = float(np.linalg.norm(pts[16, :2] - pts[13, :2]))

        avg_finger_len = max(1e-5, (index_len + middle_len + ring_len) / 3.0)

        thumb_extended = thumb_len > 0.6 * avg_finger_len

        thumb_tip_y = thumb_tip[1]
        # thresholds: how far above/below wrist counts as "up" / "down"
        up_thresh = 0.04
        down_thresh = 0.04

        thumb_up = thumb_extended and (thumb_tip_y < wrist_y - up_thresh)
        thumb_down = thumb_extended and (thumb_tip_y > wrist_y + down_thresh)

        # --- Gesture patterns (order matters) ---

        # 1) OPEN PALM: all non-thumb fingers extended
        if non_thumb_extended >= 4:
            # Use average x-position of landmarks to decide left/right
            center_x = float(np.mean(pts[:, 0]))
            if center_x < 0.4:
                return RawGestureType.PALM_LEFT
            elif center_x > 0.6:
                return RawGestureType.PALM_RIGHT
            else:
                return RawGestureType.OPEN_PALM
        
        # 1b) POINT_LEFT / POINT_RIGHT:        
        # index extended, others curled, finger mostly horizontal
        index_tip = pts[8]
        index_base = pts[5]
        idx_vec = index_tip[:2] - index_base[:2]
        dx = float(idx_vec[0])
        dy = float(idx_vec[1])

        # We are already past the OPEN_PALM / PALM_LEFT / PALM_RIGHT check,
        # so non_thumb_extended < 4 here.
        # Heuristic: mostly horizontal and non-trivial length.
        # Your debug shows dx values in the 25–70 range, so use ~15 as a floor.
        mostly_horizontal = abs(dx) > abs(dy) * 0.5 and abs(dx) > 15.0

        if mostly_horizontal:
            # Optional: temporary debug
            # print(f"[POINT DETECT] dx={dx:.3f}, dy={dy:.3f}, horiz={mostly_horizontal}")
            if dx < 0.0:
                return RawGestureType.POINT_LEFT
            else:
                return RawGestureType.POINT_RIGHT
        


        # 2) FIST: no non-thumb fingers extended and thumb not clearly up/down
        if non_thumb_extended == 0 and not (thumb_up or thumb_down):
            return RawGestureType.FIST

        # 3) THUMB UP: thumb up, others mostly curled
        if thumb_up and non_thumb_extended <= 1:
            return RawGestureType.THUMB_UP

        # 4) THUMB DOWN: thumb down, others mostly curled
        if thumb_down and non_thumb_extended <= 1:
            return RawGestureType.THUMB_DOWN

        # If we get here, we don't have a clear pattern
        return RawGestureType.NONE
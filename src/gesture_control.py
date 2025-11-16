# src/gesture_control.py
from __future__ import annotations

from dataclasses import dataclass
import time
import collections
from typing import Optional

import cv2
import numpy as np

from .onnx_gesture import ONNXHandRuntime, HandLandmarks
from .gestures.mapper import RawGestureType  # your existing enum


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
                device="cpu",  # can switch to "cuda" if you have GPU + CUDA builds
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

    def detect(self, frame_bgr) -> Optional[RawGestureType]:
        """
        Returns a RawGestureType or None.
        Applies:
          - Landmark ONNX
          - Heuristic classification
          - Temporal smoothing (majority over last few frames)
          - Cooldown so we don't trigger every frame
        """
        if not self.enabled or self.runtime is None:
            self.last_detected = None
            return None

        lm = self.runtime.detect_and_landmarks(frame_bgr)
        if lm is None:
            # no hand found / bad output
            self._history.append(RawGestureType.NONE)
        else:
            kind, score = self._classify_from_landmarks(lm)

            # Reject low-confidence / ambiguous poses by mapping to NONE
            if score < 0.10:
                kind = RawGestureType.NONE

            self._history.append(kind)

        # --- temporal smoothing: majority vote over last K frames ---
        if not self._history:
            stable = RawGestureType.NONE
        else:
            counts = {}
            for k in self._history:
                counts[k] = counts.get(k, 0) + 1
            stable = max(counts.items(), key=lambda kv: kv[1])[0]

        # --- cooldown: only emit non-NONE every cooldown_s seconds ---
        now = time.time()
        emit_kind = stable

        if emit_kind != RawGestureType.NONE:
            if now - self._last_emit_time < self.cooldown_s:
                # Too soon; ignore this gesture
                emit_kind = RawGestureType.NONE
            else:
                self._last_emit_time = now

        # Update last_detected for overlay
        if emit_kind == RawGestureType.NONE:
            self.last_detected = None
            return None

        # We don't track score here after smoothing, so just store 1.0
        from .gestures.mapper import RawGestureType as RGT
        fake_score = 1.0
        self.last_detected = DetectedGesture(
            kind=emit_kind, score=fake_score, landmarks=lm if lm is not None else None
        )
        return emit_kind


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

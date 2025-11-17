# src/onnx_gesture.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class HandLandmarks:
    """
    21 keypoints, each (x, y, z) in normalized image coordinates [0,1].

    x, y are normalized to the *full original frame* (not just the crop),
    so downstream gesture logic can keep assuming [0,1] in the full image.
    """
    points: np.ndarray  # shape (21, 3)
    score: float        # overall confidence (heuristic)


class ONNXHandRuntime:
    """
    Hand detector + landmark wrapper.

    Pipeline (Phase 2 A):

      1. Detect a hand in the full frame (CV-based detector by default).
      2. Crop the hand ROI.
      3. Run the landmark ONNX model on the crop.
      4. Reproject landmarks back into full-frame [0,1] coordinates.

    If detection fails, we gracefully fall back to running landmarks on the
    full frame (Phase 1 behavior), so existing code continues to work.

    Later, you can switch `detection_mode="onnx"` and plug in a real
    ONNX detector model at `det_model_path`.
    """

    def __init__(
        self,
        det_model_path: str = "models/hand_det.onnx",
        lm_model_path: str = "models/hand_landmarks.onnx",
        device: str = "cpu",
        detection_mode: str = "cv",   # "cv", "onnx", or "none"
    ):
        # ---- Landmark model (required) ----
        if not os.path.exists(lm_model_path):
            raise FileNotFoundError(f"Hand landmark model not found: {lm_model_path}")

        providers = ["CPUExecutionProvider"]
        if device.lower() == "cuda":
            providers.insert(0, "CUDAExecutionProvider")

        self.lm_sess = ort.InferenceSession(lm_model_path, providers=providers)
        self.lm_in_name = self.lm_sess.get_inputs()[0].name
        self.lm_out_name = self.lm_sess.get_outputs()[0].name

        # ---- Detector model (optional) ----
        self.det_model_path = det_model_path
        self.detection_mode = detection_mode.lower().strip()
        self.det_sess = None
        self.det_in_name = None
        self.det_out_name = None

        if self.detection_mode == "onnx":
            if os.path.exists(det_model_path):
                try:
                    self.det_sess = ort.InferenceSession(det_model_path, providers=providers)
                    self.det_in_name = self.det_sess.get_inputs()[0].name
                    self.det_out_name = self.det_sess.get_outputs()[0].name
                    print(f"[ONNX] Hand detector loaded from {det_model_path}")
                except Exception as e:
                    print(f"[ONNX] Failed to load detector model {det_model_path}: {e}")
                    self.det_sess = None
            else:
                print(f"[ONNX] Detector model path not found: {det_model_path}. "
                      "Falling back to CV-based detector.")
                self.detection_mode = "cv"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_and_landmarks(self, frame_bgr) -> Optional[HandLandmarks]:
        """
        Detect a hand and run landmarks on the cropped ROI.

        Returns:
          HandLandmarks(points=(21,3), score≈[0..1]) or None on failure.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w, _ = frame_bgr.shape

        # 1) Try to get a hand bounding box
        bbox = None  # (x1, y1, x2, y2) in pixel coords

        if self.detection_mode == "onnx" and self.det_sess is not None:
            bbox = self._detect_hand_onnx(frame_bgr, w, h)
        elif self.detection_mode == "cv":
            bbox = self._detect_hand_cv(frame_bgr, w, h)

        # 2) Crop ROI; if detection failed, fall back to full frame
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(w - 1, int(x1)))
            x2 = max(0, min(w, int(x2)))
            y1 = max(0, min(h - 1, int(y1)))
            y2 = max(0, min(h, int(y2)))

            if x2 <= x1 or y2 <= y1:
                roi = frame_bgr
                roi_bbox = (0, 0, w, h)
                det_score = 0.0
            else:
                roi = frame_bgr[y1:y2, x1:x2]
                roi_bbox = (x1, y1, x2, y2)
                det_score = 1.0
        else:
            # Full-frame fallback (Phase 1 behavior)
            roi = frame_bgr
            roi_bbox = (0, 0, w, h)
            det_score = 0.2   # low-ish confidence for "no real detector"

        # 3) Run landmark ONNX on ROI
        lm = self._run_landmarks_on_roi(roi, roi_bbox, frame_wh=(w, h))
        if lm is None:
            return None

        # Combine detection + landmark "confidence" into one heuristic score
        lm.score = float(min(1.0, max(0.0, 0.5 * det_score + 0.5 * lm.score)))
        return lm

    # ------------------------------------------------------------------
    # Landmark helper
    # ------------------------------------------------------------------
    def _run_landmarks_on_roi(
        self,
        roi_bgr: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        frame_wh: Tuple[int, int],
    ) -> Optional[HandLandmarks]:
        """
        Run the landmark model on a cropped ROI and reproject its
        normalized coordinates back into full-frame [0,1].

        roi_bbox = (x1, y1, x2, y2) in pixel coords in the *original* frame.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        frame_w, frame_h = frame_wh
        x1, y1, x2, y2 = roi_bbox
        roi_h, roi_w, _ = roi_bgr.shape

        if roi_w < 10 or roi_h < 10:
            return None

        # Resize ROI to 224x224 (typical for MP-hand-style models)
        resized = cv2.resize(roi_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        inp = rgb.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]  # 1x3x224x224

        try:
            out = self.lm_sess.run([self.lm_out_name], {self.lm_in_name: inp})[0]
        except Exception as e:
            print(f"[ONNX] Landmark inference failed: {e}")
            return None

        out = np.array(out).squeeze()

        # Expect 63 values = 21 * 3
        if out.ndim != 1:
            out = out.reshape(-1)
        if out.size < 63:
            return None

        out = out[:63]
        pts = out.reshape(21, 3)  # (21, 3) with x,y in [0,1] in *ROI space*

        # Reproject ROI-normalized [0,1] into full-frame [0,1]:
        #   full_x = (x1 + u * roi_w) / frame_w
        #   full_y = (y1 + v * roi_h) / frame_h
        u = pts[:, 0]
        v = pts[:, 1]
        z = pts[:, 2]

        full_x = (x1 + u * roi_w) / max(1, frame_w)
        full_y = (y1 + v * roi_h) / max(1, frame_h)

        full_pts = np.stack([full_x, full_y, z], axis=-1)

        # Dummy confidence: we could derive something from z or keypoint spread.
        score = 1.0
        return HandLandmarks(points=full_pts.astype(np.float32), score=score)

    # ------------------------------------------------------------------
    # CV-based detector (default)
    # ------------------------------------------------------------------
    def _detect_hand_cv(
        self,
        frame_bgr: np.ndarray,
        w: int,
        h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Very simple skin-color-based detector.

        This is not perfect, but it's:
          - Fast
          - Good enough to gate landmarks to a rough hand ROI
          - Works without any extra ONNX detector model

        Returns (x1, y1, x2, y2) in pixel coords, or None.
        """
        # Convert to HSV for crude skin segmentation
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Generic skin-tone range; tweak for your lighting/skin tone if needed
        lower = np.array([0, 40, 60], dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 0.005 * (w * h):
            # Too small; probably noise
            return None

        x, y, bw, bh = cv2.boundingRect(largest)

        # Pad box a bit so we don't clip fingers
        pad_x = int(0.2 * bw)
        pad_y = int(0.2 * bh)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        return (x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # ONNX-based detector (stub for future)
    # ------------------------------------------------------------------
    def _detect_hand_onnx(
        self,
        frame_bgr: np.ndarray,
        w: int,
        h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Stub for a real ONNX detector.

        This is intentionally minimal and will need adjustment once you pick
        a concrete model (YOLO, MediaPipe-export, etc.).

        Currently just returns None so we don't accidentally misinterpret
        arbitrary detector outputs.

        When you add a real detector:
          - Resize / normalize frame to detector's expected input
          - Run self.det_sess.run(...)
          - Parse the output tensor(s) into (x1,y1,x2,y2) in pixel coords
          - Return that box
        """
        # TODO: implement when a specific detector ONNX is chosen.
        # For now: no ONNX detection; fall back to CV-based or full-frame.
        return None

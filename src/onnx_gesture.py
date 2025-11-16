# src/onnx_gesture.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class HandLandmarks:
    """
    21 keypoints, each (x, y, z) in normalized image coordinates [0,1].
    """
    points: np.ndarray  # shape (21, 3)
    score: float        # dummy confidence (we don't have a detector yet)


class ONNXHandRuntime:
    """
    Minimal ONNX Runtime wrapper for a *hand landmark* model only.

    We deliberately ignore any detector model for now and simply:
      - Resize the full frame to 224x224
      - Run the landmark ONNX
      - Interpret the 63 outputs as 21 (x, y, z) points in [0,1]

    This is enough to get gesture classification working end-to-end
    without fighting with YOLO-style detector outputs.
    """

    def __init__(
        self,
        det_model_path: str = "models/hand_det.onnx",      # ignored for now
        lm_model_path: str = "models/hand_landmarks.onnx",
        device: str = "cpu",
    ):
        # Landmark model is required
        if not os.path.exists(lm_model_path):
            raise FileNotFoundError(f"Hand landmark model not found: {lm_model_path}")

        providers = ["CPUExecutionProvider"]
        if device.lower() == "cuda":
            providers.insert(0, "CUDAExecutionProvider")

        # We only use the landmark session
        self.lm_sess = ort.InferenceSession(lm_model_path, providers=providers)
        self.lm_in_name = self.lm_sess.get_inputs()[0].name
        self.lm_out_name = self.lm_sess.get_outputs()[0].name

        # Detector is currently unused; we just keep the path for future
        self.det_model_path = det_model_path

    # ---------- Public API ----------

    def detect_and_landmarks(self, frame_bgr) -> Optional[HandLandmarks]:
        """
        Run hand landmark model on the entire frame.

        Steps:
          1. Resize frame to 224x224
          2. Convert to RGB, CHW, float32, [0,1]
          3. Run ONNX landmark model
          4. Interpret output as 21 x (x,y,z) in [0,1]

        Returns:
          HandLandmarks(points=(21,3), score=1.0) or None on failure.
        """

        if frame_bgr is None or frame_bgr.size == 0:
            return None

        # Resize to 224x224 (most MP-hand-style ONNX models expect this)
        h, w, _ = frame_bgr.shape
        resized = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        inp = rgb.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]  # 1x3x224x224

        # ONNX inference
        try:
            out = self.lm_sess.run([self.lm_out_name], {self.lm_in_name: inp})[0]
        except Exception as e:
            print(f"[ONNX] Landmark inference failed: {e}")
            return None

        out = np.array(out).squeeze()

        # We expect 63 values = 21 * 3
        if out.ndim != 1:
            out = out.reshape(-1)
        if out.size < 63:
            # Not enough data
            return None

        # Take the first 63 values
        out = out[:63]
        pts = out.reshape(21, 3)  # (21, 3) with x,y in [0,1]

        # These x,y are normalized across the resized frame; since that is
        # just a scaled version of the original, [0,1] is still valid
        # normalized coordinates for the original image.
        score = 1.0  # we don't have a true detection confidence yet

        return HandLandmarks(points=pts, score=score)

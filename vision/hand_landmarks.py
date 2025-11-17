# vision/hand_landmarks.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class HandLandmarks:
    # 21 landmarks, each (x, y) in normalized [0, 1] within the crop
    points: np.ndarray  # shape: (21, 2)


class HandLandmarkModel:
    """
    Wrapper around your existing ONNX hand landmark model.
    Expects cropped hand images.
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (224, 224),
        providers: Optional[list[str]] = None,
    ) -> None:
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = input_size

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        target_w, target_h = self.input_size
        resized = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        img = np.expand_dims(img, axis=0)   # (1, C, H, W)
        return img

    def infer(self, crop_bgr: np.ndarray) -> Optional[HandLandmarks]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        img = self._preprocess(crop_bgr)
        outputs = self.session.run(self.output_names, {self.input_name: img})

        # Adapt this to your actual output format.
        # Assume: (1, 42) → 21 (x, y) pairs normalized [0,1]
        raw = outputs[0].reshape(-1, 2)  # (21, 2)
        return HandLandmarks(points=raw)

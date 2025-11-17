# vision/hand_detector.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class HandBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)


class HandDetector:
    """
    Wrapper for a hand detection ONNX model.

    Expected model behavior (you can adapt this to your actual model):
      - Input:  (1, 3, H, W) normalized float32
      - Output: bounding boxes + scores, e.g.
          boxes:  (N, 4) in xyxy normalized [0, 1]
          scores: (N,)
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (320, 320),
        score_threshold: float = 0.3,
        providers: Optional[list[str]] = None,
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.score_threshold = score_threshold

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, float]:
        h, w = frame_bgr.shape[:2]
        target_w, target_h = self.input_size

        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        img = np.expand_dims(img, axis=0)   # (1, C, H, W)

        scale_x = w / float(target_w)
        scale_y = h / float(target_h)

        return img, scale_x, scale_y

    def detect(self, frame_bgr: np.ndarray) -> List[HandBox]:
        """
        Run detection on a BGR frame and return a list of HandBox objects.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        img, scale_x, scale_y = self._preprocess(frame_bgr)

        inputs = {self.input_name: img}
        outputs = self.session.run(self.output_names, inputs)

        # You will likely need to adapt this depending on your model outputs.
        # For now we assume:
        #   boxes: (N, 4) with [x1, y1, x2, y2] normalized
        #   scores: (N,)
        boxes = outputs[0]
        scores = outputs[1]

        hand_boxes: List[HandBox] = []

        for box, score in zip(boxes, scores):
            if score < self.score_threshold:
                continue

            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            # basic sanity checks
            if x2 <= x1 or y2 <= y1:
                continue

            hand_boxes.append(HandBox(x1=x1, y1=y1, x2=x2, y2=y2, score=float(score)))

        return hand_boxes

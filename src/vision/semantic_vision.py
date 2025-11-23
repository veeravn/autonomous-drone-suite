from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False


# ---------- Data models ----------

@dataclass
class SubjectDetection:
    """
    A semantically-meaningful thing in the frame.

    All coordinates are normalized to [0,1] in image space.
    """
    label: str                         # "person", "car", "structure", "other"
    score: float                       # [0,1]
    bbox_xyxy: Tuple[float, float, float, float]  # (x1,y1,x2,y2) in [0,1]
    center: Tuple[float, float]        # (cx,cy) in [0,1]
    area: float                        # bbox area fraction [0,1]
    track_id: Optional[int] = None     # reserved for future tracking


@dataclass
class SemanticFrame:
    """
    Combined semantic view of a frame: embedding + detected subjects.
    """
    embedding: np.ndarray              # shape (D,)
    subjects: List[SubjectDetection]


# ---------- Interfaces ----------

class EmbeddingModel:
    """
    Interface for an image embedding model (CLIP-like).
    """

    def embed(self, frame_bgr: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class ObjectDetector:
    """
    Interface for object detectors used by semantic NBV.
    """

    def detect(self, frame_bgr: np.ndarray) -> List[SubjectDetection]:  # pragma: no cover - interface
        raise NotImplementedError


# ---------- Concrete implementations ----------

class MeanColorEmbedder(EmbeddingModel):
    """
    Tiny fallback embedder: uses mean color + a couple of simple stats.
    This keeps the pipeline working even without a real CLIP model.
    """

    def embed(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((8,), dtype=np.float32)

        h, w, _ = frame_bgr.shape
        if h == 0 or w == 0:
            return np.zeros((8,), dtype=np.float32)

        # downsample to reduce noise
        small = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        mean = small.mean(axis=(0, 1))          # B,G,R
        std = small.std(axis=(0, 1))            # B,G,R
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        brightness = gray.mean()

        vec = np.concatenate(
            [mean.astype(np.float32),
             std.astype(np.float32),
             np.array([contrast, brightness], dtype=np.float32)]
        )
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)


class CLIPOnnxEmbedder(EmbeddingModel):
    """
    ONNX-based CLIP embedder that works with sayantan47/clip-vit-b32-onnx
    `onnx/model_fp16.onnx`.

    - Handles multi-input (text + image) CLIP graphs.
    - Picks the 4D image input (usually 'pixel_values').
    - Picks an image-related output (prefers 'image_embeds', else 'logits_per_image').
    - Disables graph optimizations to avoid fusion bugs.
    - If anything fails, falls back to MeanColorEmbedder.
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (224, 224),
        device: str = "cpu",
    ) -> None:
        self._fallback = MeanColorEmbedder()
        self.session = None
        self.image_input_name: Optional[str] = None
        self.extra_feeds: dict[str, np.ndarray] = {}
        self.output_name: Optional[str] = None
        self.input_size = input_size

        if not _HAS_ORT:
            print("[SEMANTIC] onnxruntime not available; using MeanColorEmbedder.")
            return

        try:
            # IMPORTANT: turn OFF graph optimizations to avoid the error you saw.
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

            providers = ["CPUExecutionProvider"]
            if device.lower() == "cuda":
                providers.insert(0, "CUDAExecutionProvider")

            self.session = ort.InferenceSession(
                model_path,
                sess_options=so,
                providers=providers,
            )

            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()

            # ---- Pick image input ----
            image_input = None
            for inp in inputs:
                name = inp.name
                # Replace dynamic dims ('batch', 'channels', etc.) with 1 for shape logic
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                # Prefer 4D inputs (N,C,H,W) and names that sound like images
                if len(shape) == 4 and (image_input is None or "pixel" in name.lower()):
                    image_input = inp

            if image_input is None:
                print("[SEMANTIC] CLIP ONNX: no 4D/pixel input found; using fallback.")
                self.session = None
                return

            self.image_input_name = image_input.name

            # ---- Prepare zero feeds for non-image inputs (e.g. text) ----
            extra_feeds: dict[str, np.ndarray] = {}
            for inp in inputs:
                if inp.name == self.image_input_name:
                    continue

                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                onnx_type = getattr(inp, "type", "tensor(float)")  # e.g. "tensor(int64)"

                if "int64" in onnx_type:
                    dtype = np.int64
                else:
                    # default to float32 for everything else
                    dtype = np.float32

                extra_feeds[inp.name] = np.zeros(shape, dtype=dtype)

            self.extra_feeds = extra_feeds

            # ---- Pick an appropriate output ----
            out_name = None
            for out in outputs:
                n = out.name.lower()
                if "image_embeds" in n:
                    out_name = out.name
                    break
            if out_name is None:
                for out in outputs:
                    n = out.name.lower()
                    if "logits_per_image" in n:
                        out_name = out.name
                        break
            if out_name is None:
                out_name = outputs[0].name

            self.output_name = out_name

            print(f"[SEMANTIC] CLIPOnnxEmbedder loaded from {model_path}")
            print(f"[SEMANTIC]   image_input = {self.image_input_name}")
            print(f"[SEMANTIC]   output      = {self.output_name}")

        except Exception as e:
            print(f"[SEMANTIC] Failed to load CLIP ONNX model: {e}")
            self.session = None

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        target_w, target_h = self.input_size
        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0

        # CLIP-style normalization
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        img = (img - mean) / std

        img = img.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)
        return img

    def embed(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.session is None or frame_bgr is None or frame_bgr.size == 0:
            return self._fallback.embed(frame_bgr)

        try:
            inp_img = self._preprocess(frame_bgr)
            feeds = {self.image_input_name: inp_img}
            feeds.update(self.extra_feeds)

            out = self.session.run([self.output_name], feeds)[0]
            emb = np.array(out).reshape(-1).astype(np.float32)

            # L2 normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            return emb
        except Exception as e:
            print(f"[SEMANTIC] CLIP embedding failed: {e}")
            return self._fallback.embed(frame_bgr)

class NoOpDetector(ObjectDetector):
    """
    Fallback detector that returns no subjects.
    Useful when YOLO or other detectors aren't installed yet.
    """

    def detect(self, frame_bgr: np.ndarray) -> List[SubjectDetection]:
        return []


class YoloOnnxDetector(ObjectDetector):
    """
    YOLO-style ONNX detector. This is a skeleton; you'll need to adapt
    preprocessing + output parsing for your specific model.

    If initialization fails, this automatically falls back to NoOpDetector.
    """

    def __init__(
        self,
        model_path: str,
        class_map: dict[int, str],
        score_threshold: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.score_threshold = score_threshold
        self.class_map = class_map

        if not _HAS_ORT:
            print("[SEMANTIC] onnxruntime not available; using NoOpDetector.")
            self.session = None
            return

        try:
            providers = ["CPUExecutionProvider"]
            if device.lower() == "cuda":
                providers.insert(0, "CUDAExecutionProvider")

            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            print(f"[SEMANTIC] YoloOnnxDetector loaded from {model_path}")
        except Exception as e:
            print(f"[SEMANTIC] Failed to load YOLO ONNX model: {e}")
            self.session = None

    def detect(self, frame_bgr: np.ndarray) -> List[SubjectDetection]:
        if self.session is None or frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return []

        # NOTE: This is placeholder preprocessing. Replace with what
        # your YOLO model expects (e.g., letterbox resize, specific size).
        inp = cv2.resize(frame_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
        inp = inp.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]  # (1,3,640,640)

        try:
            outputs = self.session.run(self.output_names, {self.input_name: inp})
        except Exception as e:
            print(f"[SEMANTIC] YOLO inference failed: {e}")
            return []

        # Example assumption:
        #   detections: (N, 6) = [x1,y1,x2,y2,score,class_id] in *input* pixels (640x640)
        dets = np.array(outputs[0]).reshape(-1, 6)
        subjects: List[SubjectDetection] = []

        for x1, y1, x2, y2, score, cls_id in dets:
            score = float(score)
            if score < self.score_threshold:
                continue

            cls_id = int(cls_id)
            raw_label = self.class_map.get(cls_id, "other")

            if raw_label in ("person",):
                label = "person"
            elif raw_label in ("car", "truck", "bus"):
                label = "car"
            elif raw_label in ("building", "house"):
                label = "structure"
            else:
                label = "other"

            # Map from 640x640 back to original frame size
            x_scale = w / 640.0
            y_scale = h / 640.0
            x1p = float(x1) * x_scale
            y1p = float(y1) * y_scale
            x2p = float(x2) * x_scale
            y2p = float(y2) * y_scale

            # normalize to [0,1]
            x1_n = np.clip(x1p / max(1, w), 0.0, 1.0)
            y1_n = np.clip(y1p / max(1, h), 0.0, 1.0)
            x2_n = np.clip(x2p / max(1, w), 0.0, 1.0)
            y2_n = np.clip(y2p / max(1, h), 0.0, 1.0)

            cx = 0.5 * (x1_n + x2_n)
            cy = 0.5 * (y1_n + y2_n)
            area = max(0.0, (x2_n - x1_n) * (y2_n - y1_n))

            subjects.append(
                SubjectDetection(
                    label=label,
                    score=score,
                    bbox_xyxy=(x1_n, y1_n, x2_n, y2_n),
                    center=(cx, cy),
                    area=area,
                    track_id=None,
                )
            )

        return subjects


# ---------- Orchestrator ----------

class SemanticPerception:
    """
    High-level semantic vision wrapper that combines an embedding
    model + object detector into a single call.
    """

    def __init__(self, embedder: EmbeddingModel, detector: ObjectDetector) -> None:
        self.embedder = embedder
        self.detector = detector

    def analyze(self, frame_bgr: np.ndarray) -> SemanticFrame:
        if frame_bgr is None or frame_bgr.size == 0:
            return SemanticFrame(
                embedding=np.zeros((8,), dtype=np.float32),
                subjects=[],
            )

        emb = self.embedder.embed(frame_bgr)
        subjects = self.detector.detect(frame_bgr)
        return SemanticFrame(embedding=emb, subjects=subjects)

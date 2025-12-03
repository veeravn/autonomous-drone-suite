import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATHS = {
    "yolo": MODEL_DIR / "yolo_nano.onnx",
    "clip": MODEL_DIR / "clip_image.onnx",
    "hand_det": MODEL_DIR / "hand_det.onnx",
    "hand_landmarks": MODEL_DIR / "hand_landmarks.onnx",
    "gesture": MODEL_DIR / "gesture.onnx",  # if added later
}

def get(model_name: str):
    """Return model path."""
    path = MODEL_PATHS.get(model_name)
    if not path:
        raise ValueError(f"Unknown model name: {model_name}")
    return str(path)

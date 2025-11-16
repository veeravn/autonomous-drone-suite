# scripts/export_yolo_detector_onnx.py
from ultralytics import YOLO
from pathlib import Path

def main():
    # Tiny model; downloads automatically if not cached
    model = YOLO("yolov8n.pt")  # or "yolo11n.pt" if you prefer YOLO11

    # Export to ONNX (320x320 input)
    onnx_path = model.export(
        format="onnx",
        imgsz=320,
        opset=12,
        dynamic=False,
    )

    print(f"Exported ONNX model to: {onnx_path}")

    # Move/rename into models/hand_det.onnx
    dst = Path("models") / "hand_det.onnx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    Path(onnx_path).replace(dst)

    print(f"Moved to {dst.resolve()}")

if __name__ == "__main__":
    main()

# scripts/export_yolo_detector_onnx.py
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def main():
    repo_id = "Ultralytics/YOLO26"
    filename = "yolo26n.pt"   # change to yolo26s.pt or yolo26m.pt if you want a larger model

    # Download the YOLO26 detection weights from Hugging Face
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Load model from downloaded weights
    model = YOLO(weights_path)

    # Export to ONNX
    onnx_path = model.export(
        format="onnx",
        imgsz=320,
        opset=12,
        dynamic=False,
    )

    print(f"Exported ONNX model to: {onnx_path}")

    # Move/rename into models/yolo_nano.onnx
    dst = Path("models") / "yolo_nano.onnx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    Path(onnx_path).replace(dst)

    print(f"Moved to {dst.resolve()}")


if __name__ == "__main__":
    main()

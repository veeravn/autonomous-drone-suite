from __future__ import annotations
import cv2

class OakCapture:
    """
    Minimal VideoCapture-like wrapper for Luxonis OAK devices using DepthAI.
    Exposes: read() -> (ok, frame_bgr), release()
    """
    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0):
        try:
            import depthai as dai
        except Exception as e:
            raise RuntimeError(
                "DepthAI is not installed. Install with: pip install depthai"
            ) from e

        self._dai = dai
        self._pipeline = dai.Pipeline()

        cam = self._pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(width, height)
        cam.setInterleaved(False)
        cam.setFps(fps)

        xout = self._pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("rgb")
        cam.preview.link(xout.input)

        self._device = dai.Device(self._pipeline)
        self._q = self._device.getOutputQueue("rgb", maxSize=4, blocking=False)

    def read(self):
        msg = self._q.tryGet()
        if msg is None:
            return False, None
        frame = msg.getCvFrame()   # BGR frame usable by OpenCV  your pipeline
        return True, frame

    def release(self):
        try:
            self._device.close()
        except Exception:
            pass

def _open_opencv(index: int = 0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    return cap

def open_camera(
    index: int = 0,
    *,
    backend: str = "auto",
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
):
    """
    Open a camera capture device.

    backend:
      - "auto": try OAK (DepthAI) first, fall back to OpenCV
      - "oak": force Luxonis OAK via DepthAI
      - "opencv": force OpenCV VideoCapture
    """
    b = (backend or "auto").lower()

    if b == "opencv":
        return _open_opencv(index)

    if b in ("oak", "auto"):
        try:
            return OakCapture(width=width, height=height, fps=fps)
        except Exception:
            if b == "oak":
                raise
            # fallback for auto
            return _open_opencv(index)

    raise ValueError(f"Unknown camera backend: {backend!r}")
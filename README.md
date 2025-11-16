# Drone AI MVP – Planner + Gesture + Dedup


This repo contains a minimal prototype of an autonomous drone AI system with:
- **NBV Planner** using vision embeddings + telemetry constraints
- **Gesture → Flight** interface (context-aware mapping)
- **Shot Deduplication** using perceptual and spatiotemporal signals


## Quick Start


> ⚠️ **Python Version Note**: `mediapipe` (for gestures) currently targets Python ≤ 3.10. If you're on Python 3.11+ (e.g., 3.13), gesture module will auto-disable. Use the virtualenv in `setup.sh` to get Python 3.10 for full functionality.


### 1) Install (Linux/macOS)
```bash
./setup.sh
```

### 2) Run NBV + Dedupe + (optional) Gestures on webcam
source .venv/bin/activate
python src/main.py --use-gestures 1 --use-sitl 0 --camera 0

### 3) Run NBV unit smoke test
python tests/test_nbv.py

### 4) Visualize dedupe on sample images
python tests/visualize_dedupe.py --img-dir data/images --copy-unique data/unique

### PX4 SITL (Optional)
Use ```scripts/run_sitl.sh``` (or your own SITL launcher) and start ```main.py``` with ```--use-sitl 1```.

### Configuration
Copy .env.example → .env and update as needed.

### Modules

- planner_agent.py – computes embeddings (Torch/ResNet18) and suggests heading deltas.

- gesture_control.py – MediaPipe hand landmarks; gracefully disables if unavailable.

- shot_dedupe.py – pHash + GPS/orientation proximity; FAISS optional for scaling.

### Notes

- NBV scoring here is intentionally simple (novelty + cost). Replace with your decision graph later.

- Dedup fusion uses Hamming distance + 3D position proximity; extend with CLIP/FAISS as you grow.

### License

MIT (for prototype). If you plan to patent, keep the repo private until provisional filing.

```md
---


## requirements.txt
```txt
# Core
numpy
opencv-python
pillow
scipy
python-dotenv


# Vision / Embeddings
torch
torchvision


# Gesture (optional; Python <= 3.10 typically)
mediapipe


# Dedupe
ImageHash
faiss-cpu
```


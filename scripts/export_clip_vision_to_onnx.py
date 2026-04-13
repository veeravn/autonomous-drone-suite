import os
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
OUT_DIR = os.path.expanduser("~/autonomous-drone-suite/models")
OUT_PATH = os.path.join(OUT_DIR, "clip_image.onnx")

os.makedirs(OUT_DIR, exist_ok=True)

# Load the CLIP vision encoder and image processor
model = CLIPVisionModel.from_pretrained(MODEL_ID)
processor = CLIPImageProcessor.from_pretrained(MODEL_ID)

model.eval()

# Save processor info for later reference
processor.save_pretrained(os.path.join(OUT_DIR, "clip_processor"))

# CLIP ViT-B/32 standard image size is 224x224
dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

with torch.no_grad():
    torch.onnx.export(
        model,
        args=(dummy,),
        f=OUT_PATH,
        input_names=["pixel_values"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

print(f"Saved ONNX model to: {OUT_PATH}")
print("Saved processor config to:", os.path.join(OUT_DIR, "clip_processor"))

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch

from ultralytics import YOLO

# -------------------------
# Load YOLOv8 model
# -------------------------
yolo = YOLO("Model/RoadSeg/weights/best.pt")  # wrapper
model = yolo.model.eval()  # raw nn.Module


# -------------------------
# Preprocess image
# -------------------------
def resize_to_multiple_of_32(image):
    h, w = image.shape[:2]
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    return cv2.resize(image, (new_w, new_h))


img_path = "img.png"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = resize_to_multiple_of_32(img)

img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

# -------------------------
# Hook layers
# -------------------------
layer_outputs = {}


def get_activation(name):
    def hook(model, input, output):
        layer_outputs[name] = output.detach()

    return hook


# register hooks for Conv layers
for idx, layer in enumerate(model.model):
    if hasattr(layer, 'conv'):
        layer.register_forward_hook(get_activation(f"layer_{idx}"))

# -------------------------
# Forward pass
# -------------------------
with torch.no_grad():
    _ = model(img_tensor)

# -------------------------
# Save feature maps
# -------------------------
save_dir = Path("feature_maps")
save_dir.mkdir(exist_ok=True)
save_dir = Path("filters")
save_dir.mkdir(exist_ok=True)

# Go through layers
for idx, layer in enumerate(model.model):
    if hasattr(layer, "conv"):  # Conv layers
        weights = layer.conv.weight.detach().cpu().numpy()  # [out_channels, in_channels, kH, kW]
        out_ch, in_ch, kH, kW = weights.shape

        layer_dir = save_dir / f"layer_{idx}"
        layer_dir.mkdir(exist_ok=True)

        print(f"Layer {idx}: {out_ch} filters, size {kH}x{kW}")

        # Visualize each filter (just first input channel for clarity)
        for f in range(out_ch):
            filt = weights[f, 0, :, :]  # first input channel
            filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-6)  # normalize 0-1

            plt.imshow(filt, cmap="gray")
            plt.axis("off")
            plt.savefig(layer_dir / f"filter_{f}.png", bbox_inches="tight", pad_inches=0)
            plt.close()

'''
for name, fmap in layer_outputs.items():
    fmap = fmap.squeeze(0)  # remove batch dim
    num_channels = fmap.shape[0]

    # Create subfolder per layer
    layer_dir = save_dir / name
    layer_dir.mkdir(exist_ok=True)

    print(f"Saving {num_channels} channels from {name} -> {layer_dir}")

    for i in range(num_channels):
        plt.imshow(fmap[i], cmap="viridis")
        plt.axis("off")
        plt.savefig(layer_dir / f"ch_{i}.png", bbox_inches="tight", pad_inches=0)
        plt.close()
'''

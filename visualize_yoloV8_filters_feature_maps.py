"""
visualize_yolov8_filters_featuremaps.py

Requirements:
    pip install ultralytics matplotlib torchvision pillow

Usage:
    python visualize_yolov8_filters_featuremaps.py \
        --model path/to/your_yolov8_model.pt \
        --image path/to/sample.jpg
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image

from ultralytics import YOLO


# ---------------------------
# Utilities
# ---------------------------
def imshow_grid(tensor, title=None, nrow=8, cmap=None, savepath=None):
    """Show a tensor (N,C,H,W) as a grid."""
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=True, scale_each=True)
    npimg = grid.cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.figure(figsize=(12, 12))
    if cmap:
        plt.imshow(npimg, cmap=cmap)
    else:
        plt.imshow(npimg)
    plt.axis('off')
    if title:
        plt.title(title)
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
    plt.show()


def visualize_filters_from_conv(conv_weight, title="filters", max_filters=64, savepath=None):
    """
    Visualize convolution filters (out_channels, in_channels, kH, kW).
    """
    w = conv_weight.clone().detach().cpu()
    out_ch, in_ch, kH, kW = w.shape
    n_to_show = min(out_ch, max_filters)

    if in_ch == 3:  # RGB filters
        imgs = w[:n_to_show]
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-6)
        imshow_grid(imgs, title=title, nrow=int(math.sqrt(n_to_show)), savepath=savepath)
    else:  # average over channels
        imgs = w[:n_to_show].mean(dim=1, keepdim=True)
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-6)
        imshow_grid(imgs, title=title, nrow=int(math.sqrt(n_to_show)), cmap='gray', savepath=savepath)


def visualize_feature_maps(act_tensor, title="activations", max_maps=64, savepath=None):
    """
    Visualize feature maps from activations (1,C,H,W).
    """
    act_tensor = act_tensor[0]  # take first batch
    num = min(act_tensor.shape[0], max_maps)
    channels = []
    for i in range(num):
        ch = act_tensor[i]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
        channels.append(ch.unsqueeze(0))
    grid_tensor = torch.cat(channels, dim=0)
    imshow_grid(grid_tensor, title=title, nrow=int(math.sqrt(num)), cmap='gray', savepath=savepath)


# ---------------------------
# Hook class
# ---------------------------
class ActivationExtractor:
    def __init__(self):
        self.features = {}

    def get_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach().cpu()

        return hook


# ---------------------------
# Main
# ---------------------------
def main(args):
    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    model = YOLO(args.model)
    model.to(device)

    # Get PyTorch nn.Module backbone
    backbone = model.model

    # Find conv layers
    conv_layers = [(n, m) for n, m in backbone.named_modules() if isinstance(m, torch.nn.Conv2d)]
    print(f"Found {len(conv_layers)} Conv2d layers.")

    # Show first conv filters
    if conv_layers:
        first_name, first_conv = conv_layers[0]
        print(f"Visualizing filters from {first_name} -> {first_conv}")
        visualize_filters_from_conv(
            first_conv.weight,
            title=f"{first_name}_filters",
            max_filters=args.max_filters,
            savepath=os.path.join(args.out_dir, f"{first_name}_filters.png")
        )

    # Register forward hooks
    extractor = ActivationExtractor()
    target_layers = conv_layers[:args.hook_count]  # take first N convs
    for n, m in target_layers:
        print(f"Hooking {n}")
        m.register_forward_hook(extractor.get_hook(n))

    # Load and preprocess image
    # Load and preprocess image (manual instead of torchvision ToTensor)
    img = Image.open(args.image).convert('RGB')
    img_resized = img.resize((args.img_size, args.img_size))

    # Force conversion to numpy array
    # Force conversion to numpy array
    img_np = np.asarray(img_resized, dtype=np.float32) / 255.0  # (H,W,3)
    img_np = np.transpose(img_np, (2, 0, 1))  # (3,H,W)

    # Add this line to ensure the array is contiguous in memory
    img_np = np.ascontiguousarray(img_np)

    # Ensure it's a numpy.ndarray
    assert isinstance(img_np, np.ndarray), f"Expected numpy.ndarray, got {type(img_np)}"
    print(type(img_np))

    # Convert to tensor (the .copy() is now redundant but harmless)
    input_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        backbone(input_tensor)

    # Visualize activations
    os.makedirs(args.out_dir, exist_ok=True)
    for layer_name, feat in extractor.features.items():
        print(f"Visualizing activations from {layer_name}, shape {feat.shape}")
        savep = os.path.join(args.out_dir, f"{layer_name}_activations.png")
        visualize_feature_maps(feat, title=f"{layer_name}_activations", max_maps=args.max_maps, savepath=savep)

    print(f"Done. Results saved in {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Model/RoadSeg/weights/best.pt", help="Path to YOLOv8 .pt model")
    parser.add_argument("--image", type=str, default="img.png", help="Path to input image")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--out_dir", type=str, default="viz_output")
    parser.add_argument("--max_filters", type=int, default=64 * 64 * 5)
    parser.add_argument("--max_maps", type=int, default=64)
    parser.add_argument("--hook_count", type=int, default=3)
    parser.add_argument("--force_cpu", action='store_true')
    args = parser.parse_args()
    main(args)

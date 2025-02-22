import os
import time

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO

# Paths
image_dir = r"D:\downloadFiles\front_3\TestingVideo\TempImg"
mask_dir = r"D:\downloadFiles\front_3\TestingVideo\TempMasks"
output_file = r"F:\RunningProjects\YOLO_Model\val\Results\evaluation_results3.txt"

# Image resolutions to evaluate
image_sizes = [
    (i, j)
    for i in range(256, 1500, 128)  # Width between 128 and 2000, stepping by 128
    for j in range(256, 1500, 128)  # Height between 128 and 2000, stepping by 128
]

# Load YOLOv8 model
model = YOLO(r'F:\RunningProjects\YOLO_Model\Training\runs\segment\RoadSegmentationForMyDataset9\weights\best.pt')

# Initial system usage
cpu_start = psutil.cpu_percent(interval=1)
ram_start = psutil.virtual_memory().percent
gpu_available = torch.cuda.is_available()
gpu_start = torch.cuda.memory_allocated() / (1024 ** 2) if gpu_available else 0

# Open the output file
with open(output_file, "w") as file:
    file.write(f"Initial System Usage:\n")
    file.write(f"CPU Usage: {cpu_start:.2f}%\n")
    file.write(f"RAM Usage: {ram_start:.2f}%\n")
    file.write(f"GPU Usage: {gpu_start:.2f} MB\n\n")

    # Evaluate for each resolution
    for width, height in image_sizes:
        file.write(f"Evaluating for resolution: {width}x{height}\n")

        start_time = time.time()
        total_iou, total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0, 0
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            mask_path = os.path.join(mask_dir, image_file)

            # Load and resize images
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)
            if image is None or mask is None:
                continue

            image_resized = cv2.resize(image, (width, height))
            mask_resized = cv2.resize(mask, (width, height))

            # Ground truth mask to binary
            ground_truth_binary = np.all(mask_resized == [0, 0, 255], axis=-1).astype(np.uint8)

            # Model inference
            results = model(image_resized, imgsz=(width, height), task='segment', conf=0.8)
            if results[0].masks is not None and results[0].masks.data is not None:
                predicted_mask = results[0].masks.data[0].cpu().numpy()
                predicted_mask = cv2.resize(predicted_mask, (width, height))
                predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

                # Calculate metrics
                predicted_mask_area = np.sum(predicted_mask)  # Area of Predicted Region
                ground_truth_area = np.sum(ground_truth_binary)  # Area of Ground Truth Region

                intersection = np.sum((predicted_mask == 1) & (ground_truth_binary == 1))  # Area of Overlap
                union = predicted_mask_area + ground_truth_area - intersection  # Area of Union

                iou = intersection / union if union > 0 else 0  # IoU Calculation
                total_iou += iou

                # Update confusion metrics
                tp = intersection  # True Positives
                tn = np.sum((predicted_mask == 0) & (ground_truth_binary == 0))  # True Negatives
                fp = np.sum((predicted_mask == 1) & (ground_truth_binary == 0))  # False Positives
                fn = np.sum((predicted_mask == 0) & (ground_truth_binary == 1))  # False Negatives

                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn

        # Final metrics
        num_images = len(image_files)
        mean_iou = total_iou / num_images if num_images > 0 else 0
        end_time = time.time()
        total_time = end_time - start_time

        file.write(f"Mean IoU: {mean_iou:.4f}\n")
        file.write(f"Total True Positives: {total_tp}\n")
        file.write(f"Total True Negatives: {total_tn}\n")
        file.write(f"Total False Positives: {total_fp}\n")
        file.write(f"Total False Negatives: {total_fn}\n")
        file.write(f"Total Time: {total_time:.2f} seconds\n")
        file.write(f"Images Processed: {num_images}\n\n")

    # Final system usage
    cpu_end = psutil.cpu_percent(interval=1)
    ram_end = psutil.virtual_memory().percent
    gpu_end = torch.cuda.memory_allocated() / (1024 ** 2) if gpu_available else 0

    file.write(f"Average Final System Usage:\n")
    file.write(f"CPU Usage: {cpu_end:.2f}%\n")
    file.write(f"RAM Usage: {ram_end:.2f}%\n")
    file.write(f"GPU Usage: {gpu_end:.2f} MB\n")

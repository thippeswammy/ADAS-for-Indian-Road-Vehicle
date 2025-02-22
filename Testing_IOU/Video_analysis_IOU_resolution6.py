import os
import subprocess
import time

import cv2
import numpy as np
import pandas as pd
import psutil
from ultralytics import YOLO


def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, text=True)
    memory_info = result.stdout.strip().split('\n')
    for info in memory_info:
        used, total = map(int, info.split(', '))
        return 100 * (used / total)


def get_system_usage():
    """Returns the current system usage metrics."""
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    # gpu_usage = get_gpu_memory()
    return cpu_usage, ram_usage, 0


# Paths
original_image_dir = r"D:\\downloadFiles\\front_3\\TestingVideo\\TempImg"
mask_image_dir = r"D:\\downloadFiles\\front_3\\TestingVideo\\TempMasks"
output_file = r"/Testing_IOU\\Results\\evaluation_results6.txt"
excel_file = r"/Testing_IOU\\Results\\evaluation_results6.xlsx"
output_dir = r"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImages"
os.makedirs(output_dir, exist_ok=True)

image_sizes = [
    # 4:3 Aspect Ratio
    (640, 480),
    (800, 600), (1024, 768), (1280, 960), (1600, 1200),
    # # 16:9 Aspect Ratio
    # (1280, 720), (1366, 768), (1920, 1080), (2560, 1440),
    # # 16:10 Aspect Ratio
    # (1280, 800), (1440, 900), (1680, 1050), (1920, 1200), (2560, 1600),
    # # 1:1 Aspect Ratio
    # (500, 500), (800, 800), (1080, 1080), (2000, 2000),
    # # 21:9 Aspect Ratio (Ultra-Wide)
    # (2560, 1080),
    # # 3:2 Aspect Ratio
    # (720, 480), (1080, 720), (1620, 1080), (2160, 1440),
    # # 5:4 Aspect Ratio
    # (1280, 1024),
    # # Other Popular Resolutions
    # (800, 480), (854, 480), (1152, 768), (2048, 1080)
]

# Store results in a DataFrame
results_df = []

# Evaluate for each resolution
for width, height in image_sizes:
    resolution_results = []
    # Initial system usage
    cpu_start, ram_start, gpu_start = get_system_usage()
    start_time = time.time()

    # Load YOLOv8 model
    model = YOLO(
        r'F:\\RunningProjects\\YOLO_Model\\Training\\runs\\segment\\RoadSegmentationForMyDataset9\\weights\\best.pt').cuda()

    resolution_results.append({"Resolution": f"{width}x{height}", "Metric": "Initial System Usage",
                               "CPU (%)": cpu_start, "RAM (%)": ram_start, "GPU (%)": gpu_start})

    total_iou, total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0, 0
    original_image_files = sorted([f for f in os.listdir(original_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_image_files = sorted([f for f in os.listdir(mask_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    system_parameters = []

    for image_file, mask_file in zip(original_image_files, mask_image_files):
        image_path = os.path.join(original_image_dir, image_file)
        mask_path = os.path.join(mask_image_dir, mask_file)

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
        inference_results = model(image_resized, imgsz=(width, height), task='segment', conf=0.8)
        cpu, ram, gpu = get_system_usage()
        system_parameters.append([cpu - cpu_start, ram - ram_start, gpu - gpu_start])

        if inference_results[0].masks is not None and inference_results[0].masks.data is not None:
            predicted_mask = inference_results[0].masks.data[0].cpu().numpy()
            predicted_mask = cv2.resize(predicted_mask, (width, height))
            predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

            # Create overlay image
            overlay = image_resized.copy()
            # Calculate metrics
            intersection = np.sum((predicted_mask == 1) & (ground_truth_binary == 1))  # Area of Overlap
            union = np.sum(predicted_mask) + np.sum(ground_truth_binary) - intersection  # Area of Union

            iou = intersection / union if union > 0 else 0  # IoU Calculation
            total_iou += iou

            # Save overlay image
            overlay[(predicted_mask == 1) & (ground_truth_binary == 1)] = [0, 255, 0]  # Green for true positives
            overlay[(predicted_mask == 0) &
                    (ground_truth_binary == 1)] = [0, 0, 255]  # Red for false negatives (missed detections)
            overlay[(predicted_mask == 1) &
                    (ground_truth_binary == 0)] = [255, 0, 0]  # Blue for false positives (incorrect predictions)

            output_path = os.path.join(output_dir, f"overlay_{image_file}")
            cv2.imwrite(output_path, overlay)

            # Update confusion metrics
            tp = intersection
            tn = np.sum((predicted_mask == 0) & (ground_truth_binary == 0))
            fp = np.sum((predicted_mask == 1) & (ground_truth_binary == 0))
            fn = np.sum((predicted_mask == 0) & (ground_truth_binary == 1))

            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

    # Average system parameters
    avg_cpu = np.mean([p[0] for p in system_parameters]) if system_parameters else 0
    avg_ram = np.mean([p[1] for p in system_parameters]) if system_parameters else 0
    avg_gpu = np.mean([p[2] for p in system_parameters]) if system_parameters else 0

    # Final metrics
    num_images = len(original_image_files)
    mean_iou = total_iou / num_images if num_images > 0 else 0
    end_time = time.time()
    total_time = end_time - start_time

    # Append final results
    results_df.append({
        "Resolution": f"{width}x{height}",
        "Mean IoU": round(mean_iou, 4),
        "Total TP": total_tp,
        "Total TN": total_tn,
        "Total FP": total_fp,
        "Total FN": total_fn,
        "Avg CPU (%)": round(avg_cpu, 2),
        "Avg RAM (%)": round(avg_ram, 2),
        "Avg GPU (%)": round(avg_gpu, 2),
        "Total Time (s)": round(total_time, 2),
        "Images Processed": num_images
    })

print(results_df)
# Save results to the output file
results_df = pd.DataFrame(results_df)
results_df.to_csv(output_file, index=False)
results_df.to_excel(excel_file, index=False)

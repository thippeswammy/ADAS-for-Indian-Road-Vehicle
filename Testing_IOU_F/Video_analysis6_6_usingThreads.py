import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from ultralytics import YOLO


def create_unique_folder(base_dir):
    """
    Create a unique folder. If the folder already exists, append a number to make it unique.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return base_dir
    else:
        counter = 1
        while True:
            new_dir = f"{base_dir}_{counter}"
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                return new_dir
            counter += 1


def get_gpu_memory():
    """Gets the current GPU memory usage."""
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
    gpu_usage = get_gpu_memory()
    return cpu_usage, ram_usage, gpu_usage


def process_frame(frame_idx, original_frame, mask_frame, width, height, model, output_dir_res, selected_indices,
                  system_parameters):
    original_resized = cv2.resize(original_frame, (width, height))
    mask_resized = cv2.resize(mask_frame, (width, height), interpolation=cv2.INTER_NEAREST)

    ground_truth = cv2.cvtColor(mask_resized, cv2.COLOR_BGRA2GRAY)
    ground_truth = (ground_truth > 10).astype(np.uint8)

    # Model inference
    inference_results = model(original_resized, imgsz=(width, height), task='segment', conf=0.8)

    system_parameters.append(get_system_usage())

    if inference_results[0].masks is not None and inference_results[0].masks.data is not None:
        predicted_mask = inference_results[0].masks.data[0].cpu().numpy()
    else:
        predicted_mask = np.zeros((width, height), dtype=np.uint8)

    predicted_mask = cv2.resize(predicted_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

    intersection = np.sum((predicted_mask == 1) & (ground_truth == 1))
    union = np.sum(predicted_mask) + np.sum(ground_truth) - intersection
    iou = intersection / union if union > 0 else 0

    if frame_idx in selected_indices:
        overlay = original_resized.copy()
        overlay[(predicted_mask == 1) & (ground_truth == 1)] = [0, 255, 0]  # Green for true positives
        overlay[(predicted_mask == 0) & (ground_truth == 1)] = [0, 0, 255]  # Red for false negatives
        overlay[(predicted_mask == 1) & (ground_truth == 0)] = [255, 0, 0]  # Blue for false positives

        legend_colors = {
            "True Positive": (0, 255, 0),  # Green
            "False Negative": (0, 0, 255),  # Red
            "False Positive": (255, 0, 0),  # Blue
            "IOU": (255, 255, 255),  # White
        }
        legend_start_x = 10
        legend_start_y = 10
        legend_box_height = 20
        legend_spacing = 5
        for i, (label, color) in enumerate(legend_colors.items()):
            y_position = legend_start_y + i * (legend_box_height + legend_spacing)
            if label == "IOU":
                label = f"IOU={round(iou, 4)}"

            # Draw the color box
            cv2.rectangle(overlay,
                          (legend_start_x, y_position),
                          (legend_start_x + legend_box_height, y_position + legend_box_height),
                          color, -1)

            cv2.putText(overlay,
                        label,
                        (legend_start_x + legend_box_height + 10, y_position + legend_box_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA)
        # Save overlay image
        output_path = os.path.join(output_dir_res, f"overlay_frame_{frame_idx}.png")
        cv2.imwrite(output_path, overlay)

    tp = intersection
    tn = np.sum((predicted_mask == 0) & (ground_truth == 0))
    fp = np.sum((predicted_mask == 1) & (ground_truth == 0))
    fn = np.sum((predicted_mask == 0) & (ground_truth == 1))

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "iou": iou
    }


original_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\OriginalVideo.mp4"
mask_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\MaskVideo.mp4"
output_dir = r"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\PredictedImages"

output_dir = create_unique_folder(output_dir)
os.makedirs(output_dir, exist_ok=True)
excel_file = output_dir + "\\by_results6_6.xlsx"

resolution_results = []
image_sizes = [
    (640, 480),
    (500, 500), (800, 800), (1080, 1080), (1152, 768),
    (800, 600), (1024, 768), (1280, 960), (1600, 1200),
    (1280, 720), (1366, 768), (1920, 1080), (2560, 1440),
    (1280, 800), (1440, 900), (1680, 1050), (1920, 1200),
    (2560, 1080), (720, 480), (1080, 720), (1620, 1080),
    (2160, 1440), (1280, 1024), (800, 480), (854, 480),
]

# Load YOLOv8 model
model = YOLO(
    r'F:\\RunningProjects\\YOLO_Model\\Training\\runs\\segment\\RoadSegmentationForMyDataset9\\weights\\best.pt').cuda()

# Video capture objects
original_cap = cv2.VideoCapture(original_video_path)
mask_cap = cv2.VideoCapture(mask_video_path)

frame_count = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
selected_indices = random.sample(range(frame_count), max(1, max(1, (frame_count // 100) * 5)))
selected_indices = set(selected_indices)

results_df = []

# Evaluate for each resolution
for width, height in image_sizes:
    output_dir_res = os.path.join(output_dir, f"{width}x{height}")
    os.makedirs(output_dir_res, exist_ok=True)

    # Initial system usage
    cpu_start, ram_start, gpu_start = get_system_usage()
    start_time = time.time()

    system_parameters = []
    total_tp, total_tn, total_fp, total_fn, total_iou = 0, 0, 0, 0, 0
    original_cap = cv2.VideoCapture(original_video_path)
    mask_cap = cv2.VideoCapture(mask_video_path)

    with ThreadPoolExecutor() as executor:
        futures = []

        frame_idx = 0
        while True:
            ret_original, original_frame = original_cap.read()
            ret_mask, mask_frame = mask_cap.read()

            if not ret_original or not ret_mask:
                break

            futures.append(executor.submit(
                process_frame, frame_idx, original_frame, mask_frame, width, height, model, output_dir_res,
                selected_indices, system_parameters))

            frame_idx += 1

        for future in futures:
            result = future.result()
            total_tp += result["tp"]
            total_tn += result["tn"]
            total_fp += result["fp"]
            total_fn += result["fn"]
            total_iou += result["iou"]

    # Calculate final metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    avg_cpu = np.mean([p[0] for p in system_parameters]) if system_parameters else 0
    avg_ram = np.mean([p[1] for p in system_parameters]) if system_parameters else 0
    avg_gpu = np.mean([p[2] for p in system_parameters]) if system_parameters else 0

    mean_iou = total_iou / frame_idx if frame_idx > 0 else 0
    end_time = time.time()
    total_time = end_time - start_time

    results_df.append({
        "Resolution": f"{width}x{height}",
        "Mean IoU": round(mean_iou, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1_score, 4),
        "Total TP": total_tp,
        "Total TN": total_tn,
        "Total FP": total_fp,
        "Total FN": total_fn,
        "Avg CPU (%)": round(avg_cpu, 2),
        "Avg RAM (%)": round(avg_ram, 2),
        "Avg GPU (%)": round(avg_gpu, 2),
        "Total Time (s)": round(total_time, 2),
        "Images Processed": frame_idx
    })

    # Save system usage plot
    plt.figure(figsize=(12, 6))
    cpu_usages = [p[0] for p in system_parameters]
    ram_usages = [p[1] for p in system_parameters]
    gpu_usages = [p[2] for p in system_parameters]
    plt.plot(cpu_usages, label='CPU Usage (%)', marker='o')
    plt.plot(ram_usages, label='RAM Usage (%)', marker='s')
    plt.plot(gpu_usages, label='GPU Usage (%)', marker='^')
    plt.title(f'System Usage for Resolution {width}x{height}')
    plt.xlabel('Image Index')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'system_usage_{width}x{height}.png'), format='png')
    plt.close()

# Save results to files
results_df = pd.DataFrame(results_df)
results_df.to_excel(excel_file, index=False)

# Release video resources
original_cap.release()
mask_cap.release()

import os
import random
import subprocess
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from ultralytics import YOLO


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


# Paths
# original_image_dir = r"D:\\downloadFiles\\front_3\\TestingVideo\\TempImg"
# mask_image_dir = r"D:\\downloadFiles\\front_3\\TestingVideo\\TempMasks"
original_image_dir = r"D:\downloadFiles\road_archive (3)\data_dataset_voc\JPEGImages"
mask_image_dir = r"D:\downloadFiles\road_archive (3)\data_dataset_voc\Converted_Masks"
output_file = r"D:\downloadFiles\road_archive (3)\data_dataset_voc\archive3.txt"
excel_file = r"D:\downloadFiles\road_archive (3)\data_dataset_voc\archive3.xlsx"
output_dir = r"D:\downloadFiles\road_archive (3)\data_dataset_voc\PredictedImages"

resolution_results = []
image_sizes = [
    (640, 480),
    # (480, 640),
    # (800, 600), (1024, 768), (1280, 960), (1600, 1200), (2048, 1536),
    # (1280, 720), (1366, 768), (1920, 1080), (2560, 1440),
    # (1280, 800), (1440, 900), (1680, 1050), (1920, 1200),
    # (500, 500), (800, 800), (1080, 1080),
    # (2560, 1080),
    # (720, 480), (1080, 720), (1620, 1080), (2160, 1440),
    # (1280, 1024),
    # (800, 480), (854, 480), (1152, 768),
    # (2048, 1080)
]

# Load YOLOv8 model
model = YOLO(
    r'F:\\RunningProjects\\YOLO_Model\\Training\\runs\\segment\\RoadSegmentationForMyDataset9\\weights\\best.pt').cuda()
MASK_COLOR = [255, 255, 255]
original_image_files = sorted([f for f in os.listdir(original_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
mask_image_files = sorted([f for f in os.listdir(mask_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
# original_image_files = original_image_files[:5]
# mask_image_files = mask_image_files[:5]
# Randomly select 1% of images
total_images = len(original_image_files)
selected_indices = random.sample(range(total_images), max(1, total_images // 1))
selected_indices = set(selected_indices)

list_ = []
results_df = []

# Evaluate for each resolution
for width, height in image_sizes:
    output_dir_res = os.path.join(output_dir, f"{width}x{height}")
    os.makedirs(output_dir_res, exist_ok=True)

    # Initial system usage
    cpu_start, ram_start, gpu_start = get_system_usage()
    start_time = time.time()

    resolution_results.append({"Resolution": f"{width}x{height}", "Metric": "Initial System Usage",
                               "CPU (%)": cpu_start, "RAM (%)": ram_start, "GPU (%)": gpu_start})

    total_iou, total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0, 0

    system_parameters = []
    num_images_processed = 0

    for idx, (image_file, mask_file) in enumerate(zip(original_image_files, mask_image_files)):
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
        ground_truth_binary = np.all(mask_resized == MASK_COLOR, axis=-1).astype(np.uint8)

        # Model inference
        inference_results = model(image_resized, imgsz=(width, height), task='segment', conf=0.8)

        system_parameters.append(get_system_usage())

        if inference_results[0].masks is not None and inference_results[0].masks.data is not None:
            predicted_mask = inference_results[0].masks.data[0].cpu().numpy()
            if not list_.__contains__(predicted_mask.shape):
                list_.append(predicted_mask.shape)
        else:
            predicted_mask = np.zeros((width, height), dtype=np.uint8)
        predicted_mask = cv2.resize(predicted_mask, (width, height))
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

        # Save the binary mask (0 and 255) as an image
        # predicted_mask_255 = (predicted_mask * 255).astype(np.uint8)
        # mask_save_path = os.path.join(output_dir_res, f"predicted_mask_{image_file}")
        # cv2.imwrite(mask_save_path, predicted_mask_255)

        # Save overlay images for the selected 1%
        if idx in selected_indices:
            overlay = image_resized.copy()
            overlay[(predicted_mask == 1) & (ground_truth_binary == 1)] = [0, 255, 0]  # Green for true positives
            overlay[(predicted_mask == 0) & (ground_truth_binary == 1)] = [0, 0, 255]  # Red for false negatives
            overlay[(predicted_mask == 1) & (ground_truth_binary == 0)] = [255, 0, 0]  # Blue for false positives

            legend_colors = {
                "True Positive": (0, 255, 0),  # Green
                "False Negative": (0, 0, 255),  # Red
                "False Positive": (255, 0, 0),  # Blue
            }
            legend_start_x = 10
            legend_start_y = 10
            legend_box_height = 20
            legend_spacing = 5
            for i, (label, color) in enumerate(legend_colors.items()):
                y_position = legend_start_y + i * (legend_box_height + legend_spacing)
                # Draw the color box
                cv2.rectangle(overlay,
                              (legend_start_x, y_position),
                              (legend_start_x + legend_box_height, y_position + legend_box_height),
                              color, -1)
                # Add text next to the color box
                cv2.putText(overlay,
                            label,
                            (legend_start_x + legend_box_height + 10, y_position + legend_box_height - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA)
            # Save overlay image
            output_path = os.path.join(output_dir_res, f"overlay_{image_file}")
            cv2.imwrite(output_path, overlay)

        num_images_processed += 1
        # Calculate metrics
        intersection = np.sum((predicted_mask == 1) & (ground_truth_binary == 1))
        union = np.sum(predicted_mask) + np.sum(ground_truth_binary) - intersection
        iou = intersection / union if union > 0 else 0
        total_iou += iou

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
    mean_iou = total_iou / num_images_processed if num_images_processed > 0 else 0
    end_time = time.time()
    total_time = end_time - start_time

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
        "Images Processed": num_images_processed
    })

    # Plot system metrics
    cpu_usages = [p[0] for p in system_parameters]
    ram_usages = [p[1] for p in system_parameters]
    gpu_usages = [p[2] for p in system_parameters]

    plt.figure(figsize=(12, 6))
    plt.plot(cpu_usages, label='CPU Usage (%)', marker='o')
    plt.plot(ram_usages, label='RAM Usage (%)', marker='s')
    plt.plot(gpu_usages, label='GPU Usage (%)', marker='^')
    plt.title(f'System Usage for Resolution {width}x{height}')
    plt.xlabel('Image Index')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'system_usage_{width}x{height}.png'), format='png')  # Save the plot
    # plt.show()
    # plt.close()
# Save results to files
results_df = pd.DataFrame(results_df)
results_df.to_csv(output_file, index=True)
results_df.to_excel(excel_file, index=False)

print("\n\n\n\n", list(set(list_)))

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# GPU memory usage function
def get_gpu_memory():
    """Gets the current GPU memory usage."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, text=True)
    memory_info = result.stdout.strip().split('\n')
    used, total = map(int, memory_info[0].split(', '))
    return 100 * (used / total)


# Paths
original_image_dir = r"D:\downloadFiles\road_archive (3)\data_dataset_voc\JPEGImages"
mask_image_dir = r"D:\downloadFiles\road_archive (3)\data_dataset_voc\Converted_Masks"
output_file = r"D:\downloadFiles\road_archive (3)\data_dataset_voc\archive3.csv"
output_dir = r"D:\downloadFiles\road_archive (3)\data_dataset_voc\PredictedImages"

image_sizes = [(640, 480)]  # Test multiple resolutions if needed

# Load YOLOv8 model
model = YOLO(
    r'F:\\RunningProjects\\YOLO_Model\\Training\\runs\\segment\\RoadSegmentationForMyDataset9\\weights\\best.pt').cuda()

# Load files
original_image_files = sorted([f for f in os.listdir(original_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
mask_image_files = sorted([f for f in os.listdir(mask_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
assert len(original_image_files) == len(mask_image_files), "Mismatch in image and mask counts!"

# Randomly select 1% of images
total_images = len(original_image_files)
selected_indices = set(random.sample(range(total_images), max(1, total_images // 100)))

results_df = []


# Function to process a single image
def process_image(image_file, mask_file, width, height):
    image_path = os.path.join(original_image_dir, image_file)
    mask_path = os.path.join(mask_image_dir, mask_file)
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    if image is None or mask is None:
        return None

    # Resize images
    image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    ground_truth_binary = np.all(mask_resized == [255, 255, 255], axis=-1).astype(np.uint8)

    # YOLO inference
    results = model.predict(image_resized, imgsz=(width, height), task="segment", conf=0.8)
    predicted_mask = results[0].masks.data[0].cpu().numpy() if results[0].masks is not None else np.zeros(
        (height, width), dtype=np.uint8)

    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

    # Calculate metrics
    intersection = np.sum((predicted_mask == 1) & (ground_truth_binary == 1))
    union = np.sum(predicted_mask) + np.sum(ground_truth_binary) - intersection
    iou = intersection / union if union > 0 else 0

    return {"IoU": iou, "True Positives": intersection, "Total Pixels": np.size(predicted_mask)}


# Process all images
for width, height in image_sizes:
    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(lambda idx: process_image(original_image_files[idx], mask_image_files[idx], width, height),
                         range(total_images)))

    valid_results = [res for res in results if res is not None]
    mean_iou = np.mean([res["IoU"] for res in valid_results])
    total_tp = sum(res["True Positives"] for res in valid_results)
    total_pixels = sum(res["Total Pixels"] for res in valid_results)

    results_df.append({"Resolution": f"{width}x{height}", "Mean IoU": round(mean_iou, 4), "Total TP": total_tp,
                       "Total Pixels": total_pixels})

    end_time = time.time()
    print(
        f"Processed {len(valid_results)} images at resolution {width}x{height} in {round(end_time - start_time, 2)} seconds.")

# Save results
results_df = pd.DataFrame(results_df)
results_df.to_csv(output_file, index=False)

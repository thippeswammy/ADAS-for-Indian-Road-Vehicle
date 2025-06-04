import os
import random
import time

import cv2
import numpy as np
import pandas as pd
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


# Paths
original_image_dir = r"D:\\downloadFiles\\front_3\\TestingVideo\\TempImg"
mask_image_dir = r"D:\\downloadFiles\\front_3\\TestingVideo\\TempMasks"
output_dir = r"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\PredictedImages"
output_dir = create_unique_folder(output_dir)
output_file = f"{output_dir}/evaluation_results.txt"
excel_file = f"{output_dir}/evaluation_results.xlsx"

resolution_results = []
image_sizes = [
    (640, 480),
    (480, 640), (854, 640),
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
model = YOLO(f'../Model/RoadSeg/weights/best.pt').cuda()

original_image_files = sorted([f for f in os.listdir(original_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
mask_image_files = sorted([f for f in os.listdir(mask_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Randomly select 10% of images
total_images = len(original_image_files)
selected_indices = random.sample(range(total_images), max(1, total_images // 10))
selected_indices = set(selected_indices)

list_ = []
results_df = []
yolo_model_img_reso = []
# Evaluate for each resolution
for width, height in image_sizes:
    output_dir_res = os.path.join(output_dir, f"{width}x{height}")
    os.makedirs(output_dir_res, exist_ok=True)

    # Initial system usage
    start_time = time.time()
    resolution_results.append({"Resolution": f"{width}x{height}"})

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
        ground_truth_binary = np.all(mask_resized == [0, 0, 255], axis=-1).astype(np.uint8)

        # Model inference
        inference_results = model(image_resized, imgsz=(width, height), task='segment', conf=0.8)

        if inference_results[0].masks is not None and inference_results[0].masks.data is not None:
            predicted_mask = inference_results[0].masks.data[0].cpu().numpy()
            if not yolo_model_img_reso.__contains__(predicted_mask.shape):
                yolo_model_img_reso.append(predicted_mask.shape)
        else:
            predicted_mask = np.zeros((width, height), dtype=np.uint8)
        predicted_mask = cv2.resize(predicted_mask, (width, height))
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

        # Save overlay images for the selected 10%
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
    total_tp = total_tp / (idx + 1)
    total_tn = total_tn / (idx + 1)
    total_fp = total_fp / (idx + 1)
    total_fn = total_fn / (idx + 1)
    # Calculate final metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    mean_iou = total_iou / num_images_processed if num_images_processed > 0 else 0
    end_time = time.time()
    total_time = end_time - start_time

    results_df.append({
        "Resolution": f"{width}x{height} => {yolo_model_img_reso[-1][0]}x{yolo_model_img_reso[-1][1]}",
        "Mean IoU": round(mean_iou, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1_score, 4),
        "Total TP": total_tp,
        "Total TN": total_tn,
        "Total FP": total_fp,
        "Total FN": total_fn,
        "Total Time (s)": round(total_time, 2),
        "Images Processed": num_images_processed
    })

# Save results to files
results_df = pd.DataFrame(results_df)
results_df.to_csv(output_file, index=False)
results_df.to_excel(excel_file, index=False)

print(yolo_model_img_reso)

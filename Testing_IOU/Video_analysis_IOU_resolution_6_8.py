import os
import random
import subprocess
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from skimage.segmentation import mark_boundaries, slic
from ultralytics import YOLO

output_video_path = os.path.join('./Results', "comparison_video00.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the video file


def apply_superpixel_majority(image_rgb, mask_image):
    # Ensure that mask_image is of integer type
    mask_image = mask_image.astype(np.int32)  # Convert to integer type

    # Get the height of the image
    height = image_rgb.shape[0]

    # Initialize the modified mask
    modified_mask = np.zeros_like(mask_image)
    val = [0, 0.4 * height, 0.8 * height, 0.9 * height, 1 * height]
    n_segments_list = [10000, 5000, 2000, 400]
    # Process the image with varying superpixel configurations based on height
    for i in range(0, len(val) - 1):
        y = val[i]
        n_segments = n_segments_list[i]
        nextY = val[i + 1]
        # print(y, 0.9 * height, 0.8 * height, 0.4 * height)
        # if y > 0.9 * height:
        #     nextY = 1 * height - 1
        #     n_segments = 100
        # elif 0.8 * height < y <= 0.9 * height:
        #     nextY = 0.9 * height - 1
        #     n_segments = 1000
        # elif 0.4 * height < y <= 0.8 * height:
        #     nextY = 0.8 * height - 1
        #     n_segments = 5000
        # else:
        #     nextY = 0.4 * height - 1
        #     n_segments = 10000
        nextY = int(nextY)
        y = int(y)
        # print(y, nextY, image_rgb.shape)
        # Process the specific row of the image using SLIC
        row_segments = slic(image_rgb[y:nextY], n_segments=n_segments, compactness=10, start_label=1)

        # Iterate over each unique region in the current row
        for region_id in np.unique(row_segments):
            # Create a mask for the current region
            region_mask = (row_segments == region_id)

            # Calculate the majority label within the region
            # majority_label = np.bincount(mask_image[y:nextY][region_mask].flatten()).argmax()
            # Assign positive label if any pixel in the region is positive
            if np.any(mask_image[y:nextY][region_mask] == 1):  # Check if there's any positive pixel
                majority_label = 1
            else:
                majority_label = 0  # Assign negative label otherwise
            # Assign the determined label to the region
            # modified_mask[region_mask] = majority_label
            # Assign the majority label to the region
            modified_mask[y:nextY][region_mask] = majority_label

    # Optionally visualize the processed image
    segments_display = mark_boundaries(image_rgb, modified_mask)
    segments_display = (segments_display * 255).astype(np.uint8)

    cv2.imshow("segments_display", segments_display)
    # cv2.imwrite("segments_display.png", segments_display)
    return modified_mask, segments_display


def superpixels_methods(image, mask_image):
    # Convert the input image to RGB for skimage compatibility
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the superpixels based on majority color in mask image
    majority_mask_image, segments_display_img = apply_superpixel_majority(image_rgb, mask_image)

    return majority_mask_image, segments_display_img


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
original_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\OriginalVideo.mp4"
mask_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\MaskVideo.mp4"
output_file = r"./Results/evaluation_results6_8_2w.txt"
excel_file = r"./Results/evaluation_results6_8_2w.xlsx"
output_dir = r"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\"

resolution_results = []
image_sizes = [
    (640, 480),
]

# Load YOLOv8 model
model = YOLO(
    r'F:\\RunningProjects\\YOLO_Model\\Training\\runs\\segment\\RoadSegmentationForMyDataset9\\weights\\best.pt').cuda()

# Video capture objects
original_cap = cv2.VideoCapture(original_video_path)
mask_cap = cv2.VideoCapture(mask_video_path)

frame_count = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
selected_indices = random.sample(range(frame_count), max(1, frame_count // 1))
selected_indices = set(selected_indices)

fps = original_cap.get(cv2.CAP_PROP_FPS)  # Use the FPS of the original video
frame_width, frame_height = image_sizes[0]  # Assuming single resolution is used
grid_width = frame_width * 2
grid_height = frame_height * 2
# Create a VideoWriter object
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (grid_width, grid_height))

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

    frame_idx = 0
    while True:
        ret_original, original_frame = original_cap.read()
        ret_mask, mask_frame = mask_cap.read()

        if not ret_original or not ret_mask:
            break

        if frame_idx not in selected_indices and frame_idx % 10 != 0:
            frame_idx += 1
            continue

        # Resize frames
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
        modified_mask, segments_display_img = superpixels_methods(original_resized, predicted_mask)

        # Visualize the difference (you can display or save it)
        cv2.imshow("predicted_mask", (predicted_mask * 255).astype(np.uint8))
        # cv2.imwrite("predicted_mask.png", (predicted_mask * 255).astype(np.uint8))

        cv2.imshow("modified_mask", (modified_mask * 255).astype(np.uint8))
        # cv2.imwrite("modified_mask.png", (modified_mask * 255).astype(np.uint8))
        difference = np.abs(predicted_mask - modified_mask)
        difference_image = cv2.applyColorMap(difference.astype(np.uint8) * 255, cv2.COLORMAP_JET)
        # Stack images in a 2x2 grid
        top_row = np.hstack((segments_display_img, difference_image))
        bottom_row = np.hstack(((modified_mask * 255).astype(np.uint8), (predicted_mask * 255).astype(np.uint8)))
        bottom_row = cv2.cvtColor(bottom_row, cv2.COLOR_GRAY2BGR)
        combined_frame = np.vstack((top_row, bottom_row))

        # Write the combined frame to the video
        video_writer.write(combined_frame)
        cv2.imshow("Difference", difference_image)  # Display the difference

        predicted_mask = (modified_mask > 0.5).astype(np.uint8)
        # cv2.imwrite("Difference.png", difference_image)  # Display the difference
        # cv2.waitKey(0)  # Wait for keypress to close the image window
        # cv2.destroyAllWindows()
        key = cv2.waitKey(1)

        # Optionally save the difference image
        # cv2.imwrite("difference_image.png", difference_image)
        # converted_image = np.zeros(
        #     (predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
        # converted_image[ground_truth == 1] = [0, 0, 255]

        # print(np.unique(predicted_mask), np.unique(ground_truth))
        # print(predicted_mask.shape, ground_truth.shape, original_frame.shape)

        # Save overlay images for the selected 10%
        if frame_idx in selected_indices:
            overlay = original_resized.copy()
            overlay[(predicted_mask == 1) & (ground_truth == 1)] = [0, 255, 0]  # Green for true positives
            overlay[(predicted_mask == 0) & (ground_truth == 1)] = [0, 0, 255]  # Red for false negatives
            overlay[(predicted_mask == 1) & (ground_truth == 0)] = [255, 0, 0]  # Blue for false positives

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
            output_path = os.path.join(output_dir_res, f"overlay_frame_{frame_idx}.png")
            cv2.imwrite(output_path, overlay)

        num_images_processed += 1
        # Calculate metrics
        intersection = np.sum((predicted_mask == 1) & (ground_truth == 1))
        union = np.sum(predicted_mask) + np.sum(ground_truth) - intersection
        iou = intersection / union if union > 0 else 0
        total_iou += iou

        tp = intersection
        tn = np.sum((predicted_mask == 0) & (ground_truth == 0))
        fp = np.sum((predicted_mask == 1) & (ground_truth == 0))
        fn = np.sum((predicted_mask == 0) & (ground_truth == 1))

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        frame_idx += 1

    # Calculate final metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    avg_cpu = np.mean([p[0] for p in system_parameters]) if system_parameters else 0
    avg_ram = np.mean([p[1] for p in system_parameters]) if system_parameters else 0
    avg_gpu = np.mean([p[2] for p in system_parameters]) if system_parameters else 0

    mean_iou = total_iou / num_images_processed if num_images_processed > 0 else 0
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
        "Images Processed": num_images_processed
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

# Save results to files
results_df = pd.DataFrame(results_df)
results_df.to_csv(output_file, index=False)
results_df.to_excel(excel_file, index=False)

# Release video resources
video_writer.release()
original_cap.release()
mask_cap.release()

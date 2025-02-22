import subprocess
import time

import cv2
import numpy as np
import pandas as pd
import psutil


def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, text=True)
    memory_info = result.stdout.strip().split('\n')
    for i, info in enumerate(memory_info):
        used, total = map(int, info.split(', '))
        return 100 * (used / total)


def get_system_usage():
    """Returns the current system usage metrics."""
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    gpu_usage = get_gpu_memory()
    return cpu_usage, ram_usage, gpu_usage


start_time = time.time()
cpu_start, ram_start, gpu_start = get_system_usage()
# Paths
original_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\originalVideo.mp4"
mask_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\maskVideo.mp4"
output_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\OverlayVideo.mp4"
summary_excel_file = r"/Testing_IOU\\Results\\summary_results_video7.xlsx"

resolution_results = []
# Open original and mask videos
cap_original = cv2.VideoCapture(original_video_path)
cap_mask = cv2.VideoCapture(mask_video_path)

if not cap_original.isOpened() or not cap_mask.isOpened():
    print("Error: Unable to open input videos.")
    exit()

# Get video properties
fps = int(cap_original.get(cv2.CAP_PROP_FPS))
frame_width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Metrics storage
total_iou, total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0, 0
frame_count = 0

# Load YOLOv8 model
model = YOLO(
    r'F:\\RunningProjects\\YOLO_Model\\Training\\runs\\segment\\RoadSegmentationForMyDataset9\\weights\\best.pt')

resolution_results.append({"Resolution": f"{width}x{height}", "Metric": "Initial System Usage",
                           "CPU (%)": _cpu_start, "RAM (%)": _ram_start, "GPU (%)": _gpu_start})

while cap_original.isOpened() and cap_mask.isOpened():
    ret_original, original_frame = cap_original.read()
    ret_mask, mask_frame = cap_mask.read()

    if not ret_original or not ret_mask:
        break

    # Convert mask frame to binary format
    gray_mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    predicted_mask = (gray_mask > 128).astype(np.uint8)  # Threshold mask

    # Generate dummy ground truth (replace with actual ground truth)
    ground_truth_binary = np.zeros_like(predicted_mask)  # Replace this with actual ground truth

    # Overlay processing
    overlay = original_frame.copy()
    overlay[(predicted_mask == 1) & (ground_truth_binary == 1)] = [0, 255, 0]  # Green for true positives
    overlay[(predicted_mask == 0) & (ground_truth_binary == 1)] = [0, 0, 255]  # Red for false negatives
    overlay[(predicted_mask == 1) & (ground_truth_binary == 0)] = [255, 0, 0]  # Blue for false positives

    # Write processed frame to video
    video_writer.write(overlay)

    # Metrics calculation
    intersection = np.sum((predicted_mask == 1) & (ground_truth_binary == 1))
    union = np.sum(predicted_mask) + np.sum(ground_truth_binary) - intersection
    iou = intersection / union if union > 0 else 0

    total_iou += iou
    total_tp += intersection
    total_tn += np.sum((predicted_mask == 0) & (ground_truth_binary == 0))
    total_fp += np.sum((predicted_mask == 1) & (ground_truth_binary == 0))
    total_fn += np.sum((predicted_mask == 0) & (ground_truth_binary == 1))

    frame_count += 1

# Release video resources
cap_original.release()
cap_mask.release()
video_writer.release()

# Final metrics
end_time = time.time()
num_frames = frame_count
mean_iou = total_iou / num_frames if num_frames > 0 else 0
cpu_end, ram_end, gpu_end = get_system_usage()

# Summary metrics
summary_data = {
    "Resolution": f"{frame_width}x{frame_height}",
    "Mean IoU": round(mean_iou, 4),
    "Total TP": total_tp,
    "Total TN": total_tn,
    "Total FP": total_fp,
    "Total FN": total_fn,
    "Avg CPU (%)": round((cpu_end + cpu_start) / 2, 2),
    "Avg RAM (%)": round((ram_end + ram_start) / 2, 2),
    "Avg GPU (%)": round((gpu_end + gpu_start) / 2, 2),
    "Total Time (s)": round(end_time - start_time, 2),
    "Frames Processed": num_frames,
}

# Save summary to Excel
summary_df = pd.DataFrame([summary_data])
summary_df.to_excel(summary_excel_file, index=False)

# Summary
print("Processing Complete!")
print(f"Summary saved to: {summary_excel_file}")
print(f"Overlay video saved to: {output_video_path}")

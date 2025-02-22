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


start_time = time.time()
cpu_start, ram_start, gpu_start = get_system_usage()

original_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\originalVideo.mp4"
mask_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\maskVideo.mp4"
output_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\OverlayVideo.mp4"
summary_excel_file = r"/Testing_IOU\\Results\\summary_results_video7.xlsx"

resolution_results = []
image_sizes = [
    # 4:3 Aspect Ratio
    (384, 480),
    # (800, 600), (1024, 768), (1280, 960), (1600, 1200), (2048, 1536),
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
model = YOLO(
    r'F:\\RunningProjects\\YOLO_Model\\Training\\runs\\segment\\RoadSegmentationForMyDataset9\\weights\\best.pt').cuda()

# Open original and mask videos
cap_original = cv2.VideoCapture(original_video_path)
cap_mask = cv2.VideoCapture(mask_video_path)

if not cap_original.isOpened() or not cap_mask.isOpened():
    print("Error: Unable to open input videos.")
    exit()

# Get video properties
fps = int(cap_original.get(cv2.CAP_PROP_FPS))
frame_width = image_sizes[0][0]
frame_height = image_sizes[0][1]

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Metrics storage
total_iou, total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0, 0
frame_count = 0

resolution_results.append({"Resolution": f"{frame_width}x{frame_height}", "Metric": "Initial System Usage",
                           "CPU (%)": cpu_start, "RAM (%)": ram_start, "GPU (%)": gpu_start})
system_parameters = []
while cap_original.isOpened() and cap_mask.isOpened():
    ret_original, original_frame = cap_original.read()
    ret_mask, mask_frame = cap_mask.read()

    if not ret_original or not ret_mask:
        break
    original_frame = cv2.resize(original_frame, (frame_width, frame_height))
    mask_frame = cv2.resize(mask_frame, (frame_width, frame_height))

    # Convert mask frame to binary format
    # mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    # mask_frame = (mask_frame > 128).astype(np.uint8)  # Threshold mask

    # Ground truth mask to binary
    ground_truth_binary = np.all(mask_frame == [0, 0, 255], axis=-1).astype(np.uint8)

    # Model inference
    inference_results = model(original_frame, imgsz=(frame_width, frame_height), task='segment', conf=0.8)
    cpu, ram, gpu = get_system_usage()
    system_parameters.append([cpu - cpu_start, ram - ram_start, gpu - gpu_start])

    if not (inference_results[0].masks is not None and inference_results[0].masks.data is not None):
        predicted_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    else:
        predicted_mask = inference_results[0].masks.data[0].cpu().numpy()
        predicted_mask = cv2.resize(predicted_mask, (frame_width, frame_height))
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

    # Create overlay image
    overlay = original_frame.copy()

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

    video_writer.write(overlay)
    # Update confusion metrics
    tp = intersection
    tn = np.sum((predicted_mask == 0) & (ground_truth_binary == 0))
    fp = np.sum((predicted_mask == 1) & (ground_truth_binary == 0))
    fn = np.sum((predicted_mask == 0) & (ground_truth_binary == 1))

    total_tp += tp
    total_tn += tn
    total_fp += fp
    total_fn += fn
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

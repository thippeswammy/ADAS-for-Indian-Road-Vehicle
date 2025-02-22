import os
import time

import GPUtil
import cv2
import numpy as np
import psutil
from ultralytics import YOLO

# Paths
IMAGE_DIR = r"D:\downloadFiles\front_3\TestingVideo\TempImg"
MASK_DIR = r"D:\downloadFiles\front_3\TestingVideo\TempMasks"
OUTPUT_DIR = r"F:\RunningProjects\YOLO_Model\val\Results"
MODEL_PATH = r'F:\RunningProjects\YOLO_Model\Training\runs\segment\RoadSegmentationForMyDataset9\weights\best.pt'
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Results.txt")

# Desired image resolutions for testing
IMAGE_SIZES = [
    (384, 640), (640, 640), (1920, 1080), (1280, 720),
    (640, 480), (1024, 768), (800, 600), (320, 240)
]


# Functions
def log_initial_system_usage():
    """Logs the initial system usage of CPU, RAM, and GPU."""
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    gpu = gpus[0].load * 100 if gpus else None
    return cpu, ram, gpu


def write_to_file(filepath, text, mode="a"):
    """Writes the given text to a file."""
    with open(filepath, mode, encoding="utf-8") as file:
        file.write(text)


def resize_image(image, size):
    """Resizes the image to the specified size."""
    return cv2.resize(image, size)


def process_image(image_path, mask_path, image_size, model):
    """Processes a single image and calculates metrics."""
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # Resize image and ground truth mask
    image_resized = resize_image(image, image_size)
    mask_resized = resize_image(mask, image_size)

    # Convert ground truth mask to binary
    ground_truth_binary = np.all(mask_resized == [0, 0, 255], axis=-1).astype(np.uint8)

    # Get YOLO predictions, explicitly specifying the image size
    results = model(image_resized, imgsz=image_size, task='segment', conf=0.5)

    if results[0].masks is not None and results[0].masks.data is not None:
        predicted_mask = results[0].masks.data[0].cpu().numpy()

        # Resize predicted mask to match ground truth mask
        predicted_mask = cv2.resize(predicted_mask, (image_size[1], image_size[0]))
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

        # Verify shapes match
        if predicted_mask.shape != ground_truth_binary.shape:
            print(f"Shape mismatch: predicted {predicted_mask.shape}, ground truth {ground_truth_binary.shape}")
            return 0, 0, 0, 0

        # Calculate TP, TN, FP, FN
        tp = np.sum((predicted_mask == 1) & (ground_truth_binary == 1))
        tn = np.sum((predicted_mask == 0) & (ground_truth_binary == 0))
        fp = np.sum((predicted_mask == 1) & (ground_truth_binary == 0))
        fn = np.sum((predicted_mask == 0) & (ground_truth_binary == 1))

        return tp, tn, fp, fn

    return 0, 0, 0, 0


isStarting = True


def evaluate_resolution(image_size, image_files, model):
    """Evaluates a specific resolution and calculates metrics."""
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    total_iou = 0
    global isStarting
    isStarting = True
    for image_file in image_files:
        image_path = os.path.join(IMAGE_DIR, image_file)
        mask_path = os.path.join(MASK_DIR, image_file)

        tp, tn, fp, fn = process_image(image_path, mask_path, image_size, model)

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0
        total_iou += iou

    mean_iou = total_iou / len(image_files) if image_files else 0
    return mean_iou, total_tp, total_tn, total_fp, total_fn


def log_system_usage(cpu_usages, ram_usages, gpu_usages):
    """Logs the average system usage."""
    avg_cpu = sum(cpu_usages) / len(cpu_usages)
    avg_ram = sum(ram_usages) / len(ram_usages)
    avg_gpu = sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0
    return avg_cpu, avg_ram, avg_gpu


def main():
    # Initialize
    write_to_file(OUTPUT_FILE, "Initial System Usage:\n", "w")
    cpu_before, ram_before, gpu_before = log_initial_system_usage()
    write_to_file(OUTPUT_FILE, f"CPU Usage: {cpu_before:.2f}%\n")
    write_to_file(OUTPUT_FILE, f"RAM Usage: {ram_before:.2f}%\n")
    if gpu_before is not None:
        write_to_file(OUTPUT_FILE, f"GPU Usage: {gpu_before:.2f}%\n")
    else:
        write_to_file(OUTPUT_FILE, "GPU Usage: No GPU detected.\n")
    write_to_file(OUTPUT_FILE, "\n")

    # Load model
    model = YOLO(MODEL_PATH)

    # Process images for each resolution
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    final_cpu_usages, final_ram_usages, final_gpu_usages = [], [], []

    for image_size in IMAGE_SIZES:
        write_to_file(OUTPUT_FILE, f"Evaluating for resolution: {image_size[1]}x{image_size[0]}\n")
        start_time = time.time()

        mean_iou, total_tp, total_tn, total_fp, total_fn = evaluate_resolution(image_size, image_files, model)

        elapsed_time = time.time() - start_time
        write_to_file(OUTPUT_FILE, f"Mean IoU: {mean_iou:.4f}\n")
        write_to_file(OUTPUT_FILE, f"Total True Positives: {total_tp}\n")
        write_to_file(OUTPUT_FILE, f"Total True Negatives: {total_tn}\n")
        write_to_file(OUTPUT_FILE, f"Total False Positives: {total_fp}\n")
        write_to_file(OUTPUT_FILE, f"Total False Negatives: {total_fn}\n")
        write_to_file(OUTPUT_FILE, f"Total Time: {elapsed_time:.2f} seconds\n")
        write_to_file(OUTPUT_FILE, f"Images Processed: {len(image_files)}\n\n")

        # Track final system usage
        cpu_after, ram_after, gpu_after = log_initial_system_usage()
        final_cpu_usages.append(cpu_after)
        final_ram_usages.append(ram_after)
        if gpu_after is not None:
            final_gpu_usages.append(gpu_after)

    # Log average final system usage
    avg_cpu, avg_ram, avg_gpu = log_system_usage(final_cpu_usages, final_ram_usages, final_gpu_usages)
    write_to_file(OUTPUT_FILE, "Average Final System Usage:\n")
    write_to_file(OUTPUT_FILE, f"CPU Usage: {avg_cpu:.2f}%\n")
    write_to_file(OUTPUT_FILE, f"RAM Usage: {avg_ram:.2f}%\n")
    if avg_gpu > 0:
        write_to_file(OUTPUT_FILE, f"GPU Usage: {avg_gpu:.2f}%\n")
    else:
        write_to_file(OUTPUT_FILE, "GPU Usage: No GPU detected.\n")


# Run the main function
if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np
from ultralytics import YOLO

# Paths to images and masks
image_dir = r"D:\downloadFiles\front_3\TestingVideo\TempImg"
mask_dir = r"D:\downloadFiles\front_3\TestingVideo\TempMasks"

# Load YOLOv8 model
model = YOLO(
    r'F:\RunningProjects\YOLO_Model\Training\runs\segment\RoadSegmentationForMyDataset9\weights\best.pt')  # Replace 'seg.pt' with the path to your YOLOv8 segmentation model

# Metrics initialization
total_iou = 0
total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

# Iterate through images
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, image_file)

    # Load image and ground truth mask
    image = cv2.imread(image_path)
    ground_truth_mask = cv2.imread(mask_path)

    # Convert ground truth mask to binary (1 for road, 0 for background)
    ground_truth_binary = np.all(ground_truth_mask == [0, 0, 255], axis=-1).astype(np.uint8)  # Red is (255, 0, 0)

    # Get predictions from the YOLOv8 model
    results = model(image, task='segment', conf=0.8)

    # Check if any predictions are available
    if results[0].masks is not None and results[0].masks.data is not None:
        predicted_mask = results[0].masks.data[0].cpu().numpy()  # Binary mask

        # Resize predicted mask to match ground truth mask
        predicted_mask = cv2.resize(predicted_mask, (ground_truth_binary.shape[1], ground_truth_binary.shape[0]))
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Ensure binary mask

        # Calculate TP, TN, FP, FN
        tp = np.sum((predicted_mask == 1) & (ground_truth_binary == 1))
        tn = np.sum((predicted_mask == 0) & (ground_truth_binary == 0))
        fp = np.sum((predicted_mask == 1) & (ground_truth_binary == 0))
        fn = np.sum((predicted_mask == 0) & (ground_truth_binary == 1))

        # Update totals
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        # Calculate IoU
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0
        total_iou += iou
    else:
        print(f"No predictions for image: {image_file}")

# Final metrics
num_images = len(image_files)
mean_iou = total_iou / num_images if num_images > 0 else 0

print(f"Mean IoU: {mean_iou:.4f}")
print(f"Total True Positives: {total_tp}")
print(f"Total True Negatives: {total_tn}")
print(f"Total False Positives: {total_fp}")
print(f"Total False Negatives: {total_fn}")

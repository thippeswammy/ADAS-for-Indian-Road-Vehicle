import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from ultralytics import YOLO


# Function to compute and save ROC curve
def compute_and_save_roc(ground_truth, predicted_probs, output_dir):
    if len(ground_truth) != len(predicted_probs):
        print(
            f"Error: Ground truth and predicted probabilities lengths do not match! ({len(ground_truth)} vs {len(predicted_probs)})")
        return None, None, None

    fpr, tpr, thresholds = roc_curve(ground_truth, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Chance')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for YOLOv8 Segmentation Model')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()

    # Save the plot
    roc_plot_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_plot_path)
    plt.show()

    # Save parameters to Excel
    roc_data = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Threshold': thresholds
    })
    roc_data['AUC'] = roc_auc
    excel_path = os.path.join(output_dir, 'roc_parameters.xlsx')
    roc_data.to_excel(excel_path, index=False)

    return roc_auc, roc_plot_path, excel_path


# Function to preprocess the mask for evaluation
def preprocess_mask(mask, shape):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
    mask_resized = cv2.resize(mask, shape, interpolation=cv2.INTER_NEAREST)
    return (mask_resized > 10).astype(np.uint8)


def create_unique_folder(base_dir):
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
original_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\OriginalVideo.mp4"
mask_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\MaskVideo.mp4"
output_dir = r"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\ROC"
output_dir = create_unique_folder(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Load YOLOv8 model
model = YOLO(
    r'F:\\RunningProjects\\YOLO_Model\\Training\\runs\\segment\\RoadSegmentationForMyDataset9\\weights\\best.pt').cuda()

# Initialize video capture
original_cap = cv2.VideoCapture(original_video_path)
mask_cap = cv2.VideoCapture(mask_video_path)

frame_count = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = 854
frame_height = 480

# Initialize lists for ground truth and predicted probabilities
all_ground_truth = []
all_predicted_probs = []

frame_idx = 0

while True:
    ret_original, original_frame = original_cap.read()
    ret_mask, mask_frame = mask_cap.read()

    if not ret_original or not ret_mask:
        break

    # Resize and preprocess ground truth mask
    mask_frame = preprocess_mask(mask_frame, (frame_width, frame_height))
    original_frame = cv2.resize(original_frame, (frame_width, frame_height))
    # original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

    # Model inference
    inference_results = model(original_frame, imgsz=(frame_width, frame_height), task='segment', conf=0.5)
    if inference_results[0].masks is not None and inference_results[0].masks.data is not None:
        predicted_probs = inference_results[0].masks.data[0].cpu().numpy()
        predicted_probs_resized = cv2.resize(predicted_probs, (frame_width, frame_height),
                                             interpolation=cv2.INTER_NEAREST)
    else:
        predicted_probs_resized = np.zeros((frame_height, frame_width), dtype=np.float32)

    # Add the current frame's data to the lists
    all_ground_truth.extend(mask_frame.flatten())
    all_predicted_probs.extend(predicted_probs_resized.flatten())

    frame_idx += 1
    # if frame_idx > 100:
    #     break
    print(f"Processed frame {frame_idx}/{frame_count}")

# # Save all ground truth and predicted probabilities to Excel
# combined_data = pd.DataFrame({
#     'Ground_Truth': all_ground_truth,
#     'Predicted_Probs': all_predicted_probs
# })
# excel_path_combined = os.path.join(output_dir, 'all_ground_truth_and_predicted_probs.xlsx')
# combined_data.to_excel(excel_path_combined, index=False)
# print(f"Saved all ground truth and predicted probabilities to: {excel_path_combined}")

# Compute and save ROC curve
roc_auc, roc_plot_path, excel_path = compute_and_save_roc(all_ground_truth, all_predicted_probs, output_dir)

if roc_auc is not None:
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"ROC curve saved to: {roc_plot_path}")
    print(f"ROC parameters saved to: {excel_path}")
else:
    print("Error: Inconsistent ground truth and predicted probabilities length. ROC computation failed.")

# Release video resources
original_cap.release()
mask_cap.release()

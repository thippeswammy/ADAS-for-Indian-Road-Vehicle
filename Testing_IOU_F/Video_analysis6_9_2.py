import os

import cv2
import numpy as np
import pandas as pd
from skimage.segmentation import mark_boundaries, slic
from ultralytics import YOLO


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


def apply_segment_any_one_pixel(image_rgb, mask_image, n_segments):
    segments = slic(image_rgb, n_segments=int(5000 * n_segments), compactness=10, start_label=1)
    segments_display = mark_boundaries(image_rgb, segments)
    segments_display = (segments_display * 255).astype(np.uint8)
    modified_mask = np.zeros_like(mask_image)
    for region_id in np.unique(segments):
        region_mask = (segments == region_id)
        if np.any(mask_image[region_mask] == 1):
            majority_label = 1
        else:
            majority_label = 0
        modified_mask[region_mask] = majority_label
    return modified_mask, segments_display


def apply_segment_majority_pixel(image_rgb, mask_image, n_segments):
    segments = slic(image_rgb, n_segments=int(5000 * n_segments), compactness=5, start_label=1)
    segments_display = mark_boundaries(image_rgb, segments)
    segments_display = (segments_display * 255).astype(np.uint8)
    modified_mask = np.zeros_like(mask_image)
    for region_id in np.unique(segments):
        region_mask = (segments == region_id)
        majority_label = np.bincount(mask_image[region_mask].flatten()).argmax()
        modified_mask[region_mask] = majority_label
    return modified_mask, segments_display


def apply_segment_varying_superpixel_pixel(image_rgb, mask_image, n_segments):
    height = image_rgb.shape[0]
    modified_mask = np.zeros_like(mask_image)
    val = [0, 0.35 * height, 0.6 * height, 0.85 * height, 1 * height]
    n_segments_list = [2000 * n_segments, 1000 * n_segments, 500 * n_segments, 200 * n_segments]
    segments = np.zeros_like(mask_image)
    for i in range(0, len(val) - 1):
        y = int(val[i])
        n_segments_ = int(n_segments_list[i])
        nextY = int(val[i + 1])
        row_segments = slic(image_rgb[y:nextY], n_segments=n_segments_, compactness=10, start_label=1)
        segments[y:nextY] = row_segments
        for region_id in np.unique(row_segments):
            region_mask = (row_segments == region_id)
            if np.any(mask_image[y:nextY][region_mask] == 1):
                majority_label = 1
            else:
                majority_label = 0
            modified_mask[y:nextY][region_mask] = majority_label
    segments_display = mark_boundaries(image_rgb, segments)
    segments_display = (segments_display * 255).astype(np.uint8)
    return modified_mask, segments_display


def superpixels_methods(image, mask_image, n_segments, user_input=1):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_image = mask_image.astype(np.int32)

    if user_input == 1:
        modified_mask, segments_display = apply_segment_any_one_pixel(image_rgb, mask_image, n_segments)
    elif user_input == 2:
        modified_mask, segments_display = apply_segment_majority_pixel(image_rgb, mask_image, n_segments)
    elif user_input == 3:
        modified_mask, segments_display = apply_segment_varying_superpixel_pixel(image_rgb, mask_image, n_segments)
    elif user_input == 4:
        modified_mask, segments_display = mask_image, image

    return modified_mask, segments_display


def save_overlay_image(original_frame, modified_mask, segments_display, output_path):
    overlay = original_frame.copy()
    overlay[modified_mask == 1] = [0, 255, 0]  # Green for modified mask
    overlay_image = cv2.addWeighted(overlay, 0.5, segments_display, 0.5, 0)
    cv2.imwrite(output_path, overlay_image)


def main():
    original_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\OriginalVideo.mp4"
    mask_video_path = r"D:\\downloadFiles\\front_3\\TestingVideo\\MaskVideo.mp4"
    output_dir = r"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\SuperPixelMethods"
    val = ['any one pixel', 'majority pixel', 'varying superpixel pixel', 'with out superpixel pixel']
    output_dir = create_unique_folder(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    excel_file = os.path.join(output_dir, "results.xlsx")

    # Load YOLO model
    model = YOLO(
        r"F:\\RunningProjects\\YOLO_Model\\Training\\runs\\segment\\RoadSegmentationForMyDataset9\\weights\\best.pt").cuda()

    # Video capture objects
    original_cap = cv2.VideoCapture(original_video_path)
    mask_cap = cv2.VideoCapture(mask_video_path)

    frame_count = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = original_cap.get(cv2.CAP_PROP_FPS)
    image_sizes = [854, 480]
    frame_width = image_sizes[0]
    frame_height = image_sizes[1]

    user_inputs = [1, 2, 3, 4]
    n_segments_list = [i / 10 for i in range(1, 11)]
    results = []
    frame_idx = 0
    # selected_indices = random.sample(range(frame_count), max(1, (frame_count // 100) * 5))
    # selected_indices = [13, 46, 51, 57, 63, 74, 108, 119, 120, 157, 162, 192, 242, 284, 285, 302, 313, 329, 370, 412,
    #                     424, 438, 444, 465, 497, 546, 574, 604, 631, 637, 644, 648, 661, 673, 705, 715, 718, 738, 791,
    #                     805, 809]

    while True:
        ret_original, original_frame = original_cap.read()
        ret_mask, mask_frame = mask_cap.read()

        if not ret_original or not ret_mask:
            break

        original_frame = cv2.resize(original_frame, (frame_width, frame_height))
        mask_frame = cv2.resize(mask_frame, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        mask_resized = cv2.cvtColor(mask_frame, cv2.COLOR_BGRA2GRAY)
        ground_truth = (mask_resized > 1).astype(np.uint8)

        # YOLO model inference
        inference_results = model(original_frame, imgsz=(frame_width, frame_height), task='segment', conf=0.8)
        if inference_results[0].masks is not None and inference_results[0].masks.data is not None:
            predicted_mask_by_model = inference_results[0].masks.data[0].cpu().numpy()
        else:
            predicted_mask_by_model = np.zeros((frame_width, frame_height), dtype=np.uint8)

        for user_input in user_inputs:
            for n_segments in n_segments_list:
                if user_input == 4 and n_segments != n_segments_list[0]:
                    break
                method_dir = os.path.join(output_dir, f"{val[user_input - 1]}_segments_{int(n_segments * 10)}")
                os.makedirs(method_dir, exist_ok=True)
                predicted_mask = cv2.resize(predicted_mask_by_model, (frame_width, frame_height),
                                            interpolation=cv2.INTER_NEAREST)
                modified_mask, segments_display = superpixels_methods(original_rgb, predicted_mask,
                                                                      n_segments, user_input)
                difference_G_C = np.abs(ground_truth - modified_mask)
                difference_G_P = np.abs(ground_truth - predicted_mask)

                tp = np.sum((modified_mask == 1) & (ground_truth == 1))
                tn = np.sum((modified_mask == 0) & (ground_truth == 0))
                fp = np.sum((modified_mask == 1) & (ground_truth == 0))
                fn = np.sum((modified_mask == 0) & (ground_truth == 1))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                results.append({
                    "Frame": frame_idx,
                    "methode": val[user_input - 1],
                    "n_segments": n_segments,
                    "IoU": round(iou, 4),
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1-Score": round(f1_score, 4),
                    "TP": tp,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn
                })
                os.makedirs(os.path.join(method_dir, "predicted_mask"), exist_ok=True)
                os.makedirs(os.path.join(method_dir, "modified_mask"), exist_ok=True)
                os.makedirs(os.path.join(method_dir, "difference_mask_C"), exist_ok=True)
                os.makedirs(os.path.join(method_dir, "difference_mask_P"), exist_ok=True)
                os.makedirs(os.path.join(method_dir, "segments_images"), exist_ok=True)
                os.makedirs(os.path.join(method_dir, "overlay"), exist_ok=True)

                # Save images
                cv2.imwrite(os.path.join(method_dir, "predicted_mask", f"predicted_mask_frame_{frame_idx}.png"),
                            (predicted_mask * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(method_dir, "modified_mask", f"modified_mask_frame_{frame_idx}.png"),
                            (modified_mask * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(method_dir, "difference_mask_C", f"difference_frame_{frame_idx}.png"),
                            (difference_G_C * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(method_dir, "difference_mask_P", f"difference_frame_{frame_idx}.png"),
                            (difference_G_P * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(method_dir, "segments_images", f"segments_display_frame_{frame_idx}.png"),
                            segments_display)
                overlay_path = os.path.join(method_dir, "overlay", f"overlay_frame_{frame_idx}.png")
                save_overlay_image(original_frame, modified_mask, segments_display, overlay_path)

        frame_idx += 1

    # Save results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(excel_file, index=False)

    columns_to_average = results_df.columns[3:]  # Assuming numeric data starts from the 4th column

    # Grouping by 'Method' and 'n_segments', then calculating the average for each group
    grouped_averages = results_df.groupby(['methode', 'n_segments'])[columns_to_average].mean()

    # Resetting index to make 'Method' and 'n_segments' regular columns
    grouped_averages.reset_index(inplace=True)

    # Save the results to a new Excel file
    output_file = excel_file[:5] + "Average.xlsx"
    grouped_averages.to_excel(output_file, index=False)

    original_cap.release()
    mask_cap.release()
    print("Processing completed.")


if __name__ == "__main__":
    main()

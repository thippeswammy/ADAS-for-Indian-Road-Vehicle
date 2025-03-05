import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

colors = [
    (0, 255, 0),  # drivable fallback (bright green)
    (128, 64, 128),  # road (purple-gray)
    (255, 255, 255),  # motorcycle (bright red)
    (255, 255, 255),  # sky (light sky blue)
    (255, 255, 0),  # curb (white)
    (128, 128, 128),  # building (gray)
    (34, 139, 34),  # vegetation (forest green)
    (255, 140, 0),  # obs-str-bar-fallback (dark orange)
    (255, 228, 196),  # billboard (bisque)
    (255, 215, 0),  # auto rickshaw (gold)
    (218, 165, 32),  # pole (goldenrod)
    (0, 0, 255),  # car (bright blue)
    (139, 69, 19),  # truck (saddle brown)
    (255, 255, 0),  # person (bright yellow)
    (160, 82, 45),  # animal (sienna)
    (0, 255, 255),  # rider (cyan)
    (220, 20, 60),  # vehicle fallback (crimson)
    (255, 20, 147),  # non-drivable fallback (deep pink)
    (169, 169, 169),  # wall (dark gray)
    (240, 230, 140),  # fallback background (khaki)
    (0, 128, 0),  # fence (dark green)
    (255, 69, 0),  # traffic sign (red-orange)
    (47, 79, 79),  # guard rail (dark slate gray)
    (75, 0, 130),  # pole group (indigo)
    (255, 105, 180),  # bus (hot pink)
]
# Configuration settings inside the script
CONFIG = {
    "model1_path": "../Model/RoadSeg/weights/best.pt",
    "model2_path": "../Model/YoloPreTrained/yolo11x-seg.pt",
    "video_folder": "D:\\downloadFiles\\front_3\\",
    "output_folder": "D:\\downloadFiles\\front_3\\outputs\\",
    "log_file_path": "D:\\downloadFiles\\front_3\\processing_log.txt",
    "vehicle_classes": [1, 2, 3, 5, 7, 24, 16, 0],
    "resolution_list": [[854, 480]],
    "confidence_threshold": 0.75,
    "wantObjectMasks": True,
    "save_output": False,
    "transparency": 0.5,
    "display": True,
    "video_range": [2, 72],
}
SavingImageCount = 1


def safe_area(overlay, vehicle_positions):
    """Remove detected vehicles from safe area overlay."""
    for box in vehicle_positions:
        start, top, end, bottom = box[0], box[3], box[2], box[1]
        for y in range(overlay.shape[0] - 1, -1, -1):
            for x in range(overlay.shape[1] - 1, -1, -1):
                if (start <= x <= end and bottom <= y <= top) and (overlay[y][x] == (0, 255, 0)).all():
                    overlay[y][x] = (0, 0, 0)
    return overlay.astype(np.uint8)


def process_video(models, video_path, conf, display, save_path, transparency, input_size):
    """Run YOLO segmentation on a video."""
    cap = cv2.VideoCapture(video_path)
    global SavingImageCount
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return 0  # No frames processed

    frame_width, frame_height = input_size if input_size else (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if save_path:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    paused = False
    NumberPressed = 1
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if NumberPressed > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + NumberPressed)

            if input_size is not None:
                frame = cv2.resize(frame, (input_size[0], input_size[1]))

            combined_overlay = np.zeros_like(frame, dtype=np.uint8)
            TempFrame = frame.copy()
            for count, model in enumerate(models):
                results = model.predict(frame, conf=conf, imgsz=(frame_width, frame_height))
                for result in results:
                    vehicle_positions = []
                    if result.masks is not None:
                        for mask, box in zip(result.masks.xy, result.boxes):
                            cls_id = int(box.cls[0])
                            if cls_id in CONFIG["vehicle_classes"] and cls_id < len(colors) and count != 0:
                                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                                vehicle_positions.append([x_min, y_min, x_max, y_max])
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colors[cls_id], 2)
                            if (cls_id == 0 and count == 0) or CONFIG["wantObjectMasks"]:
                                points = np.int32(mask).reshape((-1, 1, 2))
                                # Clamp points within image boundaries
                                points[:, :, 0] = np.clip(points[:, :, 0], 0, frame_width - 1)
                                points[:, :, 1] = np.clip(points[:, :, 1], 0, frame_height - 1)
                                if points.size > 0:
                                    try:
                                        cv2.fillPoly(combined_overlay, [points], colors[cls_id])
                                    except cv2.error as e:
                                        print(f"Failed to draw polygon for class {cls_id}: {e}")

            combined_overlay = safe_area(combined_overlay, vehicle_positions)
            blended_frame = cv2.addWeighted(combined_overlay, transparency, frame, 1 - transparency, 0)

            if save_path:
                out.write(blended_frame)

        if display:
            cv2.imshow("Road Vehicle Segmentation", blended_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cv2.imwrite(f"{CONFIG['video_folder']}\\originalImages\\image{SavingImageCount}.png", TempFrame)
                cv2.imwrite(f"{CONFIG['video_folder']}\\outputMask\\image{SavingImageCount}.png", blended_frame)
                SavingImageCount = SavingImageCount + 1
            if key == ord('p'):  # Pause/unpause when "P" is pressed
                paused = not paused
            elif key in [ord(str(i)) for i in range(1, 10)]:  # Frame skip based on "1-9"
                NumberPressed = key - ord('1')  # If key = '2', skip 1 frame, '3' skip 2, and so on.
            elif key == ord('q'):  # Quit the loop
                break

    cap.release()
    if save_path:
        out.release()
    cv2.destroyAllWindows()
    return total_frames


def run_processing(config):
    """Run the YOLO segmentation process for all videos."""
    # models = [YOLO(config["model1_path"]).to('cuda'), YOLO(config["model2_path"]).to('cuda')]
    models = [YOLO(config["model1_path"]).to('cuda')]

    log_entries = []
    start_all_time = time.time()

    for i in range(config["video_range"][0], config["video_range"][1]):
        video_path = os.path.join(config["video_folder"], f"video{i}.mp4")

        if not os.path.exists(video_path):
            print(f"Skipping video {i}, not found.")
            log_entries.append(f"Video {i} not found.\n")
            continue

        for size in config["resolution_list"]:
            save_path = None
            if config["save_output"]:
                save_path = os.path.join(config["output_folder"], f"video{i}_{size[0]}_{size[1]}.mp4")

            print(f"Processing video {i} at resolution {size}...")
            start_time = time.time()

            total_frames = process_video(
                models=models,
                video_path=video_path,
                conf=config["confidence_threshold"],
                display=config["display"],
                save_path=save_path,
                transparency=config["transparency"],
                input_size=size,
            )

            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            fps = total_frames / elapsed_time if elapsed_time > 0 else 0

            log_entry = (
                f"Video {i}, Resolution: {size[0]}x{size[1]}, "
                f"Processing time: {int(hours)} hr, {int(minutes)} min, {seconds:.2f} sec, "
                f"FPS: {fps:.2f}\n"
            )
            log_entries.append(log_entry)
            print(log_entry)

    with open(config["log_file_path"], "a") as log_file:
        log_file.writelines(log_entries)

    total_elapsed = time.time() - start_all_time
    print(f"Total processing time: {total_elapsed:.2f} sec")


if __name__ == "__main__":
    run_processing(CONFIG)
    # Open the log file automatically after processing
    print(CONFIG["log_file_path"])

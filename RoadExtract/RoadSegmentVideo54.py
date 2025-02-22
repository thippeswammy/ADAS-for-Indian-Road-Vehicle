import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define class IDs and corresponding colors
vehicle_classes = [1, 2, 3, 5, 7, 24, 16, 0]
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
wantObjectMasks = False


def safe_area(overlay, vehiclePos):
    for box in vehiclePos:
        start, top, end, bottom = box[0], box[3], box[2], box[1]
        for y in range(overlay.shape[0] - 1, -1, -1):
            for x in range(overlay.shape[1] - 1, -1, -1):
                if (start <= x <= end and bottom <= y <= top) and (overlay[y][x] == (0, 255, 0)).all():
                    overlay[y][x] = (0,) * 3
    return overlay.astype(np.uint8)


def run_yolo_segmentation_on_video(models, video_path, conf=0.5, display=False, save_path=None, transparency=0.5,
                                   input_size=None):
    cap = cv2.VideoCapture(video_path)
    if input_size:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        frame_width = input_size[0]
        frame_height = input_size[1]
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if save_path:
        if not save_path.endswith(('.mp4', '.avi')):
            save_path += "\outputVideo.mp4"  # Ensure valid filename
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if not input_size:
            frame = cv2.resize(frame, (input_size[0], input_size[1]))
        combined_overlay = np.zeros_like(frame, dtype=np.uint8)
        for count, model in enumerate(models):
            results = model.predict(frame, conf=conf, imgsz=(input_size[0], input_size[1]))
            for result in results:
                vehiclePos = []
                if result.masks is not None:
                    for mask, box in zip(result.masks.xy, result.boxes):
                        cls_id = int(box.cls[0])
                        if cls_id in vehicle_classes and cls_id < len(colors) and count >= len(models) - 1:
                            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                            vehiclePos.append([x_min, y_min, x_max, y_max])
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colors[cls_id], 2)
                        if (cls_id == 0 and count != len(models) - 1) or wantObjectMasks:
                            points = np.int32(mask).reshape((-1, 1, 2))
                            # Clamp points within image boundaries
                            points[:, :, 0] = np.clip(points[:, :, 0], 0, frame_width - 1)
                            points[:, :, 1] = np.clip(points[:, :, 1], 0, frame_height - 1)
                            if points.size > 0:
                                try:
                                    cv2.fillPoly(combined_overlay, [points], colors[cls_id])
                                except cv2.error as e:
                                    print(f"Failed to draw polygon for class {cls_id}: {e}")
        combined_overlay = safe_area(combined_overlay, vehiclePos)

        # frame = cv2.addWeighted(combined_overlay, transparency, frame, 1 - transparency, 0)

        # Create a mask for non-black areas in the overlay
        mask = np.any(combined_overlay != 0, axis=-1, keepdims=True)
        # Blend only the segmented areas while keeping the original frame for the rest
        frame = np.where(mask, cv2.addWeighted(combined_overlay, transparency, frame, 1 - transparency, 0), frame)

        if save_path:
            out.write(frame)
        frame = cv2.resize(frame, (input_size[0], input_size[1]))
        if display:
            cv2.imshow("YOLO Combined Segmentation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if save_path:
        out.release()
    cv2.destroyAllWindows()
    return total_frames


def run_model_thread(model_paths, video_path, conf=0.5, display=False, save_path=None, transparency=0.5,
                     input_size=None):
    models = [YOLO(model_path).to('cuda') for model_path in model_paths]
    return run_yolo_segmentation_on_video(models=models, video_path=video_path, conf=conf,
                                          display=display, save_path=save_path,
                                          transparency=transparency, input_size=input_size)


if __name__ == "__main__":
    model1_path = r'F:\RunningProjects\YOLO_Model\Training\runs\segment\RoadSegmentationForMyDataset9\weights\best.pt'
    model2_path = r'yolo11x-seg.pt'
    for i in range(1, 2):
        print('=>' * 10, f"Processing video {i}", '<=' * 10)
        try:
            start_time = time.time()
            total_frames = run_model_thread(
                model_paths=[model1_path, model2_path], conf=0.75,
                display=True, save_path=r"D:\downloadFiles\front_3\Video", transparency=0.5,
                video_path=r"D:\downloadFiles\front_3\FullAllPossibleVideo.mp4",
                input_size=[854, 480]
            )
            elapsed_time = time.time() - start_time
            print(f"Processing time: {elapsed_time:.2f} seconds")
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Processing time: {int(hours)} hr, {int(minutes)} min, {seconds:.2f} sec")
            print(f"FPS: {total_frames / elapsed_time:.2f}")
            print(f"FPS: {total_frames / elapsed_time:.2f}")
        except FileNotFoundError as fnf_error:
            print(f"Video {i} not found: {fnf_error}")
        except Exception as e:
            print(f"Error processing video {i}: {e}")
    else:
        print(f"Skipping video {i}, not in the list.")

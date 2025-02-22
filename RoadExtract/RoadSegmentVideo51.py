import cv2
import numpy as np
from ultralytics import YOLO

# Define your class IDs and corresponding colors
vehicle_classes = [11, 12, 24, 16]  # Example vehicle class IDs: car, truck, bus, vehicle fallback
colors = [
    (0, 255, 0),  # 1: drivable fallback (bright green)
    (128, 64, 128),  # 0: road (purple-gray)
    (255, 255, 255),  # 2: motorcycle (bright red)
    (255, 255, 255),  # 3: sky (light sky blue)
    (255, 255, 0),  # 4: curb (white)
    (128, 128, 128),  # 5: building (gray)
    (34, 139, 34),  # 6: vegetation (forest green)
    (255, 140, 0),  # 7: obs-str-bar-fallback (dark orange)
    (255, 228, 196),  # 8: billboard (bisque)
    (255, 215, 0),  # 9: auto rickshaw (gold)
    (218, 165, 32),  # 10: pole (goldenrod)
    (0, 0, 255),  # 11: car (bright blue)
    (139, 69, 19),  # 12: truck (saddle brown)
    (255, 255, 0),  # 13: person (bright yellow)
    (160, 82, 45),  # 14: animal (sienna)
    (0, 255, 255),  # 15: rider (cyan)
    (220, 20, 60),  # 16: vehicle fallback (crimson)
    (255, 20, 147),  # 17: non-drivable fallback (deep pink)
    (169, 169, 169),  # 18: wall (dark gray)
    (240, 230, 140),  # 19: fallback background (khaki)
    (0, 128, 0),  # 20: fence (dark green)
    (255, 69, 0),  # 21: traffic sign (red-orange)
    (47, 79, 79),  # 22: guard rail (dark slate gray)
    (75, 0, 130),  # 23: pole group (indigo)
    (255, 105, 180),  # 24: bus (hot pink)
]


def safe_area(overlay, vehiclePos):
    for box in vehiclePos:
        blx, by, brx = box[0], box[3], box[2]
        mask_x = (blx <= np.arange(overlay.shape[1])) & (np.arange(overlay.shape[1]) <= brx)
        mask_y = (np.arange(overlay.shape[0]) <= by)
        mask = np.outer(mask_y, mask_x)
        color_mask = (overlay == (128, 64, 128)).all(axis=-1) & mask
        overlay[color_mask] = (255, 255, 255)
    return overlay.astype(np.uint8)


def run_yolo_segmentation_on_video(model, video_path, conf=0.5, display=True, save_path=None, transparency=0.5):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if save_path:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=conf)
        overlay = frame.copy()

        for result in results:
            vehiclePos = []
            if result.masks is not None:
                for mask, box in zip(result.masks.xy, result.boxes):
                    cls_id = int(box.cls[0])
                    if cls_id in vehicle_classes and cls_id < len(colors):
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                        vehiclePos.append([x_min, y_min, x_max, y_max])
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colors[cls_id], 2)

                    if cls_id < len(colors):  # Ensure the class ID is within the defined range
                        points = np.int32(mask).reshape((-1, 1, 2))
                        # Clamp points within image boundaries
                        points[:, :, 0] = np.clip(points[:, :, 0], 0, frame_width - 1)
                        points[:, :, 1] = np.clip(points[:, :, 1], 0, frame_height - 1)

                        # Check if points array is not empty before filling the polygon
                        if points.size > 0:
                            try:
                                cv2.fillPoly(overlay, [points], colors[cls_id])
                            except cv2.error as e:
                                print(f"Failed to draw polygon for class {cls_id}: {e}")

        overlayS = safe_area(overlay.copy(), vehiclePos)
        frameS = cv2.addWeighted(overlayS, transparency, frame, 1 - transparency, 0)
        frame = cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0)
        if save_path:
            out.write(frame)
        frameS = cv2.resize(frameS, (1080, 720))
        frame = cv2.resize(frame, (1080, 720))

        if display:
            cv2.imshow("YOLO Segmentation frameS", frameS)
            cv2.imshow("YOLO Segmentation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if save_path:
        out.release()
    cv2.destroyAllWindows()


for model_number in range(18, 19):
    model_path = f"F:\RunningProjects\YOLO_Model\Training\\runs\segment\RoadSegmentationForMyDataset3\weights\\best.pt"
    model = YOLO(model_path)
    cv2.namedWindow("YOLO Segmentation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("YOLO Segmentation frameS", cv2.WINDOW_NORMAL)

    for i in range(1, 22):
        run_yolo_segmentation_on_video(model=model, conf=0.65, display=True, transparency=0.5,
                                       video_path=f"F:/RunningProjects/LaneLinesDetection/InputVideo/video{i}.mp4")

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO

# Define class IDs and corresponding colors
vehicle_classes = [1, 2, 3, 5, 7, 24, 16, 0]
colors = [(0, 255, 0), (128, 64, 128), (255, 255, 255), (255, 255, 255),
          (255, 255, 0), (128, 128, 128), (34, 139, 34), (255, 140, 0)]
wantObjectMasks = False
img_size = [854, 480]
# Initialize Kalman filter for object tracking
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
kf.P *= 1000  # Initial uncertainty
kf.R *= 5  # Measurement noise

# Initialize Kalman filter for OpenCV tracking
kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
kalman.errorCovPost = np.eye(4, dtype=np.float32)

previous_masks = None  # Store previous segmentation masks for smoothing


def apply_morphology(mask):
    """ Apply morphological operations to clean the segmentation mask """
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def stabilize_objects(boxes):
    """ Apply Kalman filter to stabilize bounding boxes """
    global kalman
    stabilized_boxes = []
    for box in boxes:
        measurement = np.array([[np.float32(box[0])], [np.float32(box[1])]])
        predicted = kalman.predict()
        corrected = kalman.correct(measurement)
        if corrected is None:
            corrected = measurement  # Use original value if Kalman fails
        stabilized_boxes.append((corrected[0][0], corrected[1][0], box[2], box[3]))
    return stabilized_boxes


def run_yolo_segmentation_on_video(models, video_path, conf=0.5, display=False, save_path=None, transparency=0.5,
                                   input_size=None):
    """ Run YOLO segmentation and object detection on video """
    global previous_masks
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if save_path:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame = cv2.resize(frame, (img_size[0], img_size[1]))
        combined_overlay = np.zeros_like(frame, dtype=np.uint8)
        boxes = []

        # Process first model (road segmentation)
        results = models[0].predict(frame, conf=conf, imgsz=(img_size[0], img_size[1]))
        for result in results:
            if result.masks is None:
                print("Warning: No road segmentation detected.")
                continue
            for mask in result.masks.xy:
                points = np.int32(mask).reshape((-1, 1, 2))
                cv2.fillPoly(combined_overlay, [points], colors[1])  # Color for road segmentation

        # Process other models (object detection)
        for model in models[1:]:
            results = model.predict(frame, conf=conf, imgsz=(img_size[0], img_size[1]))
            for result in results:
                if result.boxes is None:
                    print("Warning: No object detections in this frame.")
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in vehicle_classes:
                        boxes.append([int(v) for v in box.xyxy[0]])

        # Apply temporal smoothing to segmentation
        if previous_masks is not None:
            combined_overlay = cv2.addWeighted(previous_masks, 0.5, combined_overlay, 0.5, 0)
        previous_masks = combined_overlay.copy()

        # Apply morphological operations
        combined_overlay = apply_morphology(combined_overlay)

        # Stabilize object detection
        stabilized_boxes = stabilize_objects(boxes)
        for box in stabilized_boxes:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

        # Overlay segmentation on the frame
        frame = cv2.addWeighted(combined_overlay, transparency, frame, 1 - transparency, 0)

        if save_path:
            out.write(frame)
        if display:
            cv2.imshow("Stable YOLO Segmentation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if save_path:
        out.release()
    cv2.destroyAllWindows()


def run_model_thread(model_paths, video_path, conf=0.5, display=False, save_path=None, transparency=0.5,
                     input_size=None):
    """ Load models and run segmentation on video """
    models = [YOLO(model_path).to('cuda') for model_path in model_paths]
    return run_yolo_segmentation_on_video(models=models, video_path=video_path, conf=conf, display=display,
                                          save_path=save_path, transparency=transparency, input_size=input_size)


if __name__ == "__main__":
    model1_path = '../Model/RoadSeg/weights/best.pt'  # Road segmentation model
    model2_path = '../Model/YoloPreTrained/yolo11x-seg.pt'  # Object detection model

    run_model_thread(
        model_paths=[model1_path, model2_path], conf=0.75, display=True, transparency=0.5,
        video_path='D:\\downloadFiles\\front_3\\FullAllPossibleVideo.mp4', input_size=[854, 480]
    )

from ultralytics import YOLO
import numpy as np
import random
import cv2

colors = [
    (50, 50, 50),  # 0: road (dark gray)
    (50, 150, 50),  # 1: drivable fallback (muted green)
    (0, 0, 255),  # 2: motorcycle (blue)
    (135, 206, 235),  # 3: sky (light sky blue)
    (192, 192, 192),  # 4: curb (silver)
    (169, 169, 169),  # 5: building (dark gray)
    (34, 139, 34),  # 6: vegetation (forest green)
    (255, 69, 0),  # 7: obs-str-bar-fallback (red-orange)
    (255, 222, 173),  # 8: billboard (navajo white)
    (128, 128, 128),  # 9: auto rickshaw (gray)
    (255, 165, 0),  # 10: pole (orange)
    (255, 0, 0),  # 11: car (red)
    (139, 0, 0),  # 12: truck (dark red)
    (255, 255, 0),  # 13: person (yellow)
    (160, 82, 45),  # 14: animal (saddle brown)
    (173, 216, 230),  # 15: rider (light blue)
    (220, 20, 60),  # 16: vehicle fallback (crimson)
    (255, 20, 147),  # 17: non-drivable fallback (deep pink)
    (169, 169, 169),  # 18: wall (dark gray)
    (245, 245, 220),  # 19: fallback background (beige)
    (139, 69, 19),  # 20: fence (brown)
    (255, 255, 0),  # 21: traffic sign (yellow)
    (47, 79, 79),  # 22: guard rail (dark slate gray)
    (72, 61, 139),  # 23: pole group (dark slate blue)
    (255, 255, 0),  # 24: bus (yellow)
]

vehicle_classes = [11, 12, 24, 16]  # Example vehicle class IDs: car, truck, bus, vehicle fallback


def run_yolo_segmentation_on_video(model_path, video_path, conf=0.5, display=True, save_path=None, transparency=0.5):
    global colors
    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object if saving output
    if save_path:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict with the model
        results = model.predict(frame, conf=conf)

        # Create a copy of the frame to apply the transparent mask
        overlay = frame.copy()

        # Draw the segmentation masks and bounding boxes on the frame
        for result in results:
            if result.masks is not None:  # Check if masks exist
                for mask, box in zip(result.masks.xy, result.boxes):
                    cls_id = int(box.cls[0])

                    # Process only vehicle-related classes
                    if cls_id in vehicle_classes:
                        # Get the bounding box coordinates
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                        print(f"Vehicle detected: Top-Left ({x_min}, {y_min}), Bottom-Right ({x_max}, {y_max})")

                        # Draw the bounding box on the frame
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colors[cls_id], 2)

                        # Ensure points is a 2D array
                        try:
                            points = np.int32(mask).reshape((-1, 1, 2))

                            # Clamp points within image boundaries
                            points[:, :, 0] = np.clip(points[:, :, 0], 0, frame_width - 1)
                            points[:, :, 1] = np.clip(points[:, :, 1], 0, frame_height - 1)

                            color_number = cls_id
                            cv2.fillPoly(overlay, [points], colors[color_number])

                        except Exception as e:
                            print(f"Error processing mask: {e}")
                            continue

        # Blend the original frame and the overlay with transparency
        frame = cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0)

        # Display the frame with segmentation results
        if display:
            cv2.imshow("YOLO Segmentation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Write the frame to the output video file
        if save_path:
            out.write(frame)

    # Release video capture and writer objects
    cap.release()
    if save_path:
        out.release()
    cv2.destroyAllWindows()


# Example usage:
run_yolo_segmentation_on_video(
    model_path="/Training/runs1\segment\RoadSegmentation6\weights\\best.pt",
    video_path="F:\RunningProjects\LaneLinesDetection\InputVideo/video19.mp4",
    conf=0.75,
    display=True,
    save_path="F:\RunningProjects\YOLO_Model\Testing\segmented_outputX.mp4",
    transparency=0.5  # Adjust this value to make the mask more or less transparent
)

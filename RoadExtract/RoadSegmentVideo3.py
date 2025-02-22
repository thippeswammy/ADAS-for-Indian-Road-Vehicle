import cv2
import numpy as np
from ultralytics import YOLO

vehicle_classes = [11, 12, 24, 16]  # Example vehicle class IDs: car, truck, bus, vehicle fallback

colors = [
    (128, 64, 128),  # 0: road (purple-gray)
    (0, 255, 0),  # 1: drivable fallback (bright green)
    (255, 0, 0),  # 2: motorcycle (bright red)
    (135, 206, 235),  # 3: sky (light sky blue)
    (255, 255, 255),  # 4: curb (white)
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


def safe_area1(overlay, vehiclePos):
    for box in vehiclePos:
        blx, by, brx = box[0], box[3], box[2]
        for y in range(overlay.shape[0] - 1, -1, -1):
            for x in range(overlay.shape[1] - 1, -1, -1):
                if (blx <= x <= brx and y <= by) and (overlay[y][x] == (128, 64, 128)).all():
                    overlay[y][x] = (255,) * 3
    return overlay.astype(np.uint8)


def safe_area(overlay, vehiclePos):
    # Iterate over each bounding box in vehiclePos
    for box in vehiclePos:
        blx, by, brx = box[0], box[3], box[2]

        # Create a mask for pixels within the bounding box
        mask_x = (blx <= np.arange(overlay.shape[1])) & (np.arange(overlay.shape[1]) <= brx)
        mask_y = (np.arange(overlay.shape[0]) <= by)

        # Use broadcasting to apply the mask to the entire overlay
        mask = np.outer(mask_y, mask_x)

        # Apply the condition to change only the pixels that match the color (128, 64, 128)
        color_mask = (overlay == (128, 64, 128)).all(axis=-1) & mask

        # Change the color of these pixels to white
        overlay[color_mask] = (255, 255, 255)

    return overlay.astype(np.uint8)


def run_yolo_segmentation_on_video(model, video_path, conf=0.5, display=True, save_path=None,
                                   transparency=0.5):
    global colors
    # Load the YOLO model

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get all YOLO classes
    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

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
            vehiclePos = []
            if result.masks is not None:  # Check if masks exist
                for mask, box in zip(result.masks.xy, result.boxes):
                    cls_id = int(box.cls[0])
                    try:
                        if cls_id in vehicle_classes:
                            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                            vehiclePos.append([x_min, y_min, x_max, y_max])
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colors[cls_id], 2)

                        points = np.int32(mask).reshape((-1, 1, 2))

                        color_number = classes_ids.index(int(box.cls[0]))
                        cv2.fillPoly(overlay, [points], colors[color_number])

                    except Exception as e:
                        print(f"Error processing mask: {e}")
                        continue

        # Blend the original frame and the overlay with transparency
        frame = cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0)

        # Process safe area
        overlay = safe_area(overlay, vehiclePos)

        # Ensure overlay is in the correct format before displaying
        # overlay = np.clip(overlay, 0, 255)  # Ensure pixel values are within [0, 255]
        # overlay = overlay.astype(np.uint8)  # Convert to 8-bit unsigned integer

        # Display the frame with segmentation results
        if display:
            cv2.imshow("overlay", overlay)
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
model_path = r"F:\RunningProjects\YOLO_Model\Training\runs\segment\RoadSegmentationForMyDataset9\weights\best.pt"

model = YOLO(model_path)
cv2.namedWindow("YOLO Segmentation", cv2.WINDOW_NORMAL)
for i in range(1, 22):
    run_yolo_segmentation_on_video(model=model, conf=0.7, display=True, transparency=0.5,
                                   video_path=r"D:\downloadFiles\front_3\video27.mp4",
                                   # save_path="F:/RunningProjects/YOLO_Model/Testing/segmented_outputXs.mp4",
                                   )

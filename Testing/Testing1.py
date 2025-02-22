from ultralytics import YOLO
import cv2
import torch

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the YOLO model
model = YOLO("/Training/runs1\\detect\\PersonTrainN\\weights\\best.pt")

# Source can be a video file path or a webcam index (0 for default webcam)
source = 0  # Use the default webcam

# Perform prediction with streaming
results = model.predict(source, show=False, save=True, conf=0.5, stream=True, device=device)

# Iterate over the results
for result in results:
    # Get the annotated frame
    annotated_frame = result.plot()

    # Display the annotated frame using OpenCV
    cv2.imshow('YOLO Prediction', annotated_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the OpenCV window
cv2.destroyAllWindows()

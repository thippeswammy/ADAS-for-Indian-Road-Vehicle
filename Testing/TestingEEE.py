import cv2
import torch
from ultralytics import YOLO

# Ensure that a GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the model
model = YOLO("/Training/runs1\detect\\train\weights\\best.pt")

# Source video path
source = 0

# Perform prediction
results = model.predict(source, save=True, conf=0.5, stream=True)

# Iterate over the results
for result in results:
    # Use the plot method to get the annotated image
    annotated_frame = result.plot()
    x, y, _ = annotated_frame.shape
    annotated_frame = cv2.resize(annotated_frame, None, fx=0.4, fy=0.4)
    # Display the annotated frame using OpenCV
    # annotated_frame = cv2.resize(annotated_frame,None, fx=0.5, fy=0.5)
    cv2.imshow('Prediction', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

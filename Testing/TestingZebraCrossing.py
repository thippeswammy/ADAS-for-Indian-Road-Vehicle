# from ultralytics import YOLO
# import cv2
# import torch
#
# model = YOLO("F:\\RunningProjects\\YOLO_Model\\Testing\\bestZebraCrossing.pt")
# source = "F:\\RunningProjects\\LaneLinesDetection\\InputVideo\\video7.mp4"
# result = model.predict(source, show=True, save=True, conf=0.5, stream=True)
# print(result)
#
# '''
# !yolo task=detect mode=predict model=best.pt conf=0.25 source={dataset.location}/test/images save=True
# '''


from ultralytics import YOLO
import cv2
import torch

# Ensure that a GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the model
model = YOLO("/Training/runs1/detect/ZebraCrossingTrain/weights/best.pt")

# Source video path
source = "F:\\RunningProjects\\LaneLinesDetection\\InputVideo\\video19.mov"

# Perform prediction
results = model.predict(source, show=False, save=True, conf=0.5, stream=True)

# Iterate over the results
for result in results:
    # Use the plot method to get the annotated image
    annotated_frame = result.plot()

    # Display the annotated frame using OpenCV
    cv2.imshow('Prediction', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cv2.destroyAllWindows()

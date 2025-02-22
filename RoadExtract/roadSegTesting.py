import cv2
import numpy as np
import torch
from torchvision import transforms

# Load the pre-trained segmentation model
model = torch.load(r'F:\RunningProjects\YOLO_Model\Training\runs\segment\RoadSegmentationForMyDataset\weights\best.pt')  # Replace with your model file
model.eval()

for i in range(1,4):
    # Define video input and output
    video_input_path = f'D:\downloadFiles\\front_3\\video{i}.mop4'  # Path to the input video
    video_output_path = 'output_video.mp4'  # Path to save the segmented output

    # Load the video
    cap = cv2.VideoCapture(video_input_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object to save the output video
    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Define a transformation to convert frames to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Resize to match your model's input size
        transforms.ToTensor(),
    ])

    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess the frame for the model
        input_frame = transform(frame).unsqueeze(0)  # Add batch dimension

        # Send frame to model and get prediction
        with torch.no_grad():
            prediction = model(input_frame)

        # Post-process the prediction (convert to mask)
        prediction = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()

        # Resize the predicted mask to the original frame size
        prediction = cv2.resize(prediction, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        # Convert the mask to 3 channels (for visualizing with original frame)
        mask_rgb = cv2.cvtColor((prediction * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Blend the mask with the original frame
        blended_frame = cv2.addWeighted(frame, 0.7, mask_rgb, 0.3, 0)

        # Write the frame to the output video
        out.write(blended_frame)

        # Display the frame (optional)
        cv2.imshow('Segmented Video', blended_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

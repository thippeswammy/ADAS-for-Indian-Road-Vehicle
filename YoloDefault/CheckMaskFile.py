import cv2
import numpy as np
import matplotlib.pyplot as plt

# Label data (normalized coordinates) provided by you
label_data = [0.000000, 0.785256, 0.316707, 0.610043, 0.451923, 0.557692, 0.503005, 0.548077, 0.619591, 0.539530,
              0.658654, 0.529915, 0.710337, 0.534188, 0.999479, 0.567308, 0.999479, 0.999074, 0.000000, 0.999074
              ]

# Reshape the label data to pairs of (x, y) coordinates
polygon_points = np.array(label_data).reshape(-1, 2)

# Path to the image file
image_path = 'F:\RunningProjects\YOLO_Model\DataSet/road/train\images\image3.jpg'

# Load the image
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Convert normalized coordinates to pixel values
polygon_points[:, 0] *= width
polygon_points[:, 1] *= height
polygon_points = polygon_points.astype(np.int32)

# Create a blank mask with the same dimensions as the image
mask = np.zeros((height, width), dtype=np.uint8)

# Fill the polygon defined by the points in the mask
cv2.fillPoly(mask, [polygon_points], color=255)

# Create a color overlay (e.g., red) for the mask
colored_mask = np.zeros_like(image)
colored_mask[mask == 255] = [0, 0, 255]  # BGR format (Blue, Green, Red)

# Blend the original image with the colored mask
overlay = cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0)

# Display the original image and overlay
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Image with Mask Overlay")

plt.show()

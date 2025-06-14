import os
import random
from collections import Counter

from PIL import Image


def select_and_analyze_images(directory, selection_percentage=1):
    """
    Selects a percentage of images from the directory and prints resolutions.

    :param directory: Path to the directory containing images.
    :param selection_percentage: Percentage of images to select (default 1%).
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    total_images = len(image_files)

    if total_images == 0:
        print("No image files found in the directory.")
        return

    # Calculate the number of images to select (at least 1)
    num_to_select = max(1, total_images * selection_percentage // 100)

    # Randomly select the specified percentage of images
    selected_images = random.sample(image_files, num_to_select)

    # Dictionary to store resolutions
    resolutions = Counter()

    print(f"Selected {num_to_select} images from {total_images} total images.\n")

    # Analyze and print resolutions
    for img_name in selected_images:
        img_path = os.path.join(directory, img_name)
        try:
            with Image.open(img_path) as img:
                resolution = img.size  # (width, height)
                resolutions[resolution] += 1
        except Exception as e:
            print(f"Error reading image {img_name}: {e}")

    # Print the resolution details
    print("Resolutions of selected images:")
    for resolution, count in resolutions.items():
        print(f"Resolution {resolution}: {count} images")


if __name__ == "__main__":
    image_directory = r"I:\thippe\DatasetAnnotation\complected_2\images"
    select_and_analyze_images(image_directory, selection_percentage=1)

'''
E:\SoftwareInstalls\PythonCuda\Scripts\python.exe F:\RunningProjects\YOLO_Model\RoadExtract\select_image_resolutions.py 
Selected 536 images from 53611 total images.

Resolutions of selected images:
Resolution (1920, 1080): 526 images
Resolution (2520, 1080): 10 images

Process finished with exit code 0

'''

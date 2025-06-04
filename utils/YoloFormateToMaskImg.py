import os

import cv2
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar


def parse_polygon_file(annotation_path, img_width, img_height):
    polygons = []
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            polygon_coords = [(int(coords[i] * img_width), int(coords[i + 1] * img_height)) for i in
                              range(0, len(coords), 2)]
            polygons.append(polygon_coords)
    return polygons


def create_mask_image(img_width, img_height, polygons):
    # Create a blank black mask image
    mask_image = np.zeros((img_height, img_width), dtype=np.uint8)

    # Draw each polygon on the mask image
    for polygon in polygons:
        # Convert to OpenCV format
        polygon = np.array(polygon, dtype=np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        # Draw the filled polygon
        cv2.fillPoly(mask_image, [polygon], color=255)

    return mask_image


def main():
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all image filenames
    image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    # Initialize tqdm progress bar
    for image_filename in tqdm(image_filenames, desc="Processing images"):
        # Extract the base filename without extension
        base_filename = os.path.splitext(image_filename)[0]

        # Paths to the image and corresponding annotation file
        image_path = os.path.join(images_dir, image_filename)
        annotation_path = os.path.join(labels_dir, f'{base_filename}.txt')

        # Check if the annotation file exists
        if os.path.exists(annotation_path):
            # Load the original image to get its dimensions
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading image: {image_path}")
                continue

            img_height, img_width = img.shape[:2]

            # Parse the polygon file
            polygons = parse_polygon_file(annotation_path, img_width, img_height)

            # Create the mask image
            mask_image = create_mask_image(img_width, img_height, polygons)

            # Convert the mask to a 3-channel image to overlay on the original image
            mask_colored = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

            # Set the color for the mask overlay (e.g., red)
            mask_colored[np.where((mask_colored == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

            # Blend the mask with the original image (alpha=0.5 for transparency)
            overlay_image = cv2.addWeighted(img, 1, mask_colored, 0.5, 0)

            # Save the overlayed image
            output_image_path = os.path.join(output_dir, f'{base_filename}_overlay.jpg')
            cv2.imwrite(output_image_path, overlay_image)

    print(f"Finished processing images.")


images_dir = r'../DatasetCreator/YoloDatasetProcessor/dataset_saving_working_dir/road/train/images'
labels_dir = r'../DatasetCreator/YoloDatasetProcessor/dataset_saving_working_dir/road/train/labels'
output_dir = r'../DatasetCreator/YoloDatasetProcessor/dataset_saving_working_dir/road/train/masks'

if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np
from tqdm import tqdm
import sys


def parse_polygon_file(annotation_path, img_width, img_height):
    polygons = []
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            polygon_coords = [(int(coords[i] * img_width), int(coords[i + 1] * img_height)) for i in range(0, len(coords), 2)]
            polygons.append(polygon_coords)
    return polygons


def create_mask_image(img_width, img_height, polygons):
    mask_image = np.zeros((img_height, img_width), dtype=np.uint8)
    for polygon in polygons:
        polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask_image, [polygon], color=255)
    return mask_image


def process_single_image(image_path, labels_dir, output_dir):
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    annotation_path = os.path.join(labels_dir, f'{base_filename}.txt')

    if not os.path.exists(annotation_path):
        print(f"Annotation file not found for image: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    img_height, img_width = img.shape[:2]
    polygons = parse_polygon_file(annotation_path, img_width, img_height)
    mask_image = create_mask_image(img_width, img_height, polygons)
    mask_colored = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    mask_colored[np.where((mask_colored == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
    overlay_image = cv2.addWeighted(img, 1, mask_colored, 0.5, 0)

    output_image_path = os.path.join(output_dir, f'{base_filename}_overlay.jpg')
    cv2.imwrite(output_image_path, overlay_image)


def main(input_path, labels_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.isdir(input_path):
        image_filenames = [f for f in os.listdir(input_path) if f.endswith('.jpg')]
        for image_filename in tqdm(image_filenames, desc="Processing images"):
            image_path = os.path.join(input_path, image_filename)
            process_single_image(image_path, labels_dir, output_dir)
    elif os.path.isfile(input_path):
        process_single_image(input_path, labels_dir, output_dir)
    else:
        print(f"Invalid input path: {input_path}")


# Example usage:
# You can change these paths as needed
input_path = r'../DatasetCreator/YoloDatasetProcessor/dataset_saving_working_dir/road/train/images'
labels_dir = r'../DatasetCreator/YoloDatasetProcessor/dataset_saving_working_dir/road/train/labels'
output_dir = r'../DatasetCreator/YoloDatasetProcessor/dataset_saving_working_dir/road/train/masks'

if __name__ == "__main__":
    main(input_path, labels_dir, output_dir)

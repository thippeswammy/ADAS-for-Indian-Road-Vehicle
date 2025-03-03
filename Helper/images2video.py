import os

import cv2
from tqdm import tqdm


def process_images(image_paths):
    """Reads and processes a batch of images."""
    processed_images = []
    for image_path in image_paths:
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            # Process the image (e.g., resize)
            processed_image = cv2.resize(image, (1920, 1080))
            processed_images.append(processed_image)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    return processed_images


def create_video(image_paths, output_path, fps=30, batch_size=10):
    """Creates a video from a list of image paths."""
    # Check if image paths are valid
    valid_images = [path for path in image_paths if os.path.exists(path)]
    if not valid_images:
        print("No valid images found.")
        return

    # Get the first image to set video dimensions
    first_image = cv2.imread(valid_images[0])
    height, width, layers = first_image.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process images in batches (single-threaded)
    batches = [valid_images[i:i + batch_size] for i in range(0, len(valid_images), batch_size)]
    for batch in tqdm(batches, total=len(batches), desc="Processing images", unit="batch"):
        processed_images = process_images(batch)
        for processed_image in processed_images:
            if processed_image is not None:
                video.write(processed_image)

    video.release()
    print(f"Video created successfully: {output_path}")


def main():
    # Set input and output directories
    input_dir = r"D:\downloadFiles\front_3\MaskImages"
    output_video_path = r"maskVideoDataset1.mp4"

    # Collect all image paths
    image_paths = [os.path.join(input_dir, img) for img in os.listdir(input_dir) if
                   img.endswith(('.png', '.jpg', '.jpeg'))]

    # Create a video from images
    create_video(image_paths, output_video_path, batch_size=1)  # Adjust batch size as needed


if __name__ == "__main__":
    main()

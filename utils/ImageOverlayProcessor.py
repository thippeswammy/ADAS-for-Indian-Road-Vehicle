import os
from concurrent.futures import ThreadPoolExecutor

import cv2
from tqdm import tqdm


class ImageOverlayProcessor:
    def __init__(self, original_folder='', mask_folder='', output_folder='', all_consider='', image_count=0):
        self.original_folder = original_folder
        self.mask_folder = mask_folder
        self.output_folder = output_folder
        self.all_consider = all_consider
        self.image_count = image_count
        self.valid_extensions = ('.png', '.jpg', '.jpeg')

        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

        # Filter and initialize the list of images to process
        self.original_images = self._filter_original_images()

    def _filter_original_images(self):
        """Filter images based on the provided criteria."""
        all_images = sorted(
            [img for img in os.listdir(self.original_folder) if img.lower().endswith(self.valid_extensions)]
        )
        if not self.all_consider:
            return all_images

        filtered_images = []
        count = 0
        for img in all_images:
            if img.split('_')[0] == self.all_consider:
                if count >= self.image_count:
                    filtered_images.append(img)
                count += 1
        return filtered_images

    def load_image_and_mask(self, image_name):
        """Load the original image and its corresponding mask."""
        original_image_path = os.path.join(self.original_folder, image_name)
        mask_image_name = os.path.splitext(image_name)[0] + '.png'
        mask_image_path = os.path.join(self.mask_folder, mask_image_name)

        # Check if the mask exists
        if not os.path.exists(mask_image_path):
            return None, None

        # Load images
        original_image = cv2.imread(original_image_path)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        return original_image, mask_image

    def overlay_mask_on_image(self, original_image, mask_image):
        """Overlay the mask on the original image."""
        if len(mask_image.shape) == 2:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        # Blend images

        overlay_image = cv2.addWeighted(original_image, 0.5, mask_image, 0.5, 0)
        return overlay_image

    def process_image(self, img_name):
        """Process a single image."""
        original_image, mask_image = self.load_image_and_mask(img_name)
        if original_image is not None and mask_image is not None:
            combined_image = self.overlay_mask_on_image(original_image, mask_image)
            output_image_path = os.path.join(self.output_folder, img_name)
            cv2.imwrite(output_image_path, combined_image)

    def process_all_images(self):
        """Process all selected images using multithreading."""
        with tqdm(total=len(self.original_images), desc="Processing Images") as pbar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(self.process_image, img_name): img_name for img_name in self.original_images
                }
                for future in futures:
                    future.result()  # Raise exceptions if any
                    pbar.update(1)


if __name__ == "__main__":
    # Configuration
    original_folder = r'./videos/Images'
    mask_folder = r'./videos/outputs'
    output_folder = r'./videos/overlappedImages'
    # Initialize and run the processor
    processor = ImageOverlayProcessor(original_folder, mask_folder, output_folder,
                                      all_consider='', image_count=0)
    processor.process_all_images()

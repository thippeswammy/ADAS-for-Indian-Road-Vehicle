import concurrent.futures
import os

import cv2
from tqdm import tqdm


class images_resize:
    def __init__(self, original_folder, output_original_folder):
        self.original_folder = original_folder
        self.output_original_folder = output_original_folder
        self.valid_extensions = ('.png', '.jpg', '.jpeg')

    @staticmethod
    def copy_image(src, dst):
        """Copy a single image from src to dst, handling renaming if dst already exists."""
        if os.path.exists(dst):
            base, ext = os.path.splitext(os.path.basename(src))
            counter = 1
            while os.path.exists(dst):
                new_filename = f"{base}_{counter}{ext}"
                dst = os.path.join(os.path.dirname(dst), new_filename)
                counter += 1
        # shutil.copy2(src, dst)
        img = cv2.imread(src)
        # print("size = ", img.shape[1] // 2, img.shape[0] // 2)
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(dst, img)

    def copy_images(self):
        """Copy original and mask images to their respective output folders."""
        os.makedirs(self.output_original_folder, exist_ok=True)
        os.makedirs(self.original_folder, exist_ok=True)

        original_images = [os.path.join(self.original_folder, filename) for filename in os.listdir(self.original_folder)
                           if filename.endswith(self.valid_extensions)]

        original_images_to_copy = original_images
        # Copy original images with progress bar using threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda img: self.copy_image(img, os.path.join(self.output_original_folder,
                                                                                 os.path.basename(img))),
                                   original_images_to_copy),
                      total=len(original_images_to_copy), desc='Copying Original Images'))

    def run(self):
        """Main entry point to execute the copying process."""
        self.copy_images()


if __name__ == "__main__":
    # Configuration
    original_images_folder = r'img_folder1'
    output_original_folder = r'img_folder2'

    # Create an instance of ImageCopier and run it
    copier = images_resize(original_images_folder, output_original_folder)
    copier.run()

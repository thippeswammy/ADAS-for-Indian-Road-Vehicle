import concurrent.futures
import os
import shutil

from tqdm import tqdm


class ImageCopier:
    def __init__(self, original_folder, mask_folder, overlap_images_folder, output_original_folder, output_mask_folder):
        self.original_folder = original_folder
        self.mask_folder = mask_folder
        self.overlap_images_folder = overlap_images_folder
        self.output_original_folder = output_original_folder
        self.output_mask_folder = output_mask_folder
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
        shutil.copy2(src, dst)

    def _get_overlap_filenames(self):
        """Retrieve filenames (without extensions) in the overlap folder."""
        return {os.path.splitext(filename)[0].lower() for filename in os.listdir(self.overlap_images_folder)
                if filename.lower().endswith(self.valid_extensions)}

    def _filter_images_to_copy(self, images):
        """Filter images to copy based on overlap filenames."""
        overlap_filenames = self._get_overlap_filenames()
        return [img for img in images if os.path.splitext(os.path.basename(img))[0].lower() in overlap_filenames]

    def copy_images(self):
        """Copy original and mask images to their respective output folders."""
        os.makedirs(self.output_original_folder, exist_ok=True)
        os.makedirs(self.output_mask_folder, exist_ok=True)

        original_images = [os.path.join(self.original_folder, filename) for filename in os.listdir(self.original_folder)
                           if filename.endswith(self.valid_extensions)]
        mask_images = [os.path.join(self.mask_folder, filename) for filename in os.listdir(self.mask_folder)
                       if filename.endswith(self.valid_extensions)]

        original_images_to_copy = self._filter_images_to_copy(original_images)
        mask_images_to_copy = self._filter_images_to_copy(mask_images)

        # Copy original images with progress bar using threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda img: self.copy_image(img, os.path.join(self.output_original_folder,
                                                                                 os.path.basename(img))),
                                   original_images_to_copy),
                      total=len(original_images_to_copy), desc='Copying Original Images'))

        # Copy mask images with progress bar using threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda img: self.copy_image(img, os.path.join(self.output_mask_folder,
                                                                                 os.path.basename(img))),
                                   mask_images_to_copy),
                      total=len(mask_images_to_copy), desc='Copying Mask Images'))

    def run(self):
        """Main entry point to execute the copying process."""
        self.copy_images()


if __name__ == "__main__":
    # Configuration
    original_images_folder = r'./Images'
    mask_images_folder = r'./outputs'
    overlap_images_folder = './overlappedImages'
    output_original_folder = r'./verified/TempImg'
    output_mask_folder = r'./verified/TempMasks'

    # Create an instance of ImageCopier and run it
    copier = ImageCopier(original_images_folder, mask_images_folder, overlap_images_folder,
                         output_original_folder, output_mask_folder)
    copier.run()

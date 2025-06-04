import logging
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
from tqdm import tqdm

import create_yolo_folders

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImageAugmentations:
    """Class to apply various image augmentations."""

    @staticmethod
    def apply_gaussian_blur(image, size=random.choice([3, 5])):
        """Apply Gaussian blur using a kernel for three-channel images."""
        kernel = torch.ones((3, 1, size, size), dtype=torch.float32).to(device) / (size * size)
        image = image.unsqueeze(0)  # Add batch dimension
        return torch.nn.functional.conv2d(image, kernel, padding=size // 2, groups=3).squeeze(0)

    @staticmethod
    def apply_average_blur(image, size=random.choice([3, 5])):
        """Apply average blur for three-channel images."""
        kernel = torch.ones((3, 1, size, size), dtype=torch.float32).to(device) / (size * size)
        image = image.unsqueeze(0)  # Add batch dimension
        return torch.nn.functional.conv2d(image, padding=size // 2, groups=3).squeeze(0)

    @staticmethod
    def add_gaussian_noise(image, mean=0.5, sigma=0.01):
        """Add Gaussian noise to the image."""
        noise = torch.randn(image.size()).to(device) * sigma + mean
        return (image + noise).clamp(0, 1)

    @staticmethod
    def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
        """Add salt and pepper noise to the image."""
        noisy_img = image.clone()
        num_salt = int(salt_prob * image.numel())
        num_pepper = int(pepper_prob * image.numel())
        salt_coords = [torch.randint(0, dim, (num_salt,)).to(device) for dim in image.shape]
        pepper_coords = [torch.randint(0, dim, (num_pepper,)).to(device) for dim in image.shape]
        noisy_img[salt_coords] = 1  # Salt
        noisy_img[pepper_coords] = 0  # Pepper
        return noisy_img


class YoloProcessor:
    """Class to handle YOLO data processing, including augmentation and label management."""

    def __init__(self, config):
        try:
            self.full_path, _ = create_yolo_folders.create_yolo_folder_structure(
                folder_name=config['folder_name'],
                main_path=config['dataset_saving_working_dir'],
                num_classes=config['class_names']
            )
        except Exception as e:
            logging.error(f"Error creating YOLO folder structure: {e}")
            raise

        self.train_image_count = self.val_image_count = self.test_image_count = 0
        self.train_save_path = os.path.join(self.full_path, 'train')
        self.val_save_path = os.path.join(self.full_path, 'valid')
        self.test_save_path = os.path.join(self.full_path, 'test')
        self.mask_folder_name = config['mask_folder_name']
        self.original_folder_name = config['original_folder_name']
        self.augmenter = ImageAugmentations()
        self.color_to_label = config['color_to_label']
        self.mask_type_ext = config['mask_type_ext']
        self.class_names = config['class_names']
        self.class_to_id = config['class_to_id']
        self.train_split = config['train_split']
        self.source_dir_original_img = os.path.join(config['dataset_path'], self.original_folder_name)
        self.source_dir_mask_img = os.path.join(config['dataset_path'], self.mask_folder_name)
        self.test_split = config['test_split']
        self.val_split = config['val_split']
        self.fact_times = config['augment_times']
        self.num_threads = config['num_threads']
        self.keep_val_original = config['Keep_val_dataset_original']

        if not os.path.exists(self.source_dir_original_img) or not os.path.exists(self.source_dir_mask_img):
            logging.error(f"Source directories not found: {self.source_dir_original_img} or {self.source_dir_mask_img}")
            raise FileNotFoundError(
                f"Check paths in config: {self.source_dir_original_img}, {self.source_dir_mask_img}")

    def distribute_files_with_threads(self):
        """Distribute files into training, validation, and test sets using multithreading."""
        image_paths = self.collect_image_paths(self.source_dir_original_img)
        if not image_paths:
            logging.error("No image files found in the source directory.")
            return

        total_files = len(image_paths) * self.fact_times
        self.test_image_count = int(total_files * self.test_split)
        self.val_image_count = int(total_files * self.val_split)
        self.train_image_count = total_files - self.test_image_count - self.val_image_count
        logging.info(f"Split: Train={self.train_image_count}, Val={self.val_image_count}, Test={self.test_image_count}")

        file_infos = [(os.path.basename(file_path)[:-4], file_path) for file_path in image_paths]
        random.shuffle(file_infos)
        # with tqdm(total=total_files, desc="Processing Images") as pbar:
        #     with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
        #         futures = [executor.submit(self.process_single_file, file_info, self.fact_times) for file_info in file_infos]
        #         for future in as_completed(futures):
        #             try:
        #                 future.result()
        #             except Exception as e:
        #                 logging.error(f"Processing error: {e}")
        #             pbar.update(self.fact_times)

        # Process files sequentially instead of using ThreadPoolExecutor
        with tqdm(total=total_files, desc="Processing Images") as pbar:
            for file_info in file_infos:
                try:
                    self.process_single_file(file_info, self.fact_times)
                except Exception as e:
                    logging.error(f"Processing error: {e}")
                pbar.update(self.fact_times)

    def collect_image_paths(self, directory):
        """Collect all image file paths from the given directory."""
        image_paths = [os.path.join(root, f) for root, _, files in os.walk(directory)
                       for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        logging.info(f"Found {len(image_paths)} images in {directory}.")
        return image_paths

    def process_single_file(self, file_info, times):
        """Process a single file with augmentation."""
        file_basename, file_path = file_info
        try:
            self.process_and_save(file_basename, file_path, times)
        except Exception as e:
            logging.error(f"Error processing {file_basename}: {e}")

    def get_label_path(self, image_source_path):
        """Construct the path for the label file based on the image path."""
        return image_source_path.replace(self.original_folder_name, self.mask_folder_name).replace(
            '.jpg', self.mask_type_ext).replace('.png', self.mask_type_ext)

    def process_and_save(self, file_name, image_source_path, times):
        """Process and save images and their corresponding labels."""
        if not os.path.exists(image_source_path):
            logging.warning(f"Image not found: {image_source_path}")
            return

        label_source_path = self.get_label_path(image_source_path)
        if not os.path.exists(label_source_path):
            logging.warning(f"Label not found for {image_source_path}")
            return

        yolo_polygons = self.process_mask_to_yolo_txt(label_source_path, self.class_to_id)
        for i in range(1, times + 1):
            save_img_path, save_label_path = self.get_destination_paths(file_name, i)
            if save_img_path and save_label_path:
                augmented_img, transformed_polygons = self.apply_augmentations(image_source_path, label_source_path, i)
                if augmented_img is not None:
                    try:
                        if isinstance(augmented_img, torch.Tensor):
                            image_np = (augmented_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        else:
                            image_np = np.array(augmented_img)
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        self.save_yolo_format(save_label_path, transformed_polygons)
                        cv2.imwrite(save_img_path, image_np)
                    except Exception as e:
                        logging.error(f"Error saving {save_img_path}: {e}")

    def get_destination_paths(self, file_name, num):
        """Get the destination paths for saving images and labels."""
        save_img_path, save_label_path = '', ''
        choices = [i for i, count in enumerate([self.train_image_count, self.val_image_count, self.test_image_count])
                   if count > 0]
        choice = 1 if self.keep_val_original and num == 1 else random.choice(choices)

        if choice == 0 and self.train_image_count > 0:
            save_img_path = os.path.join(self.train_save_path, 'images', f'{file_name}_{num}.jpg')
            save_label_path = os.path.join(self.train_save_path, 'labels', f'{file_name}_{num}.txt')
            self.train_image_count -= 1
        elif choice == 1 and self.val_image_count > 0:
            save_img_path = os.path.join(self.val_save_path, 'images', f'{file_name}_{num}.jpg')
            save_label_path = os.path.join(self.val_save_path, 'labels', f'{file_name}_{num}.txt')
            self.val_image_count -= 1
        elif choice == 2 and self.test_image_count > 0:
            save_img_path = os.path.join(self.test_save_path, 'images', f'{file_name}_{num}.jpg')
            save_label_path = os.path.join(self.test_save_path, 'labels', f'{file_name}_{num}.txt')
            self.test_image_count -= 1
        return save_img_path, save_label_path

    def apply_augmentations(self, img_path, label_path, num):
        """Apply augmentations to image and transform corresponding polygons."""
        try:
            img = cv2.imread(img_path)
            polygons = self.process_mask_to_yolo_txt(label_path, self.class_to_id)
            poly_coords = [p[1] for p in polygons]  # List of [(x1,y1), (x2,y2), ...]

            if self.keep_val_original and num == 1:
                return img, polygons

            # Define augmentation pipeline with geometric transforms
            aug = A.Compose([
                A.ColorJitter(brightness=(0.6, 0.9), contrast=(0.6, 0.9), p=0.5),
                A.Resize(640, 640, p=1.0),  # Match YOLO imgsz
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.01),
                # A.equalize(img),
                # A.HueSaturationValue(img),
                # A.MotionBlur(img),
                # A.RandomRain(img),
                # A.RandomScale(img),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

            augmented = aug(image=img, keypoints=poly_coords)
            img_aug = augmented['image']
            new_poly_coords = augmented['keypoints']

            # Convert back to YOLO format
            new_polygons = [(p[0], coords) for p, coords in zip(polygons, new_poly_coords)]
            img_tensor = torch.from_numpy(img_aug.transpose(2, 0, 1) / 255.0).to(device)
            if num < 6:
                return img_tensor, new_polygons
            if num % 6 == 0:
                img_tensor = self.augmenter.apply_gaussian_blur(img_tensor)
            elif num % 6 == 1:
                img_tensor = self.augmenter.apply_average_blur(img_tensor)
            elif num % 6 == 2:
                img_tensor = self.augmenter.add_gaussian_noise(img_tensor)
            elif num % 6 == 3:
                img_tensor = self.augmenter.add_salt_pepper_noise(img_tensor)
            return img_tensor, new_polygons
        except Exception as e:
            logging.error(f"Augmentation error for {img_path}: {e}")
            return None, None

    def process_mask_to_yolo_txt(self, mask_file_path, class_map):
        """Convert the mask file to YOLO format."""
        mask_image = cv2.imread(mask_file_path)
        if mask_image is None:
            logging.error(f"Failed to load mask: {mask_file_path}")
            return []
        mask_image = cv2.resize(mask_image, (640, 640))  # Match training imgsz
        image_height, image_width = mask_image.shape[:2]
        polygons = self.get_polygons(mask_image)
        return self.convert_polygons_to_yolo(image_width, image_height, polygons)

    def get_polygons(self, mask_image):
        """Extract polygons for each label in the mask image."""
        polygons = []
        for color, label in self.color_to_label.items():
            mask = np.all(mask_image == color, axis=-1).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Minimum area threshold
                    polygons.append((label, contour.reshape(-1, 2)))
        return polygons

    def convert_polygons_to_yolo(self, img_width, img_height, polygons):
        """Convert polygon coordinates to YOLO format."""
        yolo_polygons = []
        for label, polygon in polygons:
            normalized_polygon = [(x / img_width, y / img_height) for x, y in polygon]
            yolo_polygons.append((label, normalized_polygon))
        return yolo_polygons

    def save_yolo_format(self, save_label_path, yolo_polygons):
        """Save the YOLO formatted text to the specified path."""
        try:
            with open(save_label_path, 'w') as f:  # Use 'w' to overwrite
                for label, polygon in yolo_polygons:
                    if polygon:  # Ensure polygon is not empty
                        polygon_str = ' '.join(f"{x} {y}" for x, y in polygon)
                        f.write(f"{label} {polygon_str}\n")
        except Exception as e:
            logging.error(f"Error saving label {save_label_path}: {e}")


if __name__ == '__main__':
    CONFIG = {
        "dataset_path": "D:\downloadFiles\\front_3\Dataset\\road",
        "mask_folder_name": "MaskImages",  # Change if different
        "original_folder_name": "OriginImages",  # Change if different
        "dataset_saving_working_dir": 'dataset_saving_working_dir',
        "augment_times": 10,  # Number of augmentations per image
        "test_split": 0.0,  # Percentage of data for testing
        "val_split": 0.1,  # Percentage of data for validation
        "train_split": 0.9,  # Percentage of data for training
        "Keep_val_dataset_original": True,  # for keeping the original dataset has original
        "num_threads": os.cpu_count() - 2,  # Number of threads for parallel processing
        "class_to_id": {
            'road': 0,
        },
        "color_to_label": {
            (255, 255, 255): 0,
        },
        "folder_name": 'road',
        "class_names": ['road'],
        "mask_type_ext": '.png',
        "FromDataType": '',
        "ToDataTypeFormate": '',
    }

    try:
        processor = YoloProcessor(config=CONFIG)
        processor.distribute_files_with_threads()
    except Exception as e:
        logging.error(f"Critical error: {e}")

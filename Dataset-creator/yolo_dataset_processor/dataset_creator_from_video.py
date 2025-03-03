import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
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
        return torch.nn.functional.conv2d(image, kernel, padding=size // 2, groups=3).squeeze(0)

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

    def __init__(self, original_video_path, mask_video_path, train_split=0.8, val_split=0.15, test_split=0.05,
                 factTimes=7,
                 class_to_id=None, color_to_label='', dataset_saving_working_dir='', folder_name='', class_names='',
                 mask_type_ext='.jpeg', mask_foulder_name='masks', FromDataType='', ToDataTypeFormate=''):
        try:
            fullPath, dataset_folder = create_yolo_folders.create_yolo_folder_structure(folder_name=folder_name,
                                                                                        main_path=dataset_saving_working_dir,
                                                                                        num_classes=class_names)
            print(f"Dataset folder structure created at: {fullPath}")
        except Exception as e:
            logging.error(f"Error creating YOLO folder structure: {e}")
            raise

        self.train_image_count = self.val_image_count = self.test_image_count = 0
        self.train_save_path = os.path.join(fullPath, 'train')
        self.val_save_path = os.path.join(fullPath, 'valid')
        self.test_save_path = os.path.join(fullPath, 'test')
        self.mask_foulder_name = mask_foulder_name
        self.ToDataTypeFormate = ToDataTypeFormate
        self.augmenter = ImageAugmentations()
        self.color_to_label = color_to_label
        self.mask_type_ext = mask_type_ext
        self.FromDataType = FromDataType
        self.class_names = class_names
        self.class_to_id = class_to_id
        self.train_split = train_split
        self.original_video_path = original_video_path
        self.mask_video_path = mask_video_path
        self.test_split = test_split
        self.val_split = val_split
        self.main_path = dataset_saving_working_dir
        self.factTimes = factTimes

        if not os.path.exists(self.original_video_path):
            logging.error(f"Original video file '{self.original_video_path}' does not exist.")
            raise FileNotFoundError(f"Original video file '{self.original_video_path}' not found.")
        if not os.path.exists(self.mask_video_path):
            logging.error(f"Mask video file '{self.mask_video_path}' does not exist.")
            raise FileNotFoundError(f"Mask video file '{self.mask_video_path}' not found.")

    def distribute_files_with_threads(self):
        """Distribute frames into training, validation, and test sets using multithreading."""
        cap_original = cv2.VideoCapture(self.original_video_path)
        cap_mask = cv2.VideoCapture(self.mask_video_path)
        frame_count = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
        total_files = frame_count * self.factTimes

        if total_files / 10 > 10000:
            self.val_split = 10000 / total_files
            self.test_split = 1000 / total_files
        self.test_image_count = int(total_files * self.test_split)
        self.val_image_count = int(total_files * self.val_split)
        self.train_image_count = total_files - self.test_image_count - self.val_image_count

        with tqdm(total=total_files, desc="Processing Frames") as pbar:
            with ThreadPoolExecutor(max_workers=(os.cpu_count() - 8)) as executor:
                futures = []
                numbers = 1
                while cap_original.isOpened() and cap_mask.isOpened():
                    ret_orig, frame_orig = cap_original.read()
                    ret_mask, frame_mask = cap_mask.read()
                    if not ret_orig or not ret_mask:
                        break
                    futures.append(
                        executor.submit(self.process_single_frame, frame_orig, frame_mask, self.factTimes, numbers))
                    numbers = numbers + 1
                    pbar.update(self.factTimes)

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Exception during processing: {e}")

        cap_original.release()
        cap_mask.release()

    def get_label_path(self, save_label_path, num):
        """Construct the path for the label file based on the save path."""
        return save_label_path.replace(".jpg", f"_{num}.txt")

    def process_single_frame(self, frame_orig, frame_mask, Times, numbers):
        """Process and save frames and their corresponding masks."""
        for i in range(1, Times + 1):
            if i == 1:
                yolo_polygons_points_txt = self.process_mask_to_yolo_txt(frame_mask, self.class_to_id)
            save_img_path, save_label_path = self.get_destination_paths(numbers, i)
            if save_img_path and save_label_path:
                augmented_img, choice = self.apply_augmentations(frame_orig, i)
                if augmented_img is not None:
                    try:
                        image_np = np.array(augmented_img.permute(1, 2, 0).cpu().numpy() * 255, dtype=np.uint8)
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        self.save_yolo_format(self.get_label_path(save_label_path, i), yolo_polygons_points_txt)
                        cv2.imwrite(save_img_path, image_np)
                    except Exception as e:
                        logging.error(f"Error saving image {save_img_path}: {e}")

    def get_destination_paths(self, frame_number, number):
        """Get the destination paths for saving images and labels."""
        save_img_path, save_label_path = '', ''
        choices = []
        if self.train_image_count > 0:
            choices.append(0)
        elif self.val_image_count > 0:
            choices.append(1)
        elif self.test_image_count > 0:
            choices.append(2)
        choice = random.choice(choices)

        file_name = f"frame_{frame_number:05d}_{number}"
        if choice == 0:
            save_img_path = os.path.join(self.train_save_path, 'images', f'{file_name}.jpg')
            save_label_path = os.path.join(self.train_save_path, 'labels', f'{file_name}.txt')
            self.train_image_count -= 1
        elif choice == 1:
            save_img_path = os.path.join(self.val_save_path, 'images', f'{file_name}.jpg')
            save_label_path = os.path.join(self.val_save_path, 'labels', f'{file_name}.txt')
            self.val_image_count -= 1
        elif choice == 2:
            save_img_path = os.path.join(self.test_save_path, 'images', f'{file_name}.jpg')
            save_label_path = os.path.join(self.test_save_path, 'labels', f'{file_name}.txt')
            self.test_image_count -= 1
        return save_img_path, save_label_path

    def apply_augmentations(self, frame, num):
        """Apply augmentations using PyTorch and return augmented image."""
        try:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if num == 1:
                bright = [0.6, 0.9]
                contrast = [0.6, 0.8]
            elif num == 2:
                bright = [0.65, 1.1]
                contrast = [0.9, 1.1]
            elif num == 3:
                bright = [0.5, 0.9]
                contrast = [0.9, 1.1]
            elif num == 4:
                bright = [0.8, 1.1]
                contrast = [0.7, 0.8]
            else:
                bright = [0.7, 1.1]
                contrast = [0.8, 1.1]

            augmentations = T.Compose([
                T.ColorJitter(brightness=(bright[0], bright[1]), contrast=(contrast[0], contrast[1])),
                T.ToTensor(),
            ])
            img_tensor = augmentations(img).to(device)
            if num % 5 == 1:
                img_tensor = self.augmenter.apply_gaussian_blur(img_tensor)
            elif num % 5 == 2:
                img_tensor = self.augmenter.apply_average_blur(img_tensor)
            elif num % 5 == 3:
                img_tensor = self.augmenter.add_gaussian_noise(img_tensor, random.uniform(0, 0.5),
                                                               random.uniform(0.005, 0.04))
            elif num % 5 == 4:
                img_tensor = self.augmenter.add_salt_pepper_noise(img_tensor, random.uniform(0.005, 0.04),
                                                                  random.uniform(0.001, 0.05))
            return img_tensor, num % 5
        except Exception as e:
            logging.error(f"Error applying augmentations on frame: {e}")
            return None

    def process_mask_to_yolo_txt(self, mask_image, class_map):
        """Convert the mask image to YOLO format."""
        image_height, image_width = mask_image.shape[:2]
        polygons = self.get_polygons(mask_image)
        yolo_polygons_txt = self.convert_polygons_to_yolo(image_width, image_height, polygons)

        return yolo_polygons_txt

    def get_polygons(self, mask_image):
        """
        Extract polygons for each label in the mask image.
        """
        polygons = []
        for color, label in self.color_to_label.items():
            mask = np.all(mask_image == color, axis=-1).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 0:
                    polygons.append((label, contour.reshape(-1, 2)))
        return polygons

    def convert_polygons_to_yolo(self, img_width, img_height, polygons):
        """
        Convert polygon coordinates to YOLO format.
        """
        yolo_polygons = []
        for label, polygon in polygons:
            normalized_polygon = [(x / img_width, y / img_height) for (x, y) in polygon]
            yolo_polygons.append((label, normalized_polygon))
        return yolo_polygons

    def save_yolo_format(self, save_label_path, yolo_polygons):
        """Save the YOLO formatted text to the specified path."""
        try:
            with open(save_label_path, 'a') as f:
                for label, polygon in yolo_polygons:
                    polygon_str = ' '.join(f"{x} {y}" for x, y in polygon)
                    f.write(f"{label} {polygon_str}\n")
        except Exception as e:
            logging.error(f"Error saving YOLO label file {save_label_path}: {e}")


if __name__ == '__main__':
    class_to_id = {
        'road': 0,
    }
    color_to_id = {
        (255, 255, 255): 0,
    }
    original_video_path = r'../../Helper/originalVideoDataset1.mp4'
    mask_video_path = r'../../Helper/maskVideoDataset1.mp4'
    dataset_saving_working_dir = r'dataset_saving_working_dir'
    try:
        processor = YoloProcessor(original_video_path=original_video_path, mask_video_path=mask_video_path,
                                  class_to_id=class_to_id, color_to_label=color_to_id,
                                  dataset_saving_working_dir=dataset_saving_working_dir, factTimes=1,
                                  folder_name='road', class_names=list(class_to_id.keys()),
                                  mask_type_ext='.png', FromDataType='', ToDataTypeFormate='')
        processor.distribute_files_with_threads()
    except Exception as e:
        logging.error(f"Critical error: {e}")

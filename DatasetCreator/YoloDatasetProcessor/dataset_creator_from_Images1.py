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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImageAugmentations:
    @staticmethod
    def apply_gaussian_blur(image, size=random.choice([3, 5])):
        kernel = torch.ones((3, 1, size, size), dtype=torch.float32).to(device) / (size * size)
        image = image.unsqueeze(0)
        return torch.nn.functional.conv2d(image, kernel, padding=size // 2, groups=3).squeeze(0)

    @staticmethod
    def apply_average_blur(image, size=random.choice([3, 5])):
        kernel = torch.ones((3, 1, size, size), dtype=torch.float32).to(device) / (size * size)
        image = image.unsqueeze(0)
        return torch.nn.functional.conv2d(image, kernel, padding=size // 2, groups=3).squeeze(0)

    @staticmethod
    def add_gaussian_noise(image, mean=0.5, sigma=0.01):
        noise = torch.randn(image.size()).to(device) * sigma + mean
        return (image + noise).clamp(0, 1)

    @staticmethod
    def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
        noisy_img = image.clone()
        num_salt = int(salt_prob * image.numel())
        num_pepper = int(pepper_prob * image.numel())
        salt_coords = [torch.randint(0, dim, (num_salt,)).to(device) for dim in image.shape]
        pepper_coords = [torch.randint(0, dim, (num_pepper,)).to(device) for dim in image.shape]
        noisy_img[salt_coords] = 1
        noisy_img[pepper_coords] = 0
        return noisy_img


class YoloProcessor:
    def __init__(self, config):
        try:
            fullPath, dataset_folder = create_yolo_folders.create_yolo_folder_structure(
                folder_name=config['folder_name'],
                main_path=config['dataset_saving_working_dir'],
                num_classes=config['class_names']
            )
        except Exception as e:
            logging.error(f"Error creating YOLO folder structure: {e}")
            raise

        self.train_save_path = os.path.join(fullPath, 'train')
        self.val_save_path = os.path.join(fullPath, 'valid')
        self.test_save_path = os.path.join(fullPath, 'test')

        self.config = config
        self.augmenter = ImageAugmentations()

        self.original_folder = os.path.join(config['dataset_path'], config['original_folder_name'])
        self.mask_folder = os.path.join(config['dataset_path'], config['mask_folder_name'])

        self.train_split = config['train_split']
        self.val_split = config['val_split']
        self.test_split = config['test_split']

        self.augment_times = config['augment_times']
        self.num_threads = config['num_threads']

        self.class_to_id = config['class_to_id']
        self.color_to_label = config['color_to_label']
        self.mask_type_ext = config['mask_type_ext']

        if not os.path.exists(self.original_folder) or not os.path.exists(self.mask_folder):
            raise FileNotFoundError("Original or mask image folder does not exist.")

    def run(self):
        image_paths = self.collect_image_paths(self.original_folder)
        if not image_paths:
            logging.error("No image files found.")
            return

        random.shuffle(image_paths)
        total = len(image_paths)

        val_count = int(total * self.val_split)
        test_count = int(total * self.test_split)
        train_count = total - val_count - test_count

        train_files = image_paths[:train_count]
        val_files = image_paths[train_count:train_count + val_count]
        test_files = image_paths[train_count + val_count:]

        self.process_set(val_files, self.val_save_path, augment=False)
        self.process_set(test_files, self.test_save_path, augment=False)
        self.process_set(train_files, self.train_save_path, augment=True)

    def collect_image_paths(self, directory):
        image_paths = []
        for root, _, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, f))
        return image_paths

    def process_set(self, image_paths, save_root, augment=False):
        with tqdm(total=len(image_paths), desc=f"Processing {os.path.basename(save_root)}") as pbar:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(self.process_single_file, path, save_root, augment) for path in image_paths]
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        logging.error(f"Error: {e}")
                    pbar.update(1)

    def process_single_file(self, image_path, save_root, augment):
        file_basename = os.path.splitext(os.path.basename(image_path))[0]
        label_path = self.get_label_path(image_path)
        if not os.path.exists(label_path):
            logging.warning(f"Label not found: {label_path}")
            return

        yolo_data = self.process_mask_to_yolo_txt(label_path)

        times = self.augment_times if augment else 1
        for i in range(1, times + 1):
            suffix = f"{file_basename}_{i}.jpg"
            img_save_path = os.path.join(save_root, 'images', suffix)
            label_save_path = os.path.join(save_root, 'labels', suffix.replace('.jpg', '.txt'))

            img = self.apply_augmentations(image_path, i) if augment else Image.open(image_path).convert("RGB")
            if img is None:
                continue

            if isinstance(img, torch.Tensor):
                img = np.array(img.permute(1, 2, 0).cpu().numpy() * 255, dtype=np.uint8)
            else:
                img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(img_save_path, img)
            self.save_yolo_format(label_save_path, yolo_data)

    def get_label_path(self, image_path):
        return image_path.replace(self.original_folder, self.mask_folder).replace(".jpg", self.mask_type_ext).replace(
            ".png", self.mask_type_ext)

    def apply_augmentations(self, image_path, i):
        try:
            img = Image.open(image_path).convert("RGB")

            bright, contrast = [0.9, 1.1], [0.9, 1.1]
            if i % 5 == 1:
                bright, contrast = [0.7, 1.0], [0.7, 1.0]
            elif i % 5 == 2:
                bright, contrast = [0.5, 0.9], [0.9, 1.1]
            elif i % 5 == 3:
                bright, contrast = [0.8, 1.1], [0.8, 1.0]

            transform = T.Compose([
                T.ColorJitter(brightness=bright, contrast=contrast),
                T.ToTensor()
            ])
            img_tensor = transform(img).to(device)

            if i % 6 == 0:
                img_tensor = self.augmenter.apply_gaussian_blur(img_tensor)
            elif i % 6 == 1:
                img_tensor = self.augmenter.apply_average_blur(img_tensor)
            elif i % 6 == 2:
                img_tensor = self.augmenter.add_gaussian_noise(img_tensor, random.uniform(0, 0.5),
                                                               random.uniform(0.005, 0.04))
            elif i % 6 == 3:
                img_tensor = self.augmenter.add_salt_pepper_noise(img_tensor, random.uniform(0.005, 0.04),
                                                                  random.uniform(0.001, 0.05))
            return img_tensor
        except Exception as e:
            logging.error(f"Augmentation error on {image_path}: {e}")
            return None

    def process_mask_to_yolo_txt(self, mask_file_path):
        mask = cv2.imread(mask_file_path)
        mask_image = cv2.resize(mask_image, (img_width, img_height))  # Match training imgsz
        h, w = mask.shape[:2]
        polygons = self.get_polygons(mask)
        return self.convert_polygons_to_yolo(w, h, polygons)

    def get_polygons(self, mask_img):
        polygons = []
        for color, label in self.color_to_label.items():
            binary_mask = np.all(mask_img == color, axis=-1).astype(np.uint8) * 255
            # contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            '''
             Use cv2.RETR_TREE instead of RETR_EXTERNAL to capture all contours, and adjust cv2.CHAIN_APPROX_SIMPLE
             to cv2.CHAIN_APPROX_NONE for precise vertex points if road edges are complex:
            '''
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 0:
                    polygons.append((label, cnt.reshape(-1, 2)))
        return polygons

    def convert_polygons_to_yolo(self, width, height, polygons):
        yolo_polygons = []
        for label, poly in polygons:
            norm_poly = [(x / width, y / height) for (x, y) in poly]
            yolo_polygons.append((label, norm_poly))
        return yolo_polygons

    def save_yolo_format(self, path, polygons):
        try:
            with open(path, 'w') as f:
                for label, poly in polygons:
                    points = ' '.join(f"{x:.6f} {y:.6f}" for x, y in poly)
                    f.write(f"{label} {points}\n")
        except Exception as e:
            logging.error(f"Failed to save label: {e}")


if __name__ == "__main__":
    CONFIG = {
        "dataset_path": "D:\downloadFiles\\front_3\Dataset\\road",
        "mask_folder_name": "MaskImages",  # Change if different
        "original_folder_name": "OriginImages",  # Change if different
        "dataset_saving_working_dir": 'dataset_saving_working_dir',
        "augment_times": 10,  # Number of augmentations per image
        "test_split": 0.001,  # Percentage of data for testing
        "val_split": 0.1,  # Percentage of data for validation
        "train_split": 0.899,  # Percentage of data for training
        "Keep_val_dataset_original": True,  # for keeping the original dataset has original
        "num_threads": os.cpu_count() - 2,  # Number of threads for parallel processing
        "class_to_id": {'road': 0, },
        "color_to_label": {(255, 255, 255): 0, },
        "folder_name": 'road',
        "class_names": ['road'],
        "mask_type_ext": '.png',
        "FromDataType": '',
        "ToDataTypeFormate": '',
    }

    try:
        processor = YoloProcessor(config=CONFIG)
        processor.run()
    except Exception as e:
        logging.error(f"Critical error: {e}")

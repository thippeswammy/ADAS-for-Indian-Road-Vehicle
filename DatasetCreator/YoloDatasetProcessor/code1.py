import hashlib
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
    """Class to apply synchronized image augmentations for YOLO dataset preprocessing."""

    def __init__(self, config):
        self.hsv_h = config.get('hsv_h', 0.015)  # Hue augmentation
        self.hsv_s = config.get('hsv_s', 0.2)  # Saturation augmentation
        self.hsv_v = config.get('hsv_v', 0.2)  # Value augmentation
        self.scale = config.get('scale', 0.3)  # Scale augmentation
        self.perspective = config.get('perspective', 0.0001)  # Perspective transformation
        self.mosaic = config.get('mosaic', 1.0)  # Mosaic augmentation
        self.mixup = config.get('mixup', 0.0)  # Mixup augmentation
        self.flipud = config.get('flipud', 0.0)  # Vertical flip
        self.fliplr = config.get('fliplr', 0.5)  # Horizontal flip
        self.imgsz = config.get('imgsz', 640)  # Image size for resizing

    @staticmethod
    def apply_gaussian_blur(image, size=3):
        """Apply Gaussian blur using a kernel for three-channel images."""
        kernel = torch.ones((3, 1, size, size), dtype=torch.float32).to(device) / (size * size)
        image = image.unsqueeze(0)  # Add batch dimension
        return torch.nn.functional.conv2d(image, kernel, padding=size // 2, groups=3).squeeze(0)

    @staticmethod
    def apply_average_blur(image, size=3):
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

    def apply_mosaic(self, images, masks, cx, cy):
        """Apply mosaic augmentation by combining four images and their masks."""
        if len(images) < 4:
            return images[0], masks[0]
        h, w = self.imgsz, self.imgsz
        mosaic_img = torch.zeros((3, h, w), dtype=torch.float32).to(device)
        mosaic_mask = torch.zeros((3, h, w), dtype=torch.float32).to(device)

        for i, (img, mask) in enumerate(zip(images[:4], masks[:4])):
            img = T.Resize((h, w))(img)
            mask = T.Resize((h, w), interpolation=Image.NEAREST)(mask)
            if i == 0:  # Top-left
                mosaic_img[:, :cy, :cx] = img[:, :cy, :cx]
                mosaic_mask[:, :cy, :cx] = mask[:, :cy, :cx]
            elif i == 1:  # Top-right
                mosaic_img[:, :cy, cx:] = img[:, :cy, :w - cx]
                mosaic_mask[:, :cy, cx:] = mask[:, :cy, :w - cx]
            elif i == 2:  # Bottom-left
                mosaic_img[:, cy:, :cx] = img[:, :h - cy, :cx]
                mosaic_mask[:, cy:, :cx] = mask[:, :h - cy, :cx]
            elif i == 3:  # Bottom-right
                mosaic_img[:, cy:, cx:] = img[:, :h - cy, :w - cx]
                mosaic_mask[:, cy:, cx:] = mask[:, :h - cy, :w - cx]
        return mosaic_img, mosaic_mask

    def apply_mixup(self, img1, mask1, img2, mask2, lam):
        """Apply mixup augmentation, selecting the dominant mask based on lam."""
        img = lam * img1 + (1 - lam) * img2
        # Use the mask of the dominant image to avoid label ambiguity
        mask = mask1 if lam >= 0.5 else mask2
        return img, mask


class YoloProcessor:
    """Class to handle YOLO data processing with deterministic augmentations for images and masks."""

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
        self.mask_folder_name = config['mask_folder_name']
        self.original_folder_name = config['original_folder_name']
        self.ToDataTypeFormate = config['ToDataTypeFormate']
        self.augmenter = ImageAugmentations(config)
        self.color_to_label = config['color_to_label']
        self.mask_type_ext = config['mask_type_ext']
        self.FromDataType = config['FromDataType']
        self.class_names = config['class_names']
        self.class_to_id = config['class_to_id']
        self.train_split = config['train_split']
        self.source_dir_original_img = config['dataset_path'] + "/" + self.original_folder_name
        self.source_dir_mask_img = config['dataset_path'] + "/" + self.mask_folder_name
        self.test_split = config['test_split']
        self.val_split = config['val_split']
        self.main_path = config['dataset_saving_working_dir']
        self.factTimes = config['augment_times']
        self.num_threads = config['num_threads']
        self.keepValDatasetOriginal = config['Keep_val_dataset_original']
        self.all_image_paths = []  # Store all image paths for mixup/mosaic
        self.all_mask_paths = []  # Store all mask paths for mixup/mosaic

        if not os.path.exists(self.source_dir_original_img) or not os.path.exists(self.source_dir_mask_img):
            logging.error(
                f"Source directories '{self.source_dir_original_img}' or '{self.source_dir_mask_img}' do not exist.")
            raise FileNotFoundError(
                f"Source directories '{self.source_dir_original_img}' or '{self.source_dir_mask_img}' not found.")

    def distribute_files_with_threads(self):
        """Distribute files into training, validation, and test sets using multithreading."""
        self.all_image_paths = self.collect_image_paths(self.source_dir_original_img)
        self.all_mask_paths = [self.get_label_path(path) for path in self.all_image_paths]
        if not self.all_image_paths:
            logging.error("No image files were found in the source directory.")
            return

        # Shuffle image and mask paths together
        combined = list(zip(self.all_image_paths, self.all_mask_paths))
        random.shuffle(combined)
        self.all_image_paths, self.all_mask_paths = zip(*combined)

        # Split image paths into train, val, test
        total_files = len(self.all_image_paths)
        self.test_image_count = int(total_files * self.test_split)
        self.val_image_count = int(total_files * self.val_split)
        self.train_image_count = total_files - self.test_image_count - self.val_image_count

        train_paths = self.all_image_paths[:self.train_image_count]
        val_paths = self.all_image_paths[self.train_image_count:self.train_image_count + self.val_image_count]
        test_paths = self.all_image_paths[self.train_image_count + self.val_image_count:]

        # Process each set separately
        with tqdm(total=len(self.all_image_paths) * self.factTimes, desc="Processing Images") as pbar:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Process training set
                futures = [
                    executor.submit(self.process_single_file, (os.path.basename(path)[:-4], path), self.factTimes,
                                    "train")
                    for path in train_paths
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Exception during processing: {e}")
                    pbar.update(self.factTimes)

                # Process validation set
                futures = [
                    executor.submit(self.process_single_file, (os.path.basename(path)[:-4], path), self.factTimes,
                                    "val")
                    for path in val_paths
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Exception during processing: {e}")
                    pbar.update(self.factTimes)

                # Process test set
                futures = [
                    executor.submit(self.process_single_file, (os.path.basename(path)[:-4], path), self.factTimes,
                                    "test")
                    for path in test_paths
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Exception during processing: {e}")
                    pbar.update(self.factTimes)

        # Verify splits
        # self.verify_splits()

    def collect_image_paths(self, directory):
        """Collect all image file paths from the given directory."""
        image_paths = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(root, filename))
        logging.info(f"Found {len(image_paths)} images in the directory: {directory}.")
        return image_paths

    def process_single_file(self, file_info, Times, set_type):
        """Process a single file and its augmentations for a specific set."""
        file_basename, file_path = file_info
        try:
            self.process_and_save(file_basename, file_path, Times, set_type)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Error processing file {file_basename}: {e}")

    def get_label_path(self, image_source_path):
        """Construct the path for the label file based on the image path."""
        label_path = (image_source_path.replace(self.original_folder_name, self.mask_folder_name)
                      ).replace(".png", self.mask_type_ext).replace('jpg', self.mask_type_ext)
        return label_path

    def process_and_save(self, file_name, image_source_path, Times, set_type):
        """Process and save images, masks, and their corresponding YOLO annotations with matching filenames."""
        if not os.path.exists(image_source_path):
            logging.warning(f"Image file not found: {image_source_path}")
            return

        label_source_path = self.get_label_path(image_source_path)
        if not os.path.exists(label_source_path):
            logging.warning(f"Label file not found for image: {image_source_path}")
            return

        # For validation set with keepValDatasetOriginal, only save the resized original image and mask
        if set_type == "val" and self.keepValDatasetOriginal:
            save_img_path = os.path.join(self.val_save_path, 'images', f'{file_name}_1.jpg')
            save_label_path = os.path.join(self.val_save_path, 'labels', f'{file_name}_1.txt')
            try:
                img = Image.open(image_source_path).convert("RGB")
                mask = Image.open(label_source_path).convert("RGB")
                # Resize both to imgsz
                img = T.Resize((self.augmenter.imgsz, self.augmenter.imgsz))(img)
                mask = T.Resize((self.augmenter.imgsz, self.augmenter.imgsz), interpolation=Image.NEAREST)(mask)
                # Convert mask to YOLO annotations
                mask_np = np.array(mask)
                yolo_polygons_points_txt = self.process_mask_to_yolo_txt_from_array(mask_np, self.class_to_id)
                # Save image and annotations
                image_np = np.array(img)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                self.save_yolo_format(save_label_path, yolo_polygons_points_txt)
                cv2.imwrite(save_img_path, image_np)
            except Exception as e:
                logging.error(f"Error saving image {save_img_path}: {e}")
            return

        # For other sets or if keepValDatasetOriginal is False, apply augmentations
        for i in range(1, Times + 1):
            if set_type == "train":
                save_img_path = os.path.join(self.train_save_path, 'images', f'{file_name}_{i}.jpg')
                save_label_path = os.path.join(self.train_save_path, 'labels', f'{file_name}_{i}.txt')
            elif set_type == "val":
                save_img_path = os.path.join(self.val_save_path, 'images', f'{file_name}_{i}.jpg')
                save_label_path = os.path.join(self.val_save_path, 'labels', f'{file_name}_{i}.txt')
            elif set_type == "test":
                save_img_path = os.path.join(self.test_save_path, 'images', f'{file_name}_{i}.jpg')
                save_label_path = os.path.join(self.test_save_path, 'labels', f'{file_name}_{i}.txt')
            else:
                continue

            augmented_img, augmented_mask_txt = self.apply_augmentations(image_source_path, label_source_path,
                                                                         file_name, i)
            if augmented_img is not None and augmented_mask_txt is not None:
                try:
                    if isinstance(augmented_img, torch.Tensor):
                        image_np = np.array(augmented_img.permute(1, 2, 0).cpu().numpy() * 255, dtype=np.uint8)
                    else:
                        image_np = np.array(augmented_img)  # PIL Image to numpy array
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    self.save_yolo_format(save_label_path, augmented_mask_txt)
                    cv2.imwrite(save_img_path, image_np)
                except Exception as e:
                    logging.error(f"Error saving image {save_img_path}: {e}")

    def get_augmentation_params(self, file_name, num):
        """Generate deterministic augmentation parameters based on file name and iteration."""
        # Create a unique seed for this file and iteration
        seed_str = f"{file_name}_{num}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Brightness and contrast
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
        elif num == 5:
            bright = [0.7, 1.1]
            contrast = [0.8, 1.1]
        else:
            bright = [0.99, 1.11]
            contrast = [0.99, 1.11]

        # Flip decisions
        do_fliplr = random.random() < self.augmenter.fliplr
        do_flipud = random.random() < self.augmenter.flipud

        # Mosaic parameters
        do_mosaic = random.random() < self.augmenter.mosaic
        cx = random.randint(self.augmenter.imgsz // 4, 3 * self.augmenter.imgsz // 4) if do_mosaic else None
        cy = random.randint(self.augmenter.imgsz // 4, 3 * self.augmenter.imgsz // 4) if do_mosaic else None
        mosaic_indices = random.sample(range(len(self.all_image_paths)),
                                       min(3, len(self.all_image_paths))) if do_mosaic else []

        # Mixup parameters
        do_mixup = random.random() < self.augmenter.mixup
        mixup_idx = random.choice(range(len(self.all_image_paths))) if do_mixup else None
        lam = np.random.beta(0.5, 0.5) if do_mixup else None

        # Additional augmentation type
        aug_type = num % 6

        # Reset random seed to avoid affecting other operations
        random.seed()
        np.random.seed()
        torch.manual_seed(torch.initial_seed())

        return {
            'bright': bright,
            'contrast': contrast,
            'do_fliplr': do_fliplr,
            'do_flipud': do_flipud,
            'do_mosaic': do_mosaic,
            'cx': cx,
            'cy': cy,
            'mosaic_indices': mosaic_indices,
            'do_mixup': do_mixup,
            'mixup_idx': mixup_idx,
            'lam': lam,
            'aug_type': aug_type
        }

    def apply_augmentations(self, source_img_path, source_mask_path, file_name, num):
        """Apply synchronized YOLO dataset augmentations to both original and mask images."""
        try:
            # Load images
            img = Image.open(source_img_path).convert("RGB")
            mask = Image.open(source_mask_path).convert("RGB")

            # Get deterministic augmentation parameters
            params = self.get_augmentation_params(file_name, num)

            # Initialize augmentation pipeline for both image and mask
            augmentation_list = [
                T.Resize((self.augmenter.imgsz, self.augmenter.imgsz), interpolation=Image.BILINEAR),  # Resize image
                T.ColorJitter(
                    brightness=params['bright'],
                    contrast=params['contrast'],
                    hue=self.augmenter.hsv_h,  # HSV hue augmentation
                    saturation=self.augmenter.hsv_s  # HSV saturation augmentation
                ),
                T.RandomAffine(
                    degrees=0,
                    scale=(1 - self.augmenter.scale, 1 + self.augmenter.scale),  # Scale augmentation
                    translate=(self.augmenter.perspective, self.augmenter.perspective)  # Perspective augmentation
                )
            ]
            mask_augmentation_list = [
                T.Resize((self.augmenter.imgsz, self.augmenter.imgsz), interpolation=Image.NEAREST),  # Resize mask
                T.RandomAffine(
                    degrees=0,
                    scale=(1 - self.augmenter.scale, 1 + self.augmenter.scale),  # Scale augmentation
                    translate=(self.augmenter.perspective, self.augmenter.perspective)  # Perspective augmentation
                )
            ]

            # Apply flip augmentations
            if params['do_fliplr']:
                augmentation_list.append(T.RandomHorizontalFlip(p=1.0))  # Horizontal flip
                mask_augmentation_list.append(T.RandomHorizontalFlip(p=1.0))
            if params['do_flipud']:
                augmentation_list.append(T.RandomVerticalFlip(p=1.0))  # Vertical flip
                mask_augmentation_list.append(T.RandomVerticalFlip(p=1.0))

            # Apply augmentations
            augmentations = T.Compose(augmentation_list + [T.ToTensor()])
            mask_augmentations = T.Compose(mask_augmentation_list + [T.ToTensor()])
            img_tensor = augmentations(img).to(device)
            mask_tensor = mask_augmentations(mask).to(device)

            # Apply additional augmentations
            if params['aug_type'] == 0:
                img_tensor = self.augmenter.apply_gaussian_blur(img_tensor)
            elif params['aug_type'] == 1:
                img_tensor = self.augmenter.apply_average_blur(img_tensor)
            elif params['aug_type'] == 2:
                img_tensor = self.augmenter.add_gaussian_noise(img_tensor, mean=0.5, sigma=0.01)
            elif params['aug_type'] == 3:
                img_tensor = self.augmenter.add_salt_pepper_noise(img_tensor, salt_prob=0.01, pepper_prob=0.01)

            # Apply mosaic augmentation
            if params['do_mosaic'] and params['aug_type'] == 4:
                mosaic_images = [img_tensor]
                mosaic_masks = [mask_tensor]
                for idx in params['mosaic_indices']:
                    try:
                        other_img = Image.open(self.all_image_paths[idx]).convert("RGB")
                        other_mask = Image.open(self.all_mask_paths[idx]).convert("RGB")
                        other_img_tensor = T.Compose([
                            T.Resize((self.augmenter.imgsz, self.augmenter.imgsz)),
                            T.ToTensor()
                        ])(other_img).to(device)
                        other_mask_tensor = T.Compose([
                            T.Resize((self.augmenter.imgsz, self.augmenter.imgsz), interpolation=Image.NEAREST),
                            T.ToTensor()
                        ])(other_mask).to(device)
                        mosaic_images.append(other_img_tensor)
                        mosaic_masks.append(other_mask_tensor)
                    except Exception as e:
                        logging.warning(f"Error loading mosaic image/mask {self.all_image_paths[idx]}: {e}")
                img_tensor, mask_tensor = self.augmenter.apply_mosaic(mosaic_images, mosaic_masks, params['cx'],
                                                                      params['cy'])

            # Apply mixup augmentation
            if params['do_mixup'] and params['aug_type'] == 5:
                try:
                    other_img = Image.open(self.all_image_paths[params['mixup_idx']]).convert("RGB")
                    other_mask = Image.open(self.all_mask_paths[params['mixup_idx']]).convert("RGB")
                    other_img_tensor = T.Compose([
                        T.Resize((self.augmenter.imgsz, self.augmenter.imgsz)),
                        T.ToTensor()
                    ])(other_img).to(device)
                    other_mask_tensor = T.Compose([
                        T.Resize((self.augmenter.imgsz, self.augmenter.imgsz), interpolation=Image.NEAREST),
                        T.ToTensor()
                    ])(other_mask).to(device)
                    img_tensor, mask_tensor = self.augmenter.apply_mixup(
                        img_tensor, mask_tensor, other_img_tensor, other_mask_tensor, params['lam']
                    )
                except Exception as e:
                    logging.warning(f"Error loading mixup image/mask {self.all_image_paths[params['mixup_idx']]}: {e}")

            # Process augmented mask to YOLO annotations
            mask_np = np.array(mask_tensor.permute(1, 2, 0).cpu().numpy() * 255, dtype=np.uint8)
            yolo_polygons_points_txt = self.process_mask_to_yolo_txt_from_array(mask_np, self.class_to_id)

            return img_tensor, yolo_polygons_points_txt
        except Exception as e:
            logging.error(f"Error applying augmentations on {source_img_path}: {e}")
            return None, None

    def process_mask_to_yolo_txt_from_array(self, mask_array, class_map):
        """Convert a mask array to YOLO format annotations."""
        image_height, image_width = mask_array.shape[:2]
        polygons = self.get_polygons(mask_array)
        yolo_polygons_txt = self.convert_polygons_to_yolo(image_width, image_height, polygons)
        return yolo_polygons_txt

    def process_mask_to_yolo_txt(self, mask_file_path, class_map):
        """Convert the mask file to YOLO format."""
        mask_image = cv2.imread(mask_file_path)
        return self.process_mask_to_yolo_txt_from_array(mask_image, class_map)

    def get_polygons(self, mask_image):
        """Extract polygons for each label in the mask image."""
        polygons = []
        for color, label in self.color_to_label.items():
            mask = np.all(mask_image == color, axis=-1).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    # Simplify contour to reduce fragmentation
                    epsilon = 0.001 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    polygons.append((label, approx.reshape(-1, 2)))
        return polygons

    def convert_polygons_to_yolo(self, img_width, img_height, polygons):
        """Convert polygon coordinates to YOLO format."""
        yolo_polygons = []
        for label, polygon in polygons:
            normalized_polygon = [(max(0, min(x / img_width, 1)), max(0, min(y / img_height, 1))) for (x, y) in polygon]
            if len(normalized_polygon) >= 3:  # Ensure valid polygon
                yolo_polygons.append((label, normalized_polygon))
        return yolo_polygons

    def save_yolo_format(self, save_label_path, yolo_polygons):
        """Save the YOLO formatted text to the specified path."""
        try:
            with open(save_label_path, 'w') as f:
                for label, polygon in yolo_polygons:
                    polygon_str = ' '.join(f"{x:.6f} {y:.6f}" for x, y in polygon)
                    f.write(f"{label} {polygon_str}\n")
        except Exception as e:
            logging.error(f"Error saving YOLO label file {save_label_path}: {e}")

    def verify_splits(self):
        """Verify that train, val, and test sets are mutually exclusive."""
        train_images = set(os.listdir(os.path.join(self.train_save_path, 'images')))
        val_images = set(os.listdir(os.path.join(self.val_save_path, 'images')))
        test_images = set(os.listdir(os.path.join(self.test_save_path, 'images')))

        # Extract base filenames (without augmentation suffix)
        train_bases = {img.split('_')[0] for img in train_images}
        val_bases = {img.split('_')[0] for img in val_images}
        test_bases = {img.split('_')[0] for img in test_images}

        # Check for overlaps
        train_val_overlap = train_bases & val_bases
        train_test_overlap = train_bases & test_bases
        val_test_overlap = val_bases & test_bases

        if train_val_overlap or train_test_overlap or val_test_overlap:
            logging.error("Data leakage detected!")
            logging.error(f"Train-Val overlap: {train_val_overlap}")
            logging.error(f"Train-Test overlap: {train_test_overlap}")
            logging.error(f"Val-Test overlap: {val_test_overlap}")
        else:
            logging.info("No data leakage detected. Splits are mutually exclusive.")


if __name__ == '__main__':
    CONFIG = {
        "dataset_path": "D:\downloadFiles\\front_3\Dataset\\road",
        "mask_folder_name": "MaskImages",
        "original_folder_name": "OriginImages",
        "dataset_saving_working_dir": 'dataset_saving_working_dir',
        "augment_times": 3,
        "test_split": 0.02,
        "val_split": 0.09,
        "train_split": 0.89,
        "Keep_val_dataset_original": True,
        "num_threads": os.cpu_count() - 2,
        "class_to_id": {'road': 0},
        "color_to_label": {(255, 255, 255): 0},
        "folder_name": 'road',
        "class_names": ['road'],
        "mask_type_ext": '.png',
        "FromDataType": '',
        "ToDataTypeFormate": '',
        "imgsz": 640,
        "hsv_h": 0.3,
        "hsv_s": 0.3,
        "hsv_v": 0.3,
        "fliplr": 0.3,
        "mixup": 0.0,
        "scale": 0.0,
        "flipud": 0.0,
        "mosaic": 0.0,
        "perspective": 0.0,
    }

    try:
        processor = YoloProcessor(config=CONFIG)
        processor.distribute_files_with_threads()
    except Exception as e:
        logging.error(f"Critical error: {e}")

from Helper import YoloFormateToMaskImg

# Example usage:
# You can change these paths as needed
input_path = r'../YoloDatasetProcessor/dataset_saving_working_dir/road/train/images'
labels_dir = r'../YoloDatasetProcessor/dataset_saving_working_dir/road/train/labels'
output_dir = r'../YoloDatasetProcessor/dataset_saving_working_dir/road/train/masks'

if __name__ == "__main__":
    YoloFormateToMaskImg.main(input_path, labels_dir, output_dir)

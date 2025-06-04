import cv2
import os
import math
import random
import shutil
import numpy as np
import create_yolo_folders
from PIL import Image, ImageEnhance
from tqdm import tqdm

TrainImageSize = 1000
ValImageSize = 1000
TestImageSize = 1000
datasetFileName, _ = create_yolo_folders.create_yolo_folder_structure('')
train_path_for_saving = f'../{datasetFileName}/train'
val_path_for_saving = f'../{datasetFileName}/valid'
test_path_for_saving = f'../{datasetFileName}/test'

source_path_ = '../FromYolovDataByYolo'


def distribute_files():
    global TrainImageSize, ValImageSize, TestImageSize
    # Get the total number of .jpg files in the source path
    total_files = sum([len(files) for r, d, files in os.walk(source_path_)
                       if any(f.endswith('.jpg') for f in files)])
    TrainImgLen = int(total_files * 0.7)
    TestImgLen = int(total_files * 0.1)
    ValImgLen = total_files - TrainImgLen - TestImgLen
    print(f'TrainImgLen : {TrainImgLen}, TestImgLen : {TestImgLen}, ValImgLen : {ValImgLen}')
    # Initialize the progress bar
    with tqdm(total=total_files, desc="Processing Images") as pbar:
        for root, dirs, files in os.walk(source_path_):
            for filename in files:
                if filename.endswith('.jpg'):
                    file_path = os.path.join(root, filename)
                    ProcessAndSave(filename[:-4], file_path)
                    pbar.update(1)  # Update the progress bar for each processed file


def ProcessAndSave(file_name, ImageSourcePath):
    global TrainImageSize, ValImageSize, train_path_for_saving
    global val_path_for_saving, test_path_for_saving, TestImageSize
    destination_path_img = ''
    destination_path_lab = ''
    ram_num = random.randint(1, 3)
    ImageSourcePath = os.path.normpath(ImageSourcePath)
    LabelSourcePath = ImageSourcePath.split(os.sep)
    LabelSourcePath[-2] = 'labels'
    LabelSourcePath = '/'.join(LabelSourcePath)
    LabelSourcePath = LabelSourcePath[:-4] + '.txt'
    if os.path.exists(LabelSourcePath):
        if TrainImgLen > 0 and (ram_num == 1 or ValImgLen == 0 or TestImgLen == 0):
            destination_path_img = train_path_for_saving + str('/images/') + file_name + '.jpg'
            destination_path_lab = train_path_for_saving + str('/labels/') + file_name + '.txt'
            TrainImgLen -= 1
        elif ValImgLen > 0 and (ram_num == 2 or TrainImgLen == 0 or TestImgLen == 0):
            destination_path_img = val_path_for_saving + str('/images/') + file_name + '.jpg'
            destination_path_lab = val_path_for_saving + str('/labels/') + file_name + '.txt'
            ValImgLen -= 1
        elif TestImgLen > 0 and (ram_num == 3 or ValImgLen == 0 or TrainImgLen == 0):
            destination_path_img = test_path_for_saving + str('/images/') + file_name + '.jpg'
            destination_path_lab = test_path_for_saving + str('/labels/') + file_name + '.txt'
            TestImgLen -= 1
        img = apply_Augmentations(source_img_path=ImageSourcePath,
                                  source_lab_path=LabelSourcePath,
                                  output_image_path=destination_path_img[:-4],
                                  output_label_path=destination_path_lab[:-4])
        cv2.imwrite(destination_path_img, img)


def apply_Augmentations(source_img_path, source_lab_path,
                        output_image_path, output_label_path, _return=True):
    img = cv2.imread(source_img_path)
    contrast_factor_dark = random.uniform(0.2, 0.8)
    enhancer = cv2.addWeighted(img, contrast_factor_dark, 0, 0, 0)
    cv2.imwrite(output_image_path + 'contrastD.jpg', enhancer)
    shutil.copyfile(source_lab_path, output_label_path + 'contrastD.txt')
    contrast_factor_bright = random.uniform(0.9, 1.6)
    enhancer = cv2.addWeighted(img, contrast_factor_bright, 0, 0, 0)
    cv2.imwrite(output_image_path + 'contrastB.jpg', enhancer)
    shutil.copyfile(source_lab_path, output_label_path + 'contrastB.txt')
    ram = random.choice([1, 2, 3, 4, 5, 6])
    if ram == 1:
        gaussianBlur = ApplyGaussianBlur(img)
        if _return:
            return gaussianBlur
        cv2.imwrite(output_image_path + 'GaussianBlur.jpg', gaussianBlur)
        hutil.copyfile(source_lab_path, output_label_path + 'GaussianBlur.txt')
    elif ram == 2:
        averageBlur = ApplyAverageBlur(img)
        if _return:
            return averageBlur
        cv2.imwrite(output_image_path + 'AverageBlur.jpg', averageBlur)
        hutil.copyfile(source_lab_path, output_label_path + 'AverageBlur.txt')
    elif ram == 3:
        gaussianNoise = AddGaussianNoise(img)
        if _return:
            return gaussianNoise
        cv2.imwrite(output_image_path + 'GaussianNoise.jpg', gaussianNoise)
        hutil.copyfile(source_lab_path, output_label_path + 'GaussianNoise.txt')
    elif ram == 4:
        saltPepperNoise = AddSaltPepperNoise(img)
        if _return:
            return saltPepperNoise
        cv2.imwrite(output_image_path + 'SaltPepperNoise.jpg', saltPepperNoise)
        hutil.copyfile(source_lab_path, output_label_path + 'SaltPepperNoise.txt')
    if ram <= 6:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if _return:
            return gray
        cv2.imwrite(output_image_path + 'gray.jpg', gray)
        hutil.copyfile(source_lab_path, output_label_path + 'gray.txt')


def ApplyGaussianBlur(image, size=3):
    Gauss = cv2.GaussianBlur(image, (size, size), 0)
    return Gauss


def ApplyAverageBlur(image, size=3):
    kernel = np.ones((size, size), np.float32) / (size * size)
    averaged_image = cv2.filter2D(image, -1, kernel)
    return averaged_image


def AddGaussianNoise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = image + gauss.astype(np.uint8)
    return noisy


def AddSaltPepperNoise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = (255, 255, 255)
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = (0, 0, 0)
    return noisy


distribute_files()

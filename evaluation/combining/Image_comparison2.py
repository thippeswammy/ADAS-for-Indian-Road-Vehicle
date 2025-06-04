import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def create_image_comparison(original_paths, original_mark_paths, predict_mask_paths, superpixel_image_paths,
                            modified_mask_paths, difference_mask_paths_P, difference_mask_paths_C,
                            overlapped_modified_mask_paths,
                            num_rows=4, num_cols=7):
    """
    Creates a grid of image comparisons with six subplots per row, each with a box drawn around it.

    Args:
        original_paths: List of paths to the original images.
        predict_mask_paths: List of paths to the predicted mask images.
        superpixel_image_paths: List of paths to the superpixel images.
        modified_mask_paths: List of paths to the modified mask images.
        difference_mask_paths: List of paths to the difference mask images.
        overlapped_modified_mask_paths: List of paths to the overlapped modified mask images.
        num_rows: Number of rows in the grid (default: 3).
        num_cols: Number of columns in the grid (default: 6).

    Returns:
        None
        :param difference_mask_paths_P:
    """

    # Validate input lengths
    if len(original_paths) != len(predict_mask_paths) != len(superpixel_image_paths) != len(modified_mask_paths) != len(
            difference_mask_paths) != len(overlapped_modified_mask_paths):
        raise ValueError("Input lists must have the same length.")

    total_images = len(original_paths)
    num_subplots = num_rows * num_cols

    if total_images > num_subplots:
        print(f"Warning: Only displaying the first {num_subplots} images due to grid size limitations.")
        original_paths = original_paths[:num_subplots]
        original_mark_paths = original_mark_paths[:num_subplots]
        predict_mask_paths = predict_mask_paths[:num_subplots]
        superpixel_image_paths = superpixel_image_paths[:num_subplots]
        modified_mask_paths = modified_mask_paths[:num_subplots]
        difference_mask_paths_P = difference_mask_paths_P[:num_subplots]
        difference_mask_paths_C = difference_mask_paths_C[:num_subplots]
        overlapped_modified_mask_paths = overlapped_modified_mask_paths[:num_subplots]

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20 * num_cols // 7, 10 * num_rows))

    # Flatten subplot array for easier iteration
    axs = axs.flatten()

    for i in range(min(len(original_paths), num_subplots)):
        # Load images
        original_img = cv2.imread(original_paths[i])
        original_mark_img = cv2.imread(original_mark_paths[i])
        predict_mask = cv2.imread(predict_mask_paths[i], cv2.IMREAD_GRAYSCALE)
        superpixel_img = cv2.imread(superpixel_image_paths[i])
        modified_mask = cv2.imread(modified_mask_paths[i], cv2.IMREAD_GRAYSCALE)
        difference_mask_P = cv2.imread(difference_mask_paths_P[i], cv2.IMREAD_GRAYSCALE)
        difference_mask_C = cv2.imread(difference_mask_paths_C[i], cv2.IMREAD_GRAYSCALE)
        # overlapped_modified_mask = cv2.imread(overlapped_modified_mask_paths[i])

        # Display images in subplots
        axs[i * 8].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axs[i * 8].set_title('original')
        axs[i * 8].axis('off')
        axs[i * 8].add_patch(
            Rectangle((0, 0), original_img.shape[1], original_img.shape[0], linewidth=1, edgecolor='b',
                      facecolor='none'))

        # Display images in subplots
        axs[i * 8 + 1].imshow(cv2.cvtColor(original_mark_img, cv2.COLOR_BGR2GRAY))
        axs[i * 8 + 1].set_title('groundTruth')
        axs[i * 8 + 1].axis('off')
        axs[i * 8 + 1].add_patch(
            Rectangle((0, 0), original_mark_img.shape[1], original_mark_img.shape[0], linewidth=1, edgecolor='b',
                      facecolor='none'))

        axs[i * 8 + 2].imshow(predict_mask, cmap='gray')
        axs[i * 8 + 2].set_title('predictOutput')
        axs[i * 8 + 2].axis('off')
        axs[i * 8 + 2].add_patch(
            Rectangle((0, 0), predict_mask.shape[1], predict_mask.shape[0], linewidth=1, edgecolor='b',
                      facecolor='none'))

        axs[i * 8 + 3].imshow(difference_mask_P, cmap='gray')
        axs[i * 8 + 3].set_title('differenceInPredicted')
        axs[i * 8 + 3].axis('off')
        axs[i * 8 + 3].add_patch(
            Rectangle((0, 0), difference_mask_P.shape[1], difference_mask_P.shape[0], linewidth=1, edgecolor='b',
                      facecolor='none'))

        axs[i * 8 + 4].imshow(cv2.cvtColor(superpixel_img, cv2.COLOR_BGR2RGB))
        axs[i * 8 + 4].set_title('superpixelImage')
        axs[i * 8 + 4].axis('off')
        axs[i * 8 + 4].add_patch(
            Rectangle((0, 0), superpixel_img.shape[1], superpixel_img.shape[0], linewidth=1, edgecolor='b',
                      facecolor='none'))

        axs[i * 8 + 5].imshow(modified_mask, cmap='gray')
        axs[i * 8 + 5].set_title('modifiedOutput')
        axs[i * 8 + 5].axis('off')
        axs[i * 8 + 5].add_patch(
            Rectangle((0, 0), modified_mask.shape[1], modified_mask.shape[0], linewidth=1, edgecolor='b',
                      facecolor='none'))

        axs[i * 8 + 6].imshow(difference_mask_C, cmap='gray')
        axs[i * 8 + 6].set_title('differenceInModified')
        axs[i * 8 + 6].axis('off')
        axs[i * 8 + 6].add_patch(
            Rectangle((0, 0), difference_mask_C.shape[1], difference_mask_P.shape[0], linewidth=1, edgecolor='b',
                      facecolor='none'))

        masked_img = original_img.copy()
        modified_mask = cv2.resize(modified_mask, (original_img.shape[1], original_img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        masked_img[modified_mask == 0] = 0
        axs[i * 8 + 7].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        axs[i * 8 + 7].set_title('ExtractingOutput')
        axs[i * 8 + 7].axis('off')
        axs[i * 8 + 7].add_patch(
            Rectangle((0, 0), original_img.shape[1], original_img.shape[0], linewidth=1,
                      edgecolor='b', facecolor='none'))

    plt.tight_layout()
    plt.show()


# List of image numbers to process (assuming you have 18 images)
# image_numbers_O = ["road1_00162.png", "road1_00438.png", "road3_00028.png", "road3_00158.png"]
# image_numbers = [162, 438, 661, 791]

# image_numbers_O = ["road1_00013.png", "road3_00072.png", "road2_00006.png", "road3_00158.png"]
# image_numbers = [13, 705, 604, 791]


# image_numbers_O = ["road1_00162.png", "road1_00438.png", "road3_00028.png", "road3_00158.png"]
image_numbers_O = ["road3_00004.png", "road3_00004.png", "road3_00004.png", "road3_00004.png"]
methode_names = ["any one pixel_segments", "majority pixel_segments", "varying superpixel pixel_segments",
                 "with out superpixel pixel_segments"]
image_numbers = [1, 3, 6, 10]
# image_numbers = [162, 438, 661, 791]

# Create lists of image paths using f-strings
original_image_paths = [f"D:\\downloadFiles\\front_3\\TestingVideo\\TempImg - Copy\\{image_num}" for
                        image_num in image_numbers_O]
original_mark_paths = [f"D:\\downloadFiles\\front_3\\TestingVideo\\TempMasks - Copy\\{image_num}" for
                       image_num in image_numbers_O]
numer = 10
predict_mask_paths = [
    f"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\SuperPixelMethods1\\{methode_names[0]}_{image_num}\\predicted_mask\\predicted_mask_frame_{637}.png"
    for image_num in image_numbers]
superpixel_image_paths = [
    f"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\SuperPixelMethods1\\{methode_names[0]}_{image_num}\\segments_images\\segments_display_frame_{637}.png"
    for image_num in image_numbers]
modified_mask_paths = [
    f"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\SuperPixelMethods1\\{methode_names[0]}_{image_num}\\modified_mask\\modified_mask_frame_{637}.png"
    for image_num in image_numbers]
difference_mask_paths_P = [
    f"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\SuperPixelMethods1\\{methode_names[0]}_{image_num}\\difference_mask_P\\difference_frame_{637}.png"
    for image_num in image_numbers]
difference_mask_paths_C = [
    f"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\SuperPixelMethods1\\{methode_names[0]}_{image_num}\\difference_mask_C\\difference_frame_{637}.png"
    for image_num in image_numbers]
overlapped_modified_mask_paths = [
    f"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\SuperPixelMethods1\\{methode_names[0]}_{image_num}\\overlay\\overlay_frame_{637}.png"
    for image_num in image_numbers]

create_image_comparison(original_image_paths, original_mark_paths, predict_mask_paths,
                        superpixel_image_paths, modified_mask_paths, difference_mask_paths_P, difference_mask_paths_C,
                        overlapped_modified_mask_paths, num_rows=4, num_cols=8)

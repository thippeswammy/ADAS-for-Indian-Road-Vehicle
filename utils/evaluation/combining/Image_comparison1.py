import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def create_image_comparison(original_paths, original_mark_paths, predict_mask_paths, superpixel_image_paths,
                            modified_mask_paths,
                            difference_mask_paths, overlapped_modified_mask_paths, num_rows=4, num_cols=7):
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
        difference_mask_paths = difference_mask_paths[:num_subplots]
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
        difference_mask = cv2.imread(difference_mask_paths[i], cv2.IMREAD_GRAYSCALE)
        overlapped_modified_mask = cv2.imread(overlapped_modified_mask_paths[i])

        # Display images in subplots
        axs[i * 7].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axs[i * 7].set_title('original')
        axs[i * 7].axis('off')
        axs[i * 7].add_patch(Rectangle((0, 0), original_img.shape[1], original_img.shape[0], linewidth=2, edgecolor='r',
                                       facecolor='none'))

        # Display images in subplots
        axs[i * 7 + 1].imshow(cv2.cvtColor(original_mark_img, cv2.COLOR_BGR2GRAY))
        axs[i * 7 + 1].set_title('originalMask')
        axs[i * 7 + 1].axis('off')
        axs[i * 7 + 1].add_patch(
            Rectangle((0, 0), original_mark_img.shape[1], original_mark_img.shape[0], linewidth=2, edgecolor='r',
                      facecolor='none'))
        axs[i * 7 + 2].imshow(predict_mask, cmap='gray')
        axs[i * 7 + 2].set_title('predictMask')
        axs[i * 7 + 2].axis('off')
        axs[i * 7 + 2].add_patch(
            Rectangle((0, 0), predict_mask.shape[1], predict_mask.shape[0], linewidth=2, edgecolor='r',
                      facecolor='none'))

        axs[i * 7 + 3].imshow(cv2.cvtColor(superpixel_img, cv2.COLOR_BGR2RGB))
        axs[i * 7 + 3].set_title('superpixelImage')
        axs[i * 7 + 3].axis('off')
        axs[i * 7 + 3].add_patch(
            Rectangle((0, 0), superpixel_img.shape[1], superpixel_img.shape[0], linewidth=2, edgecolor='r',
                      facecolor='none'))

        axs[i * 7 + 4].imshow(modified_mask, cmap='gray')
        axs[i * 7 + 4].set_title('modifiedMask')
        axs[i * 7 + 4].axis('off')
        axs[i * 7 + 4].add_patch(
            Rectangle((0, 0), modified_mask.shape[1], modified_mask.shape[0], linewidth=2, edgecolor='r',
                      facecolor='none'))

        axs[i * 7 + 5].imshow(difference_mask, cmap='gray')
        axs[i * 7 + 5].set_title('differenceInMask')
        axs[i * 7 + 5].axis('off')
        axs[i * 7 + 5].add_patch(
            Rectangle((0, 0), difference_mask.shape[1], difference_mask.shape[0], linewidth=2, edgecolor='r',
                      facecolor='none'))
        masked_img = original_img.copy()
        modified_mask = cv2.resize(modified_mask, (original_img.shape[1], original_img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        masked_img[modified_mask == 0] = 0
        axs[i * 7 + 6].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        axs[i * 7 + 6].set_title('Extracting')
        axs[i * 7 + 6].axis('off')
        axs[i * 7 + 6].add_patch(
            Rectangle((0, 0), original_img.shape[1], original_img.shape[0], linewidth=2,
                      edgecolor='r', facecolor='none'))

    plt.tight_layout()
    plt.show()


# List of image numbers to process (assuming you have 18 images)
# image_numbers_O = ["road1_00162.png", "road1_00438.png", "road3_00028.png", "road3_00158.png"]
# image_numbers = [162, 438, 661, 791]

# image_numbers_O = ["road1_00013.png", "road3_00072.png", "road2_00006.png", "road3_00158.png"]
# image_numbers = [13, 705, 604, 791]


image_numbers_O = ["road1_00013.png", "road3_00072.png", "road2_00006.png", "road3_00158.png"]
image_numbers = [13, 705, 604, 791]

# Create lists of image paths using f-strings
original_image_paths = [f"D:\\downloadFiles\\front_3\\TestingVideo\\TempImg - Copy\\{image_num}" for
                        image_num in image_numbers_O]
original_mark_paths = [f"D:\\downloadFiles\\front_3\\TestingVideo\\TempMasks - Copy\\{image_num}" for
                       image_num in image_numbers_O]
predict_mask_paths = [
    f"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\SuperPixelMethods\\any one pixel_segments_1\\predicted_mask\\predicted_mask_frame_{image_num}.png"
    for image_num in image_numbers]
superpixel_image_paths = [
    f"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\SuperPixelMethods\\any one pixel_segments_1\\segments_images\\segments_display_frame_{image_num}.png"
    for image_num in image_numbers]
modified_mask_paths = [
    f"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\SuperPixelMethods\\any one pixel_segments_1\\modified_mask\\modified_mask_frame_{image_num}.png"
    for image_num in image_numbers]
difference_mask_paths = [
    f"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\SuperPixelMethods\\any one pixel_segments_1\\difference_mask\\difference_frame_{image_num}.png"
    for image_num in image_numbers]
overlapped_modified_mask_paths = [
    f"D:\\downloadFiles\\front_3\\TestingVideo\\PredictedImagesByMyModel\\SuperPixelMethods\\any one pixel_segments_1\\overlay\\overlay_frame_{image_num}.png"
    for image_num in image_numbers]

create_image_comparison(original_image_paths, original_mark_paths, predict_mask_paths,
                        superpixel_image_paths, modified_mask_paths, difference_mask_paths,
                        overlapped_modified_mask_paths, num_rows=4, num_cols=7)

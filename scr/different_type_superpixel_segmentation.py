import cv2
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries, slic, quickshift, felzenszwalb


def process_slic(image):
    slic_segments = slic(image, n_segments=200, compactness=10, start_label=1)
    return mark_boundaries(image, slic_segments)


def process_felzenszwalb(image):
    felzenszwalb_segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    return mark_boundaries(image, felzenszwalb_segments)


def process_quickshift(image):
    quickshift_segments = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
    return mark_boundaries(image, quickshift_segments)


def process_lsc(image):
    if hasattr(cv2.ximgproc, 'createSuperpixelLSC'):
        lsc = cv2.ximgproc.createSuperpixelLSC(image, region_size=25, ratio=0.075)
        lsc.iterate(10)
        lsc_mask = lsc.getLabelContourMask()
        lsc_image = image.copy()
        lsc_image[lsc_mask == 255] = [255, 0, 0]  # Add contour in red
        return lsc_image
    else:
        print("LSC method is not available in your OpenCV installation.")
        return image


def process_seeds(image):
    if hasattr(cv2.ximgproc, 'createSuperpixelSEEDS'):
        seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], 200, 10)
        seeds.iterate(image, 10)
        seeds_mask = seeds.getLabelContourMask()
        seeds_image = image.copy()
        seeds_image[seeds_mask == 255] = [0, 255, 0]  # Add contour in green
        return seeds_image
    else:
        print("SEEDS method is not available in your OpenCV installation.")
        return image


def process_crs(image):
    if hasattr(cv2.ximgproc, 'createSuperpixelCRS'):
        crs = cv2.ximgproc.createSuperpixelCRS(image, num_iterations=10)
        crs.iterate(10)
        crs_mask = crs.getLabelContourMask()
        crs_image = image.copy()
        crs_image[crs_mask == 255] = [0, 0, 255]  # Add contour in blue
        return crs_image
    else:
        print("CRS method is not available in your OpenCV installation.")
        return image


def superpixels_methods(image):
    # Load the input image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for skimage compatibility

    # Initialize result storage
    results = []

    # Method 1: SLIC
    slic_image = process_slic(image)
    results.append(("SLIC", slic_image))

    # Method 2: Felzenszwalb
    felzenszwalb_image = process_felzenszwalb(image)
    results.append(("Felzenszwalb", felzenszwalb_image))

    # Method 3: Quickshift
    quickshift_image = process_quickshift(image)
    results.append(("Quickshift", quickshift_image))

    # Method 4: LSC (Edge-Aware Superpixels)
    lsc_image = process_lsc(image)
    results.append(("LSC", lsc_image))

    # Method 5: SEEDS (Superpixels by SEEDS)
    seeds_image = process_seeds(image)
    results.append(("SEEDS", seeds_image))

    # Method 6: CRS (Contour Relaxed Superpixels)
    crs_image = process_crs(image)
    results.append(("CRS", crs_image))

    # Plot all results
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    for ax, (title, result_image) in zip(axes.ravel(), results):
        ax.imshow(result_image)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    return results


img = cv2.imread(r"D:\downloadFiles\front_3\TestingVideo\TempImg - Copy\road1_00001.png")
# Provide the path to your image
superpixels_methods(img)

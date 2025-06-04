# Data Processing (`data_processing/`)

This directory contains scripts and resources for preparing and processing datasets, primarily for YOLO-based model training.

## Key Scripts and Components

### `YoloDatasetProcessor/`
This sub-directory is central to creating YOLO compatible datasets.
- **`create_yolo_folders.py`**: Sets up the standard YOLO directory structure (train/valid/test with images/labels subfolders) and a `data.yaml` file.
- **`dataset_creator_from_Images.py`**: Processes a collection of source images and their corresponding masks to generate a YOLO dataset. It handles splitting into train/validation/test sets and applies augmentations.
  - **Usage**: Typically run as a script, configured via a `CONFIG` dictionary within the file.
- **`dataset_creator_from_video.py`**: Processes source video and a corresponding mask video to generate a YOLO dataset. It extracts frames, applies augmentations, and creates labels.
  - **Usage**: Typically run as a script, configured via a `CONFIG` dictionary within the file, and requires paths to original and mask videos.

### `ApplyAugmentationToYoloDatasetFormate.py`
- **Purpose**: Applies augmentations to an existing dataset that is already in YOLO format.
- **Usage**: (Describe general usage or refer to script for details)

### `iddDatasetSegmentation.py`
- **Purpose**: Likely contains specific logic for processing or segmenting the IDD (Indian Driving Dataset).
- **Usage**: (Describe general usage or refer to script for details)

### `Model_info1.py`
- **Purpose**: A script to extract and summarize layer information from a YOLO model, saving the output to `yolo_detailed_model_summary.csv`.
- **Usage**: Run as a script, requires a path to a trained model checkpoint.
  ```bash
  python data_processing/Model_info1.py
  ```

### `YoloFormateReference/data.yaml`
- An example `data.yaml` file, which is crucial for YOLO models to understand dataset paths and class names.

## General Workflow
1. Use scripts in `YoloDatasetProcessor/` (like `dataset_creator_from_Images.py` or `dataset_creator_from_video.py`) to convert raw image/video data into YOLO formatted datasets. This usually involves:
    - Setting up the folder structure using `create_yolo_folders.py`.
    - Processing source images/videos and masks.
    - Applying augmentations.
    - Generating label files (e.g., `.txt` files with normalized coordinates).
2. (Optional) Use `ApplyAugmentationToYoloDatasetFormate.py` if further augmentation is needed on an already formatted YOLO dataset.
3. (Informational) Use `Model_info1.py` to inspect model architectures.

**Note**: Many scripts are configured internally (e.g., via `CONFIG` dictionaries). Please refer to the specific scripts for detailed usage and configuration instructions.

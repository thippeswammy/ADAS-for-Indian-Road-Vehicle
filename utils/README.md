# Utilities (`utils/`)

This directory contains various utility scripts for common tasks needed across the project.

## Scripts

### `video2images.py`
- **Purpose**: Converts video files into a sequence of images (frames).
- **Usage**: (Provide a basic command line example if possible, or describe parameters)
  ```bash
  python utils/video2images.py --video_path <path_to_video> --output_folder <path_to_output_frames>
  # (Actual arguments might differ - to be verified)
  ```

### `images2video.py`
- **Purpose**: Creates a video file from a sequence of images.
- **Usage**: (Provide a basic command line example or describe parameters)
  ```bash
  python utils/images2video.py --image_folder <path_to_frames> --output_video_path <path_to_output_video>
  # (Actual arguments might differ - to be verified)
  ```

### `gpu_memory_utilization.py`
- **Purpose**: Monitors and logs GPU memory utilization. Useful for checking resource usage during model training or inference.
- **Usage**: (Describe how to run it or if it's imported)
  ```bash
  python utils/gpu_memory_utilization.py
  # (Might take arguments for logging interval, duration etc. - to be verified)
  ```

### `YoloFormateToMaskImg.py`
- **Purpose**: Converts YOLO format annotations (likely text files with bounding boxes/polygons) to mask images.
- **Usage**: (Describe how to run it or if it's imported, and what inputs it expects)

### `dataset_helpers/CheckMaskFile.py`
- **Purpose**: A utility to check or validate mask files, possibly for correctness or format.
- **Usage**: (Describe how to run it or if it's imported)

**Note**: The usage examples above are illustrative. Please refer to the scripts themselves for actual arguments and detailed usage instructions.

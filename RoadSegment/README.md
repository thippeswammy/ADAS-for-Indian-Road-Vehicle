# Model Training (`training/`)

This directory contains scripts for training road segmentation and object detection models, primarily using YOLO.

## Main Training Script

### `RoadSegment.py`
- **Purpose**: This is the primary script for testing YOLO models (likely YOLOv8-Seg) for road segmentation and potentially other object detection tasks.
- **Configuration**: Training is typically configured using command-line arguments.
- **Usage Example** (as referenced in the main project README.md):
  ```bash
  python training/RoadSegment.py
  ```
  *(Ensure `dataset.yaml`, model configuration `.yaml`, and pretrained weights `.pt` paths are correctly specified.)*

## Other Scripts

### `RoadSegmentPresentTesting.py`
- **Purpose**: Likely used for testing or demonstrating a trained `RoadSegment` model, possibly on sample images or videos.
- **Usage**: (Refer to script for details)

### `RoadSegmentWithMultiModel.py`
- **Purpose**: experimenting with multiple models simultaneously or in sequence.
- **Usage**: (Refer to script for details)

### `RoadSegmentWithMultiModelMultiVideoSaver.py`
- **Purpose**: Similar to `RoadSegmentWithMultiModel.py`, but possibly with added functionality to save outputs from multiple models on multiple videos.
- **Usage**: (Refer to script for details)

### `Road_segment_for_creating_exe_file.py`
- **Purpose**: Appears to be a version of the road segmentation script specifically packaged or prepared for compilation into an executable (`.exe`) file, perhaps using tools like PyInstaller.
- **Usage**: (Refer to script for details on its use or the executable creation process)

**Note**: For detailed usage, arguments, and configurations, please refer to the content of each script.

# Model Evaluation and Analysis (`evaluation/`)

This directory houses scripts dedicated to evaluating model performance, calculating metrics like Intersection over Union (IoU), analyzing results under different conditions (e.g., resolution, superpixel segmentation), and monitoring system resource usage during these processes.

## Key Areas of Evaluation

### IoU Calculation & Analysis
- Scripts like `Images_analysis_IOU_by_resolution.py`, `Video_analysis_IOU_by_resolution.py`, and their variants (`...usages.py`, `...usage_by_thread.py`) are used to calculate and analyze IoU metrics for image and video data, often considering different resolutions.
- `analyze_resolution_performance.py` and `analyze_resolution_performance2.py` likely further process or visualize these IoU results based on resolution.
- `analyze_superpixel_iou.py` focuses on IoU calculations in the context of superpixel segmentation.

### Superpixel Segmentation Analysis
- `superpixel_segmentation_video_analysis.py` is dedicated to analyzing video segmentation with superpixel methods.
- The `analysis_with_super_pixel/` subdirectory contains various plotting scripts (`data_analysis_plots[1-5].py`) likely used to visualize different aspects of superpixel segmentation performance.

### Resolution-Based Analysis
- The `analysis_with_resolution/` subdirectory contains plotting scripts (`data_analysis_plots[1-4].py`) for visualizing performance across different image/video resolutions.
- `select_image_resolutions.py` might be a utility to pick or manage different resolutions for testing.

### System Usage Monitoring
- `monitor_system_usage2.py`: A utility to monitor and plot CPU, RAM, and GPU usage. This can be imported and used by other evaluation scripts to track resource consumption during demanding tasks.

### Image Comparison
- Scripts in the `combining/` subdirectory (`Image_comparison1.py`, `Image_comparison2.py`) are likely used for visual comparison of model outputs, possibly side-by-side or overlaid.

## General Usage
- Many scripts in this folder are likely standalone analyses or plotting utilities.
- Some, like `monitor_system_usage2.py`, are designed to be imported by other scripts.
- Due to the varied nature of these scripts, please refer to the individual Python files for specific usage instructions, expected inputs, and outputs.

**Note**: The `data_analysis_plots*.py` files in subdirectories are kept as different versions or experiments as per user feedback.

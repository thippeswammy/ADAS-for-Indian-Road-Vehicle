# ADAS for Indian Road Vehicle

Advanced Driver Assistance Systems (ADAS) tailored for Indian road conditions. This project focuses on road segmentation, object detection, and instance segmentation using the YOLOv8-Seg model to enhance road safety and navigation in diverse Indian road scenarios.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)

---

## Project Overview
This project aims to develop an ADAS system specifically designed for Indian road conditions. The system leverages the YOLOv8-Seg model for:
- **Road Segmentation**: Identifying and segmenting road areas.
- **Object Detection**: Detecting vehicles, pedestrians, and other obstacles.
- **Instance Segmentation**: Providing detailed masks for detected objects.

The model is trained on a diverse dataset of Indian roads, including various weather conditions, lighting scenarios, and road types.

---

## Features
- **YOLOv8-Seg Model**: State-of-the-art instance segmentation for accurate road and object detection.
- **Diverse Dataset**: Includes 45,000+ annotated images of Indian roads under various conditions.
- **Superpixel Segmentation**: Enhances segmentation accuracy using SLIC superpixel methods.
- **Real-Time Performance**: Optimized for mid-range resolutions (e.g., 854×480) to balance accuracy and computational efficiency.
- **Data Augmentation**: Techniques like brightness adjustment, noise addition, and blur effects to improve model robustness.

---

## Dataset
The dataset consists of 40 videos recorded across different Indian road scenarios, converted into 45,000+ annotated images. Key features of the dataset:
- **Road Types**: Highways, urban roads, rural roads, and more.
- **Weather Conditions**: Sunny, cloudy, rainy, and nighttime.
- **Annotations**: Generated using the Segment Anything Model (SAM2) for efficient mask creation.

Dataset statistics:
- **Training**: 80%
- **Validation**: 10%
- **Testing**: 10%

---

## Project Structure
This project is organized into several Python packages and directories:

- **`data_processing/`**: Scripts and modules for dataset preparation, augmentation, and format conversion, especially for YOLO. Includes `YoloDatasetProcessor` for creating YOLO datasets.
- **`training/`**: Contains scripts for model training, primarily `train.py` for YOLOv8-Seg.
- **`utils/`**: A package for common utility scripts and modules.
    - **`utils/dataset_helpers/`**: Utilities specifically for dataset manipulation (e.g., `CheckMaskFile.py`).
    - **`utils/evaluation/`**: Scripts and modules for model evaluation, IoU calculation, performance analysis, and system monitoring. Contains sub-packages like `analysis_with_resolution`, `analysis_with_super_pixel`, etc.
- **`Model/`**: Contains model outputs, logs, and visualizations from training runs (e.g., `Model/RoadSeg/`).
- **`ROC/`**: Scripts related to ROC curve generation.
- **`scr/`**: Miscellaneous source scripts.
- **`weights/`**: Intended for storing downloaded pretrained model weights (e.g., `yolov8l-seg.pt`).
- **`dataset/`**: Intended for storing datasets in the structure required by training scripts.

Many of these directories (like `utils`, `data_processing`, `training`, `utils/evaluation` and their subdirectories) are structured as Python packages, allowing for modular imports.

---

## Installation
To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/thippeswammy/ADAS-for-Indian-Road-Vehicle.git
   cd ADAS-for-Indian-Road-Vehicle
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pretrained Weights**:
   You can download the **YOLOv8-Seg pretrained model** from the link below:
   📥 [Download Model Weights](https://drive.google.com/file/d/1jWOH5aDEPbf-oOUqRSweqnsfHgOfhyfb/view?usp=sharing)
   
   After downloading, place the file in the `weights/` directory:
   ```
   weights/
   ├── yolov8l-seg.pt  # Downloaded weights
   ```

4. **Prepare Dataset**:
   Ensure the dataset is organized in the following structure:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
   ```

---

## Usage
### Running the Model
To train the YOLOv8-Seg model:
```bash
python -m training/train.py 
```

### Inference on Test Images
To run inference on test images:
```bash
python python RoadSegment/RoadSegment.py
```

### Superpixel Segmentation
To apply superpixel segmentation:
```bash
python scr/different_type_superpixel_segmentation.py --method majority_pixel --n_segments 500
```
---

## Results
### Performance Metrics
- **Mean IoU**: 0.8983 (at 854×480 resolution)
- **Precision**: 0.9692
- **Recall**: 0.9501
- **Accuracy**: 95.21%

### Visual Results
#### Sample Output Video
https://github.com/user-attachments/assets/c799c248-3dcf-4942-b0d6-d8d7b0c3a33e

📹 **Watch the full demo video**: [Click here to view](https://drive.google.com/file/d/112LswURMs_aveyLxTfg6pWAl1nmTxpx8/view?usp=drive_link)
---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact
For questions or feedback, please contact:
- **Thippeswamy** - [thippeswamy636408@gmail.com](mailto:thippeswamy636408@gmail.com)
- **GitHub**: [thippeswammy](https://github.com/thippeswammy)

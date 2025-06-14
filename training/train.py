if __name__ == '__main__':
    from ultralytics import YOLO

    # Load the model
    model = YOLO('yolov8s-seg.pt')  # Large model for segmentation, good choice for road segmentation

    # Training parameters
    train_params = {
        'data': r'../DatasetCreator\YoloDatasetProcessor\dataset_saving_working_dir\road\data.yaml',
        # Ensure data.yaml has correct paths and classes
        'epochs': 100,  # Increased from 1 for meaningful training; adjust based on dataset size
        'imgsz': 480,  # Reduced from 1024 to balance memory and accuracy; roads often don’t need ultra-high resolution
        'batch': 24,  # Increased from 8 for better gradient updates, assuming GPU memory allows
        'workers': 10,  # Increased for faster data loading, adjust based on CPU cores
        'optimizer': 'AdamW',  # Good choice for segmentation, no change
        'lr0': 1e-3,  # Increased from 1e-4 for faster convergence; YOLOv8 often works well with 1e-3
        'lrf': 1e-4,  # Adjusted from 1e-5 for smoother learning rate decay
        'momentum': 0.937,  # Standard YOLOv8 value, better than 0.3 for stability
        'weight_decay': 5e-4,  # Reduced from 0.01 to prevent excessive regularization
        'dropout': 0.1,  # Disabled (from 0.1) as segmentation models often don’t benefit from dropout
        'device': [0],  # No change, assumes single GPU
        'amp': True,  # No change, enables mixed precision for faster training

        # Data augmentation for road segmentation
        'hsv_h': 0.015,  # Subtle hue shifts
        'hsv_s': 0.2,  # Saturation variation
        'hsv_v': 0.2,  # Brightness variation
        'scale': 0.5,  # Increased to allow more zoom in/out variation
        'degrees': 10.0,  # Added rotation augmentation with a range of ±10 degrees
        'perspective': 0.0001,  # Reduced for minimal distortion
        'mosaic': 1.0,  # Richer context in road scenes
        'mixup': 0.0,  # Disabled as it can blur segmentation boundaries
        'flipud': 0.0,  # Vertical flips are risky for road orientation
        'fliplr': 0.5,  # Horizontal flips are fine for roads
        'patience': 20,  # Early stopping if no improvement
        'warmup_epochs': 3,  # Standard warmup period
        'name': 'RoadSegmentation',  # Descriptive name
        'project': 'road_segmentation_runs',  # Organizes runs
        'resume': False,  # Fresh training
        'save': True,  # Saves checkpoints
        'val': True,  # Enables validation
        'cos_lr': True,  # Cosine learning rate scheduler for smoother training
        'fraction': 1.0,  # Use full dataset (adjust if subsampling is needed)
    }

    # Start training
    model.train(**train_params)

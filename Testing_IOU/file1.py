from ultralytics import YOLO

if __name__ == '__main__':
    path = r'F:\RunningProjects\YOLO_Model\Training\runs\segment\RoadSegmentationForMyDataset9\weights\best.pt'
    # Load a model
    model = YOLO(path)
    resolutionList = [240, 360, 480, 560, 640, 720, 1080, 1500, 1980]
    # Validate the model
    for i in resolutionList:
        metrics = model.val(data=r'F:\data.yaml', task='segment', imgsz=i)

        # Print metrics
        print("mAP@50-95:", metrics.box.map)  # map50-95
        print("mAP@50:", metrics.box.map50)  # map50
        print("mAP@75:", metrics.box.map75)  # map75
        print("Class-wise mAPs:", metrics.box.maps)  # List of map50-95 for each class

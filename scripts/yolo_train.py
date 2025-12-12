from ultralytics import YOLO

def train_yolo():
    # 1. Load the Model
    # "yolov8l.pt" is the Large version (Fair comparison to ResNet50)
    # Use "yolov8m.pt" (Medium) if you want it faster.
    model = YOLO('yolov8s.pt') 

    # 2. Train
    print("Starting YOLOv8 training...")
    results = model.train(
        data=r"C:\Users\gsamu\object_Detection_waymo\bdd.yaml",    # Path to your config
        epochs=3,           # Same epochs as Faster R-CNN
        imgsz=512,          # Standard YOLO size (Faster R-CNN uses ~800-1333, but 640 is standard for speed)
        batch=4,            # YOLO is efficient, try 8 or 16
        name=r"C:\Users\gsamu\object_Detection_waymo\results\bdd_yolo_run_s2", # Name of the folder where results will save
        device=0,           # GPU ID
        workers=2,
        exist_ok=True,       # Overwrite existing run folder
        
        # Fair Comparison settings (Optional)
        # mosaic=0.0         # Disable Mosaic augmentation if you want STRICT comparison (optional)
    )
    
    # 3. Validation (Get mAP)
    print("Training finished. Running validation...")
    metrics = model.val()
    
    print(f"mAP@50: {metrics.box.map50}")
    print(f"mAP@50-95: {metrics.box.map}")

if __name__ == "__main__":
    train_yolo()
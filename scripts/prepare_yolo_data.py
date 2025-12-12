import os
import json
import random
import shutil
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURATION ---
# Point these to your existing BDD folders
SOURCE_IMG_TRAIN = r"C:\Users\gsamu\object_Detection_waymo\dataset\train_images"
SOURCE_LBL_TRAIN = r"C:\Users\gsamu\object_Detection_waymo\annotations\train"
SOURCE_IMG_VAL = r"C:\Users\gsamu\object_Detection_waymo\dataset\val_images"
SOURCE_LBL_VAL = r"C:\Users\gsamu\object_Detection_waymo\annotations\val"

# Where to save the YOLO ready data
DEST_DIR = r"C:\Users\gsamu\object_Detection_waymo\dataset\bdd_yolo"
os.makedirs(DEST_DIR, exist_ok=True)
# BDD to YOLO Class Mapping
# NOTE: YOLO is 0-indexed. We removed "background".
CLASS_MAP = {
            "pedestrian": 0, "person": 0,"rider": 1, "car": 2, "truck": 3, 
            "bus": 4, "train": 5,"motor":6, "motorcycle": 6, "bike":7,"bicycle": 7,
            "traffic light": 8, "traffic sign": 9
        }

def convert_subset(img_source, lbl_source, split_name, limit=None):
    # Create directories
    save_img_dir = os.path.join(DEST_DIR, "images", split_name)
    save_lbl_dir = os.path.join(DEST_DIR, "labels", split_name)
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lbl_dir, exist_ok=True)

    # Get file list
    all_images = sorted([f for f in os.listdir(img_source) if f.endswith('.jpg')])
    
    # --- CRITICAL: MATCH FASTER R-CNN SUBSET ---
    if limit is not None and limit < len(all_images):
        print(f"Selecting random {limit} images for {split_name} (Seed 42)...")
        random.seed(42) # MUST match your Faster R-CNN seed
        random.shuffle(all_images)
        all_images = all_images[:limit]

    print(f"Processing {len(all_images)} images for {split_name}...")

    for img_file in tqdm(all_images):
        # 1. Copy Image (Symlink is faster/saves space)
        src_img_path = os.path.join(img_source, img_file)
        dst_img_path = os.path.join(save_img_dir, img_file)
        
        if not os.path.exists(dst_img_path):
            try:
                os.symlink(os.path.abspath(src_img_path), dst_img_path)
            except OSError:
                shutil.copy(src_img_path, dst_img_path)

        # 2. Convert Label
        json_file = img_file.replace('.jpg', '.json')
        json_path = os.path.join(lbl_source, json_file)
        
        yolo_lines = []
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract objects based on JSON structure
            objs = []
            if 'frames' in data: objs = data['frames'][0]['objects']
            elif 'labels' in data: objs = data['labels']
            
            # BDD Image Size
            img_w, img_h = 1280, 720

            for obj in objs:
                cat = obj.get('category', '')
                if 'box2d' in obj and cat in CLASS_MAP:
                    cls_id = CLASS_MAP[cat]
                    b = obj['box2d']
                    
                    # Math: Convert x1,y1,x2,y2 -> x_center, y_center, width, height (Normalized 0-1)
                    x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
                    
                    w = x2 - x1
                    h = y2 - y1
                    xc = x1 + (w / 2)
                    yc = y1 + (h / 2)
                    
                    # Normalize
                    xc /= img_w
                    yc /= img_h
                    w /= img_w
                    h /= img_h
                    
                    # YOLO Format: class_id center_x center_y width height
                    yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # Write .txt file
        txt_name = img_file.replace('.jpg', '.txt')
        with open(os.path.join(save_lbl_dir, txt_name), 'w') as f:
            f.write('\n'.join(yolo_lines))

if __name__ == "__main__":
    # Convert Train (30k Subset)
    convert_subset(SOURCE_IMG_TRAIN, SOURCE_LBL_TRAIN, "train", limit=30000)
    
    # Convert Val (Full set, or subset if you prefer)
    convert_subset(SOURCE_IMG_VAL, SOURCE_LBL_VAL, "val", limit=None) 
    
    print("✅ Conversion Complete. Ready for YOLO.")
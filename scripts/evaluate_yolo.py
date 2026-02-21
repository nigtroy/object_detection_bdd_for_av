import os
import json
import shutil
import yaml
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# --- CONFIGURATION ---
JSON_LABEL_DIR = "annotations/val/" 
# CAREFUL: These must match exactly where your 'prepare_yolo_data.py' saved things
YOLO_ROOT = "dataset/bdd_yolo"
YOLO_IMG_DIR = os.path.join(YOLO_ROOT, "images/val")
YOLO_LBL_DIR = os.path.join(YOLO_ROOT, "labels/val") # <--- NEW: Source of .txt files
MODEL_PATH = "results/bdd_yolo_run_s2/weights/best.pt"
RESULTS_FILE = "yolo_final_results_v2.json"

def create_subset(subset_name, file_list):
    """
    Creates a temporary folder structure for BOTH images and labels.
    """
    # 1. Setup Directories
    subset_img_dir = os.path.join(YOLO_ROOT, "images", subset_name)
    subset_lbl_dir = os.path.join(YOLO_ROOT, "labels", subset_name)
    
    # Clean up previous runs
    if os.path.exists(subset_img_dir): shutil.rmtree(subset_img_dir)
    if os.path.exists(subset_lbl_dir): shutil.rmtree(subset_lbl_dir)
    
    os.makedirs(subset_img_dir)
    os.makedirs(subset_lbl_dir)
    
    print(f"Creating subset '{subset_name}' with {len(file_list)} samples...")
    
    # 2. Link Images AND Labels
    for fname in file_list:
        # --- A. Handle Image ---
        # FIX: Convert Source to ABSOLUTE PATH
        src_img = os.path.abspath(os.path.join(YOLO_IMG_DIR, fname))
        dst_img = os.path.join(subset_img_dir, fname)
        
        if os.path.exists(src_img):
            try:
                os.symlink(src_img, dst_img)
            except OSError:
                # Fallback for Windows if Developer Mode is off
                shutil.copy(src_img, dst_img)
        
        # --- B. Handle Label ---
        txt_name = fname.replace('.jpg', '.txt')
        # FIX: Convert Source to ABSOLUTE PATH
        src_lbl = os.path.abspath(os.path.join(YOLO_LBL_DIR, txt_name))
        dst_lbl = os.path.join(subset_lbl_dir, txt_name)
        
        if os.path.exists(src_lbl):
            try:
                os.symlink(src_lbl, dst_lbl)
            except OSError:
                shutil.copy(src_lbl, dst_lbl)

    # 3. Create YAML
    # We point 'val' to the relative path 'images/subset_name'
    # YOLO automatically infers 'labels/subset_name' from this structure.
    yaml_data = {
        'path': os.path.abspath(YOLO_ROOT), 
        'train': f"images/{subset_name}", 
        'val': f"images/{subset_name}",   
        'nc': 10,
        'names': {0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus', 
                  5: 'train', 6: 'motorcycle', 7: 'bicycle', 8: 'traffic light', 9: 'traffic sign'}
    }
    
    yaml_file = f"bdd_{subset_name}.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_data, f)
        
    return yaml_file
    

def evaluate_yolo():
    # 1. Identify Day vs Night files
    print("Scanning JSONs for Day/Night attributes...")
    day_files = []
    night_files = []
    
    # Verify we can find the images
    if not os.path.exists(YOLO_IMG_DIR):
        print(f"❌ Error: Image dir not found at {YOLO_IMG_DIR}")
        return

    valid_yolo_images = set(os.listdir(YOLO_IMG_DIR))
    json_files = sorted([f for f in os.listdir(JSON_LABEL_DIR) if f.endswith('.json')])
    
    for json_file in tqdm(json_files):
        img_name = json_file.replace('.json', '.jpg')
        if img_name in valid_yolo_images:
            with open(os.path.join(JSON_LABEL_DIR, json_file)) as f:
                data = json.load(f)
                time_of_day = data.get('attributes', {}).get('timeofday', '')
                
                if time_of_day == 'daytime':
                    day_files.append(img_name)
                elif time_of_day == 'night':
                    night_files.append(img_name)

    print(f"Found {len(day_files)} Day images and {len(night_files)} Night images.")

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
        
    model = YOLO(MODEL_PATH)

    # 3. Evaluate OVERALL
    print("\n--- EVALUATING OVERALL PERFORMANCE ---")
    metrics_overall = model.val(data='bdd.yaml', split='val')
    map50_95_overall = metrics_overall.box.map
    map50_overall = metrics_overall.box.map50
    
    per_class_ap = {}
    class_names = metrics_overall.names
    for i, ap in enumerate(metrics_overall.box.maps):
        per_class_ap[class_names[i]] = ap

    # 4. Evaluate DAY
    print("\n--- EVALUATING DAY PERFORMANCE ---")
    # Using 500 images for speed. Increase if needed.
    day_yaml = create_subset("day", day_files[:500]) 
    metrics_day = model.val(data=day_yaml)
    
    map50_95_day = metrics_day.box.map
    map50_day = metrics_day.box.map50

    # 5. Evaluate NIGHT
    print("\n--- EVALUATING NIGHT PERFORMANCE ---")
    night_yaml = create_subset("night", night_files[:500])
    metrics_night = model.val(data=night_yaml)
    
    map50_95_night = metrics_night.box.map
    map50_night = metrics_night.box.map50

    # 6. Calculate Drop-offs
    drop_off_pct = 0.0
    if map50_95_day > 0:
        drop_off_pct = ((map50_95_day - map50_95_night) / map50_95_day) * 100
        
    print(f"\n📉 Performance Drop (mAP@50:95): {drop_off_pct:.2f}%")

    # 7. Save Results
    results = {
        "model": "YOLOv8-Small",
        "mAP_50_95_overall": map50_95_overall,
        "mAP_50_overall": map50_overall,
        "mAP_50_95_day": map50_95_day,
        "mAP_50_day": map50_day,
        "mAP_50_95_night": map50_95_night,
        "mAP_50_night": map50_night,
        "drop_off_percent": drop_off_pct,
        "per_class_ap_50_95": per_class_ap
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    evaluate_yolo()
import torch
import numpy as np
import time
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

# --- IMPORT YOUR DATASET CLASS ---
# Assuming it is in a file named 'train.py' or 'dataset.py'
# If it's in the same folder, just import it. 
# Otherwise, paste the BDD100K_IndividualFiles_Dataset class here.
from train import BDD100K_IndividualFiles_Dataset, collate_fn

# --- CONFIGURATION ---
VAL_IMG_DIR = r"C:\Users\gsamu\object_Detection_waymo\dataset\val_images"
VAL_LBL_DIR = r"C:\Users\gsamu\object_Detection_waymo\annotations\val"
MODEL_PATH = "models/bdd_model_epoch_2.pth" # Point to your best/last epoch
NUM_CLASSES = 11 # 10 objects + 1 background
RESULTS_FILE = "faster_rcnn_final_results_2.json"  # <--- NEW: Output file

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def calculate_ap(precisions, recalls):
    """ Calculate Area Under Curve (AP) using 11-point interpolation """
    ap = 0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.
    return ap

def evaluate_manual(model, loader, description="Evaluating"):
    model.eval()
    print(f"--- {description} ---")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for i in range(len(images)):
                keep = outputs[i]['scores'] > 0.05
                all_preds.append({
                    'boxes': outputs[i]['boxes'][keep].cpu(),
                    'scores': outputs[i]['scores'][keep].cpu(),
                    'labels': outputs[i]['labels'][keep].cpu()
                })
                all_targets.append({
                    'boxes': targets[i]['boxes'].cpu(),
                    'labels': targets[i]['labels'].cpu()
                })

    print("Calculating metrics...")
    class_aps = {}
    
    # BDD100K Classes 1-10
    class_names = {
            "person": 1, "pedestrian": 1,
            "rider": 2,
            "car": 3,
            "truck": 4,
            "bus": 5,
            "train": 6,
            "motor": 7, "motorcycle": 7,
            "bike": 8, "bicycle": 8,
            "traffic light": 9,
            "traffic sign": 10
        }

    for class_id in range(1, NUM_CLASSES):
        true_positives = []
        scores = []
        num_gt_instances = 0
        
        for i in range(len(all_preds)):
            pred_boxes = all_preds[i]['boxes']
            pred_scores = all_preds[i]['scores']
            pred_labels = all_preds[i]['labels']
            
            gt_boxes = all_targets[i]['boxes']
            gt_labels = all_targets[i]['labels']
            
            # Filter by class
            p_boxes = pred_boxes[pred_labels == class_id]
            p_scores = pred_scores[pred_labels == class_id]
            g_boxes = gt_boxes[gt_labels == class_id]
            
            num_gt_instances += len(g_boxes)
            
            if len(p_boxes) == 0: continue
            
            # Sort
            sorted_idxs = torch.argsort(p_scores, descending=True)
            p_boxes = p_boxes[sorted_idxs]
            p_scores = p_scores[sorted_idxs]
            
            if len(g_boxes) == 0:
                true_positives.extend([0] * len(p_boxes))
                scores.extend(p_scores.tolist())
                continue
                
            ious = box_iou(p_boxes, g_boxes)
            matched_gt = set()
            
            for j in range(len(p_boxes)):
                best_iou, best_gt_idx = torch.max(ious[j], dim=0)
                if best_iou > 0.5 and best_gt_idx.item() not in matched_gt:
                    true_positives.append(1)
                    matched_gt.add(best_gt_idx.item())
                else:
                    true_positives.append(0)
                scores.append(p_scores[j].item())

        if num_gt_instances == 0:
            class_aps[class_names.get(class_id, str(class_id))] = 0.0
            continue
            
        scores = np.array(scores)
        true_positives = np.array(true_positives)
        sorted_indices = np.argsort(-scores)
        true_positives = true_positives[sorted_indices]
        
        cumsum_tp = np.cumsum(true_positives)
        recalls = cumsum_tp / num_gt_instances
        precisions = cumsum_tp / (np.arange(len(true_positives)) + 1)
        
        ap = calculate_ap(precisions, recalls)
        class_name = class_names.get(class_id, str(class_id))
        class_aps[class_name] = ap
        print(f"{class_name} AP: {ap:.4f}")

    mAP = np.mean(list(class_aps.values())) if class_aps else 0.0
    print(f"✅ mAP@50 ({description}): {mAP:.4f}")
    
    return mAP, class_aps

def measure_speed(model, dataset):
    print("\n⏱️ Measuring Inference Speed...")
    model.eval()
    img, _ = dataset[0]
    img = [img.to(device)]
    
    for _ in range(5): 
        with torch.no_grad(): model(img)
        
    latencies = []
    for _ in range(50):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad(): model(img)
        torch.cuda.synchronize()
        end = time.time()
        latencies.append((end - start) * 1000)

    avg_lat = np.mean(latencies)
    fps = 1000 / avg_lat
    print(f"GPU Latency: {avg_lat:.2f} ms | FPS: {fps:.2f}")
    return avg_lat, fps

# --- HELPER FOR DAY/NIGHT ---
def get_day_night_indices(dataset):
    day_indices = []
    night_indices = []
    print("Scanning for Day/Night...")
    for idx, filename in enumerate(tqdm(dataset.image_files)):
        json_file = filename.replace('.jpg', '.json')
        json_path = os.path.join(dataset.label_dir, json_file)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                tod = data.get('attributes', {}).get('timeofday', 'undefined')
                if tod == 'daytime': day_indices.append(idx)
                elif tod == 'night': night_indices.append(idx)
    return day_indices, night_indices

if __name__ == "__main__":
    # 1. Load Data
    transform = T.Compose([T.ToTensor()])
    val_dataset = BDD100K_IndividualFiles_Dataset(image_dir=VAL_IMG_DIR, label_dir=VAL_LBL_DIR, transform=transform)
    
    # 2. Load Model
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    
    # 3. Overall Eval
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    overall_map, class_aps = evaluate_manual(model, val_loader, "Overall Validation")
    
    # 4. Day vs Night
    day_idx, night_idx = get_day_night_indices(val_dataset)
    # Using larger subset for accuracy (1000 images each)
    day_loader = DataLoader(Subset(val_dataset, day_idx[:1000]), batch_size=4, collate_fn=collate_fn)
    night_loader = DataLoader(Subset(val_dataset, night_idx[:1000]), batch_size=4, collate_fn=collate_fn)
    
    print("\n--- Day vs Night ---")
    day_map, _ = evaluate_manual(model, day_loader, "Day Subset")
    night_map, _ = evaluate_manual(model, night_loader, "Night Subset")
    
    # 5. Speed
    latency, fps = measure_speed(model, val_dataset)

    # --- 6. SAVE RESULTS TO FILE ---
    results_data = {
        "model": "Faster R-CNN",
        "mAP_50_overall": overall_map,
        "mAP_50_day": day_map,
        "mAP_50_night": night_map,
        "drop_off_percent": ((day_map - night_map) / day_map * 100) if day_map > 0 else 0,
        "latency_ms": latency,
        "fps": fps,
        "per_class_ap": class_aps
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results_data, f, indent=4)
        
    print(f"\n✅ Results saved successfully to {RESULTS_FILE}")
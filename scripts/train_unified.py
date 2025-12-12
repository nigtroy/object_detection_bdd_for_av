import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import sys
import os
import json
import glob
import re
import random
from PIL import Image

# --- CONFIGURATION ---
IMAGE_DIR = r"C:\Users\gsamu\object_Detection_waymo\dataset\train_images"
LABEL_DIR = r"C:\Users\gsamu\object_Detection_waymo\annotations\train"
MODEL_SAVE_DIR = "models"
CHECKPOINT_DIR = "models/epochs"

NUM_CLASSES = 11      # 10 BDD objects + 1 background
BATCH_SIZE = 4        # Adjust based on VRAM
NUM_EPOCHS = 3
LEARNING_RATE = 0.005 # Initial LR (will be lower if resuming)
SAVE_FREQUENCY = 500  # Save every 500 steps
LIMIT_IMAGES = 30000  # 30k subset
NUM_WORKERS = 4       # Set to 0 if Windows gives errors

# --- 1. DATASET CLASS (With Fixed Class Mapping) ---
class BDD100K_IndividualFiles_Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, limit=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # Get list of all images
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        
        # Shuffle and Slice
        if limit is not None and limit < len(self.image_files):
            print(f"Subsetting: Selecting {limit} random images from {len(self.image_files)} total.")
            random.seed(42) 
            random.shuffle(self.image_files)
            self.image_files = self.image_files[:limit]
            
        print(f"Final Dataset Size: {len(self.image_files)} images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        
        # Load Image
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        
        # Load Label
        json_filename = img_filename.replace('.jpg', '.json')
        json_path = os.path.join(self.label_dir, json_filename)
        
        boxes = []
        labels = []
        
        # --- FIXED CLASS MAPPING (Handling Aliases) ---
        class_map = {
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

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            objs = []
            if 'frames' in data: objs = data['frames'][0]['objects']
            elif 'labels' in data: objs = data['labels']
            
            for obj in objs:
                category = obj.get('category', '')
                if 'box2d' in obj and category in class_map:
                    b = obj['box2d']
                    x1, y1, x2, y2 = float(b['x1']), float(b['y1']), float(b['x2']), float(b['y2'])
                    
                    # Clip and Validate
                    w_img, h_img = image.size
                    x1 = max(0, min(x1, w_img))
                    y1 = max(0, min(y1, h_img))
                    x2 = max(0, min(x2, w_img))
                    y2 = max(0, min(y2, h_img))

                    if (x2 > x1) and (y2 > y1):
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_map[category])
        
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        if self.transform:
            image = self.transform(image)

        return image, target

# --- 2. MODEL SETUP ---
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

# --- 3. CHECKPOINT SEARCH LOGIC ---
def find_latest_checkpoint():
    """
    Scans for the absolute latest checkpoint (Step-based OR Epoch-based).
    Returns: (path, epoch, step)
    """
    if not os.path.exists(CHECKPOINT_DIR): return None, 0, -1
    
    # 1. Look for Step-Based Checkpoints (e.g. epoch_1_step_500.pth)
    search_pattern = os.path.join(CHECKPOINT_DIR, "bdd_checkpoint_epoch_*_step_*.pth")
    all_checkpoints = glob.glob(search_pattern)
    
    found_checkpoints = []
    for filename in all_checkpoints:
        match = re.search(r"epoch_(\d+)_step_(\d+)\.pth", filename)
        if match:
            found_checkpoints.append((int(match.group(1)), int(match.group(2)), filename))
            
    # 2. Look for End-of-Epoch Models (e.g. bdd_model_epoch_1.pth) in root model dir
    epoch_pattern = os.path.join(MODEL_SAVE_DIR, "bdd_model_epoch_*.pth")
    for filename in glob.glob(epoch_pattern):
        match = re.search(r"epoch_(\d+)\.pth", filename)
        if match:
            # Treated as step -1 of the NEXT epoch
            found_checkpoints.append((int(match.group(1)) + 1, -1, filename))

    if not found_checkpoints:
        return None, 0, -1

    # Sort by Epoch (Descending), then Step (Descending)
    found_checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    return found_checkpoints[0] # Returns tuple: (epoch, step, filename)


# --- 4. MAIN TRAINING FUNCTION ---
def train(resume=False):
    # Setup Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on: {device}")
    
    # Setup Directories
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Setup Data
    print("Loading Dataset...")
    transform = T.Compose([T.ToTensor()])
    dataset = BDD100K_IndividualFiles_Dataset(
        IMAGE_DIR, LABEL_DIR, transform=transform, limit=LIMIT_IMAGES
    )
    
    data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize Model
    model = get_model(NUM_CLASSES)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # Default LR, will be lowered if resuming
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    start_epoch = 0
    resume_step = -1

    # --- RESUME LOGIC ---
    if resume:
        latest_ep, latest_st, latest_file = find_latest_checkpoint()
        
        if latest_file:
            print(f"✅ Found checkpoint: {latest_file}")
            print(f"   Resuming from Epoch {latest_ep}, Step {latest_st + 1}")
            
            checkpoint = torch.load(latest_file)
            
            # Load Weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                # Optional: Load optimizer state if available
                # if 'optimizer_state_dict' in checkpoint:
                #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                model.load_state_dict(checkpoint) # Legacy format
            
            start_epoch = latest_ep
            resume_step = latest_st
            
            # Lower LR for stability when resuming
            if start_epoch > 0:
                print("   Reducing learning rate to 0.001 for stability.")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001
        else:
            print("❌ No checkpoint found. Starting from scratch.")

    # --- TRAINING LOOP ---
    print(f"Starting Training from Epoch {start_epoch}...")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(data_loader):
            
            # Fast Forward Check
            if epoch == start_epoch and i <= resume_step:
                if i % 100 == 0:
                    print(f"⏩ Skipping processed batch {i}/{resume_step}...", end='\r')
                continue 

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward Pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # NaN Check
            if not torch.isfinite(losses):
                print(f"⚠️ WARNING: Loss is {losses.item()} at step {i}. Skipping.")
                optimizer.zero_grad()
                continue

            # Backward Pass
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_loss += losses.item()
            
            # Logging
            if i % 20 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {losses.item():.4f}")

            # Mid-Epoch Save
            if i > 0 and i % SAVE_FREQUENCY == 0:
                save_path = os.path.join(CHECKPOINT_DIR, f"bdd_checkpoint_epoch_{epoch}_step_{i}.pth")
                print(f"💾 Saving backup: {save_path}")
                torch.save({
                    'epoch': epoch,
                    'step': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses.item()
                }, save_path)

        # End of Epoch Save
        avg_loss = epoch_loss / len(data_loader)
        print(f"--- End of Epoch {epoch} | Avg Loss: {avg_loss:.4f} ---")
        
        save_path = os.path.join(MODEL_SAVE_DIR, f"bdd_model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved: {save_path}")

if __name__ == "__main__":
    # To run: python train_unified.py
    # To resume: Change valid to True below, or use argparse if you prefer
    train(resume=True)
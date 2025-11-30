import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torchvision.transforms as T
import sys
import os
from train import BDD100K_IndividualFiles_Dataset
import glob
import re
import os

# --- CONFIGURATION ---
NUM_WORKERS = 4          # try 0, 4, or 8. If you get "Broken Pipe" on Windows, set to 0.
BATCH_SIZE = 4           
START_EPOCH = 1          # Resuming from Epoch 1
TOTAL_EPOCHS = 3         
SAVE_FREQUENCY = 500

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_resume():
    # 1. CHECK DEVICE SPEED
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Resuming training on: {device}")
    
    if device.type == 'cpu':
        print("⚠️ WARNING: You are training on CPU! This will be very slow (10s/step).")
        print("   If you have a GPU, make sure you installed the CUDA version of PyTorch.")
    elif torch.cuda.get_device_name(0):
         print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")

    # 2. SETUP DATASET
    print("Loading Training Data...")
    transform = T.Compose([T.ToTensor()])
    
    train_dataset = BDD100K_IndividualFiles_Dataset(
        image_dir=r"C:\Users\gsamu\object_Detection_waymo\dataset\train_images", 
        label_dir=r"C:\Users\gsamu\object_Detection_waymo\annotations\train",    
        transform=transform,
        limit=30000 
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 3. LOAD PREVIOUS MODEL
    # We look for the epoch BEFORE the one we want to start
    # e.g., if starting Epoch 1, we load Epoch 0.
    # 4. LOAD LATEST CHECKPOINT (AUTO-DETECT)
    # 4. LOAD LATEST CHECKPOINT (AUTO-DETECT FIXED)
    print("Searching for the latest checkpoint...")
    
    checkpoint_dir = "models/epochs"
    if not os.path.exists(checkpoint_dir):
        print(f"⚠️ Warning: Directory '{checkpoint_dir}' not found. Checking current directory...")
        checkpoint_dir = "."

    # Get all step-based checkpoints
    search_pattern = os.path.join(checkpoint_dir, "bdd_checkpoint_epoch_*_step_*.pth")
    all_checkpoints = glob.glob(search_pattern)

    resume_epoch = 0
    resume_step = -1
    checkpoint_path = None
    
    # --- LOGIC FIX: Sort by Tuple (Epoch, Step) ---
    found_checkpoints = []

    for filename in all_checkpoints:
        # Extract numbers using Regex
        match = re.search(r"epoch_(\d+)_step_(\d+)\.pth", filename)
        if match:
            ep = int(match.group(1))
            st = int(match.group(2))
            # Store as a tuple: (epoch, step, filename)
            found_checkpoints.append((ep, st, filename))

    if found_checkpoints:
        # Sort the list. Python sorts tuples element-by-element.
        # So it sorts by Epoch first, then by Step.
        # We use reverse=True to get the highest epoch/step at index 0.
        found_checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        latest_ep, latest_st, latest_file = found_checkpoints[0]
        
        print(f"✅ Found latest checkpoint: {latest_file}")
        print(f"   (Epoch {latest_ep}, Step {latest_st})")
        
        resume_epoch = latest_ep
        resume_step = latest_st
        checkpoint_path = latest_file
        
        # We start IN the current epoch (to finish it)
        START_EPOCH = resume_epoch
        print(f"⏩ Will Fast-Forward to Epoch {resume_epoch}, Step {resume_step + 1}")

    else:
        # Fallback: Look for standard full-epoch saves if no step-saves exist
        print("No step-based checkpoints found. Looking for completed epoch models...")
        epoch_files = glob.glob(os.path.join("models", "bdd_model_epoch_*.pth"))
        # Also check local dir just in case
        if not epoch_files:
             epoch_files = glob.glob("bdd_model_epoch_*.pth")

        if epoch_files:
            # Sort by epoch number
            latest_file = max(epoch_files, key=lambda f: int(re.search(r"epoch_(\d+)", f).group(1)))
            print(f"✅ Found latest epoch model: {latest_file}")
            
            found_epoch = int(re.search(r"epoch_(\d+)", latest_file).group(1))
            
            # Since the epoch finished, we start from the NEXT one
            START_EPOCH = found_epoch + 1 
            resume_epoch = found_epoch + 1
            resume_step = -1 # Don't skip anything in the new epoch
            
            checkpoint_path = latest_file
        else:
            print("❌ No checkpoints found. Starting from scratch.")
            START_EPOCH = 0
            checkpoint_path = None

    # Load the model
    num_classes = 11
    model = get_model(num_classes)
    model.to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        # Handle loading: 'model_state_dict' vs full model
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print("Weights loaded successfully.")
    
    # Check if a step-based checkpoint exists if you crashed mid-epoch
    # e.g. "bdd_checkpoint_epoch_1_step_500.pth"
    # If you want to load that instead, change 'checkpoint_path' manually here.
    
    # 1. SETUP THE RESUME VARIABLES
    resume_epoch = 0
    resume_step = -1

    if checkpoint_path:
        # If we found a file like "...epoch_1_step_4000.pth"
        if "step" in checkpoint_path:
            match = re.search(r"epoch_(\d+)_step_(\d+)\.pth", checkpoint_path)
            resume_epoch = int(match.group(1))
            resume_step = int(match.group(2))
            
            # We want to start at the NEXT step, not redo step 4000
            START_EPOCH = resume_epoch
            print(f"⏩ Will Fast-Forward to Epoch {resume_epoch}, Step {resume_step + 1}")
            
        else:
            # It was a full epoch save (epoch_1.pth), so we start fresh on the next epoch
            match = re.search(r"epoch_(\d+)", checkpoint_path)
            resume_epoch = int(match.group(1))
            START_EPOCH = resume_epoch + 1
            resume_step = -1 # No steps to skip in the new epoch

    # Optimizer (Low LR for stability)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    print(f"Starting Epoch {START_EPOCH}...")

    # --- SAFETY: ENSURE SAVE DIRECTORIES EXIST ---
   
    os.makedirs("models/epochs", exist_ok=True)
    print("✅ Save directories verified.")
    
    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(train_loader):


            if epoch == resume_epoch and i <= resume_step:
                if i % 100 == 0:
                    print(f"⏩ Skipping processed batch {i}/{resume_step}...", end='\r')
                continue # Jump to the next iteration immediately

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward & Backward Pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Safety Check for NaN
            if not torch.isfinite(losses):
                print(f"WARNING: Loss is {losses.item()} at step {i}. Skipping.")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            losses.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()

            epoch_loss += losses.item()
            
            # Logging
            if i % 20 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {losses.item():.4f}")

            # --- NEW: SAVE EVERY 500 STEPS ---
            if i > 0 and i % SAVE_FREQUENCY == 0:
                step_filename = f"models/epochs/bdd_checkpoint_epoch_{epoch}_step_{i}.pth"
                print(f"💾 Saving backup: {step_filename} ...")
                torch.save(model.state_dict(), step_filename)

        # End of Epoch Save
        print(f"--- End of Epoch {epoch} | Avg Loss: {epoch_loss/len(train_loader):.4f} ---")
        torch.save(model.state_dict(), f"models/bdd_model_epoch_{epoch}.pth")

if __name__ == "__main__":
    train_resume()
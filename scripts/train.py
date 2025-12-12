import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import sys
import os
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T
import random


class BDD100K_IndividualFiles_Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, limit=None):
        """
        Args:
            image_dir (str): Path to folder with images (e.g., 'bdd100k/images/100k/val')
            label_dir (str): Path to folder with the 10,000 JSONs
            transform: PyTorch transforms
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # Get list of all images
        # We assume image 'abc.jpg' has a label 'abc.json'
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        
        # 1. Get all files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        
        # 2. Shuffle and Slice (The "30k" Logic)
        if limit is not None and limit < len(self.image_files):
            print(f"Subsetting: Selecting {limit} random images from {len(self.image_files)} total.")
            # We fix the random seed so every time you run this, you get the SAME 30k images
            # This ensures your experiments are reproducible.
            random.seed(42) 
            random.shuffle(self.image_files)
            self.image_files = self.image_files[:limit]
            
        print(f"Final Dataset Size: {len(self.image_files)} images.")
        
        # Verify a few matches exist to catch path errors early
        if len(self.image_files) > 0:
            test_img = self.image_files[0]
            test_json = test_img.replace('.jpg', '.json')
            if not os.path.exists(os.path.join(label_dir, test_json)):
                print(f"WARNING: Could not find label {test_json} in {label_dir}!")
                print("Make sure your image filenames match your json filenames.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        
        # 1. Load Image
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        
        # 2. Load the specific JSON file for this image
        json_filename = img_filename.replace('.jpg', '.json')
        json_path = os.path.join(self.label_dir, json_filename)
        
        boxes = []
        labels = []
        
        # BDD Class Mapping
        class_map = {
            "pedestrian": 1, "person": 1,"rider": 2, "car": 3, "truck": 4, 
            "bus": 5, "train": 6,"motor":7, "motorcycle": 7, "bike":8,"bicycle": 8,
            "traffic light": 9, "traffic sign": 10
        }

        # Handle case where label file might be missing
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Note: The structure inside individual files might vary.
            # Usually it's data['frames'][0]['objects'] OR just data['labels']
            # We try to accommodate common formats:
            objs = []
            if 'frames' in data:
                objs = data['frames'][0]['objects']
            elif 'labels' in data:
                objs = data['labels']
            
            for obj in objs:
                # Check for box2d and valid class
                category = obj.get('category', '')
                if 'box2d' in obj and category in class_map:
                    b = obj['box2d']
                    x1, y1, x2, y2 = float(b['x1']), float(b['y1']), float(b['x2']), float(b['y2'])
                    # box2d is usually x1, y1, x2, y2
                    # 1. Clip coordinates to image size to avoid out-of-bounds
                    w_img, h_img = image.size
                    x1 = max(0, min(x1, w_img))
                    y1 = max(0, min(y1, h_img))
                    x2 = max(0, min(x2, w_img))
                    y2 = max(0, min(y2, h_img))

                    # 2. Check for positive area
                    if (x2 > x1) and (y2 > y1):
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_map[category])
        
        # Convert to Tensor
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # Handle images with no objects (background only)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transform:
            image = self.transform(image)

        return image, target
    

def get_model(num_classes):
    # Load a pre-trained model (trained on COCO dataset)
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the "head" (the final layer) to match your number of classes
    # 1024 is the input size for the classifier in ResNet50
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# --- 3. Custom Collate Function ---
# This is REQUIRED because each image has a different number of objects
def collate_fn(batch):
    return tuple(zip(*batch))

# --- 4. Main Training Loop ---
def train():
    # A. Setup Hyperparameters
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on: {device}")
    
    num_classes = 11  # 10 BDD objects + 1 background class (always index 0)
    batch_size = 4    # Reduce this if you run out of GPU memory
    num_epochs = 3
    learning_rate = 0.005

    # B. Load Data
    # NOTE: Point these to your actual paths
    print("Loading data...")
    
    # Converting from PIL Image to Pytorch tensor 
    transform = T.Compose([
        T.ToTensor()
    ])
    
    dataset = BDD100K_IndividualFiles_Dataset(
        image_dir=r"C:\Users\gsamu\object_Detection_waymo\dataset\train_images",  # Using val as train for this demo
        label_dir=r"C:\Users\gsamu\object_Detection_waymo\annotations\train",
        transform=transform,
        limit=30000
    )
    
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collate_fn
    )

    # C. Load Model
    print("Initializing model...")
    model = get_model(num_classes)
    model.to(device)

    # D. Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    # E. Training Loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train() # Switch to training mode
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(data_loader):
            # Move data to the right device (GPU/CPU)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 1. Forward Pass
            # The model calculates the loss automatically when in training mode
            loss_dict = model(images, targets)
            
            # The model returns multiple losses (classifier, box regressor, objectness, etc.)
            # We sum them up for the final loss
            losses = sum(loss for loss in loss_dict.values())

            # 2. Backward Pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Logging
            epoch_loss += losses.item()
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {losses.item():.4f}")

        print(f"--- End of Epoch {epoch} | Avg Loss: {epoch_loss/len(data_loader):.4f} ---")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"models/bdd_model_epoch_{epoch}.pth")
        print("Model saved.")

if __name__ == "__main__":
    train()
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

class BDD100K_IndividualFiles_Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
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
        
        print(f"Found {len(self.image_files)} images.")
        
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
            "pedestrian": 1, "rider": 2, "car": 3, "truck": 4, 
            "bus": 5, "train": 6, "motorcycle": 7, "bicycle": 8,
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
                    # box2d is usually x1, y1, x2, y2
                    boxes.append([b['x1'], b['y1'], b['x2'], b['y2']])
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
    

def visualize_sample(dataset, idx=0):
    # 1. Get image and target
    image, target = dataset[idx]
    boxes = target['boxes']
    labels = target['labels']

    # 2. Setup Plot
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # 3. Draw Boxes
    # BDD boxes are [x1, y1, x2, y2]
    for box in boxes:
        x1, y1, x2, y2 = box.numpy()
        width = x2 - x1
        height = y2 - y1

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

    plt.title(f"Sample {idx}: Found {len(boxes)} Objects")
    plt.show()


    
    
# --- Quick Test Block ---
if __name__ == "__main__":
    # Update these paths to your folders
    dataset = BDD100K_IndividualFiles_Dataset(
        image_dir="../dataset/val_images/", 
        label_dir="../annotations/val/"  # Folder containing the 10k jsons
    )
    
    if len(dataset) > 0:
        img, target = dataset[0]
        print(f"Success! Image shape: {img.size}")
        print(f"Found {len(target['boxes'])} objects in first image.")
    
    visualize_sample(dataset, idx=0)
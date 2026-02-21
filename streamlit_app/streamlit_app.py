import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="AV Detection Benchmark", layout="wide")

# --- LOAD MODELS (Cached for Speed) ---
@st.cache_resource
def load_yolo():
    # Replace with your actual path
    return YOLO("models/best.pt") 

@st.cache_resource
def load_rcnn():
    # Setup Architecture
    num_classes = 11 # 10 objects + background
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load Weights
    # map_location='cpu' ensures it works even on non-GPU cloud instances
    checkpoint = torch.load("models/bdd_model_epoch_2.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# --- CLASSES ---
BDD_CLASSES = [
    "Background", "Pedestrian", "Rider", "Car", "Truck", 
    "Bus", "Train", "Motorcycle", "Bicycle", "Traffic Light", "Traffic Sign"
]

# --- INFERENCE FUNCTIONS ---
def run_yolo(model, image, conf_threshold):
    start = time.time()
    results = model.predict(image, conf=conf_threshold)
    end = time.time()
    latency = (end - start) * 1000
    
    # YOLOv8 has a built-in plot() function that returns a numpy array
    res_plotted = results[0].plot() 
    return res_plotted, latency

def run_rcnn(model, image, conf_threshold):
    # Preprocess
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0)
    
    start = time.time()
    with torch.no_grad():
        predictions = model(img_tensor)
    end = time.time()
    latency = (end - start) * 1000
    
    # Post-process predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    # Filter by threshold
    keep = scores >= conf_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    # Draw on Image using OpenCV
    img_np = np.array(image)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        class_name = BDD_CLASSES[label]
        
        # Draw Box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Draw Label
        cv2.putText(img_np, f"{class_name} {score:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    return img_np, latency

# --- SIDEBAR ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a Page", ["Project Overview", "Run Inference", "Research Results"])

# --- PAGE 1: OVERVIEW ---
if app_mode == "Project Overview":
    st.title("🚗 Speed vs. Safety in Autonomous Driving")
    st.markdown("""
    This project evaluates the trade-off between **Faster R-CNN** (Two-Stage) and **YOLOv8** (Single-Stage) 
    object detectors for autonomous vehicle perception.
    
    **Dataset:** [BDD100K (Berkeley DeepDrive)](https://bdd-data.berkeley.edu/)
    
    ### The Challenge
    Autonomous vehicles need to detect objects:
    1.  **Accurately:** To avoid collisions (Pedestrians, Cars).
    2.  **Fast:** To react in real-time (Latency < 30ms).
    
    ### The Models
    * **YOLOv8-Small:** Optimized for speed (Edge Devices).
    * **Faster R-CNN (ResNet-50):** Optimized for precision (Safety Critical).
    """)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/60/Autonomous_car_sensors.png", caption="AV Sensor Suite")

# --- PAGE 2: INFERENCE ---
elif app_mode == "Run Inference":
    st.title("⚡ Interactive Inference Demo")
    
    model_choice = st.sidebar.radio("Select Model", ["YOLOv8-Small", "Faster R-CNN"])
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    
    uploaded_file = st.file_uploader("Upload a Street View Image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Original Image", use_column_width=True)
        
        if st.button("Detect Objects"):
            with st.spinner("Running Detection..."):
                if model_choice == "YOLOv8-Small":
                    model = load_yolo()
                    result_img, lat = run_yolo(model, image, conf_thresh)
                else:
                    model = load_rcnn()
                    result_img, lat = run_rcnn(model, image, conf_thresh)
            
            st.success(f"Inference Complete! Latency: {lat:.2f} ms")
            st.image(result_img, caption=f"{model_choice} Detections", use_column_width=True)

# --- PAGE 3: RESULTS ---
elif app_mode == "Research Results":
    st.title("📊 Experimental Results")
    
    st.header("1. Speed vs. Accuracy Trade-off")
    st.markdown("YOLOv8 is ~40x faster, but Faster R-CNN provides better precision for vulnerable road users.")
    
    # You can hardcode your final results here for the table
    data = {
        "Metric": ["mAP@50 (Overall)", "Pedestrian AP", "Latency (GPU)", "FPS"],
        "Faster R-CNN": ["0.41", "0.601", "54 ms", "18"],
        "YOLOv8-Small": ["0.62", "0.441", "1.3 ms", "700+"]
    }
    st.table(data)
    
    st.header("2. Robustness (Day vs. Night)")
    st.markdown("Both models suffer performance degradation in low-light conditions.")
    # Add your Day/Night drop-off table here
    
    st.header("3. Failure Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Faster R-CNN Confusion Matrix**")
        # st.image("assets/rcnn_confusion.png") # Uncomment when you have the image
        st.info("Shows strong diagonal coherence.")
    with col2:
        st.markdown("**YOLOv8 Confusion Matrix**")
        # st.image("assets/yolo_confusion.png") # Uncomment when you have the image
        st.info("Shows slightly higher background confusion.")

    st.header("4. Qualitative 'Golden Image'")
    st.markdown("Faster R-CNN detecting a distant vehicle that YOLOv8 missed.")
    # st.image("assets/qualitative_comparison.png")
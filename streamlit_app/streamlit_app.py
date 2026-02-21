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
st.set_page_config(
    page_title="AV Detection Benchmark",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Root variables */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: #161d2e;
        --accent-primary: #00d4ff;
        --accent-secondary: #7c3aed;
        --accent-warning: #f59e0b;
        --text-primary: #f1f5f9;
        --text-muted: #64748b;
        --border: #1e2d40;
    }

    /* Global background */
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'DM Sans', sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    /* Logo / Title in sidebar */
    .sidebar-logo {
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--accent-primary);
        letter-spacing: 0.05em;
        padding: 0 1rem 0.5rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    .sidebar-logo span {
        color: var(--text-muted);
        font-weight: 400;
    }

    /* Navigation label */
    .nav-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        color: var(--text-muted);
        text-transform: uppercase;
        padding: 0 1.25rem;
        margin-bottom: 0.5rem;
    }

    /* Nav buttons */
    .stButton > button {
        width: 100%;
        background: transparent !important;
        border: 1px solid transparent !important;
        color: var(--text-muted) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        padding: 0.6rem 1.25rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        margin-bottom: 0.2rem !important;
    }
    .stButton > button:hover {
        background: rgba(0, 212, 255, 0.08) !important;
        color: var(--accent-primary) !important;
        border-color: rgba(0, 212, 255, 0.2) !important;
    }

    /* Active nav button */
    .nav-active .stButton > button {
        background: rgba(0, 212, 255, 0.12) !important;
        color: var(--accent-primary) !important;
        border-color: rgba(0, 212, 255, 0.3) !important;
    }

    /* Page title */
    .page-title {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    .page-subtitle {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .title-accent {
        color: var(--accent-primary);
    }

    /* Cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent-primary);
    }
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.25rem;
    }

    /* Info box */
    .info-box {
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: var(--text-primary);
    }

    /* Section header */
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 1rem;
        color: var(--accent-primary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem;
    }

    /* Table styling */
    .stTable table {
        background: var(--bg-card) !important;
        border-radius: 10px;
        overflow: hidden;
    }
    .stTable th {
        background: var(--bg-secondary) !important;
        color: var(--accent-primary) !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em !important;
    }
    .stTable td {
        color: var(--text-primary) !important;
        border-color: var(--border) !important;
    }

    /* Uploader */
    [data-testid="stFileUploader"] {
        background: var(--bg-card) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Slider label */
    .stSlider label {
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Radio button */
    .stRadio label {
        color: var(--text-primary) !important;
    }

    /* Latency badge */
    .latency-badge {
        display: inline-block;
        background: rgba(0, 212, 255, 0.15);
        border: 1px solid rgba(0, 212, 255, 0.4);
        color: var(--accent-primary);
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        margin-top: 0.5rem;
    }

    /* Caption text under images */
    .img-caption {
        text-align: center;
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
        font-family: 'Space Mono', monospace;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Divider */
    hr {
        border-color: var(--border) !important;
    }

    /* Streamlit text color overrides */
    p, li, .stMarkdown {
        color: var(--text-primary) !important;
    }
    h1, h2, h3 {
        color: var(--text-primary) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS (Cached for Speed) ---
@st.cache_resource
def load_yolo():
    return YOLO("models/best.pt")

@st.cache_resource
def load_rcnn():
    num_classes = 11
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load("models/bdd_model_epoch_2.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    # Warm up the model so first inference is faster
    dummy = torch.zeros(1, 3, 320, 320)
    with torch.no_grad():
        model([dummy[0]])
    return model

# --- CLASSES ---
BDD_CLASSES = [
    "Background", "Pedestrian", "Rider", "Car", "Truck",
    "Bus", "Train", "Motorcycle", "Bicycle", "Traffic Light", "Traffic Sign"
]

# Color palette per class (BGR for OpenCV)
CLASS_COLORS = {
    "Pedestrian":    (255, 80,  80),
    "Rider":         (255, 140, 0),
    "Car":           (0,   200, 255),
    "Truck":         (0,   160, 200),
    "Bus":           (80,  80,  255),
    "Train":         (160, 0,   255),
    "Motorcycle":    (255, 200, 0),
    "Bicycle":       (0,   220, 120),
    "Traffic Light": (0,   255, 80),
    "Traffic Sign":  (220, 220, 0),
    "Background":    (128, 128, 128),
}

# --- INFERENCE FUNCTIONS ---
def run_yolo(model, image, conf_threshold):
    start = time.time()
    results = model.predict(image, conf=conf_threshold, verbose=False)
    latency = (time.time() - start) * 1000
    res_plotted = results[0].plot()
    return res_plotted, latency

def run_rcnn(model, image, conf_threshold):
    # Resize image to speed up inference (max dimension 800)
    orig_size = image.size  # (W, H)
    max_dim = 800
    scale = min(max_dim / max(orig_size), 1.0)
    if scale < 1.0:
        new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
        image_infer = image.resize(new_size, Image.BILINEAR)
    else:
        image_infer = image

    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image_infer)

    start = time.time()
    with torch.no_grad():
        predictions = model([img_tensor])
    latency = (time.time() - start) * 1000

    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    # Filter by threshold
    keep = scores >= conf_threshold
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    # Scale boxes back to original image size if resized
    if scale < 1.0:
        boxes = boxes / scale

    # Draw on original image
    img_np = np.array(image.copy())
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        class_name = BDD_CLASSES[label] if label < len(BDD_CLASSES) else "Unknown"
        color = CLASS_COLORS.get(class_name, (200, 200, 200))

        # Box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        # Label background
        label_text = f"{class_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_np, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_np, label_text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)

    return img_np, latency

# --- SESSION STATE FOR NAVIGATION ---
if "page" not in st.session_state:
    st.session_state.page = "Overview"

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo">AV·DETECT <span>benchmark</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)

    pages = {
        "Overview":  "01 — Project Overview",
        "Inference": "02 — Run Inference",
        "Results":   "03 — Research Results",
    }
    for key, label in pages.items():
        css_class = "nav-active" if st.session_state.page == key else ""
        with st.container():
            if css_class:
                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            if st.button(label, key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()
            if css_class:
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.75rem; color:#374151; font-family:Space Mono,monospace; padding:0 1rem;">BDD100K · YOLOv8 · Faster R-CNN</p>', unsafe_allow_html=True)

app_mode = st.session_state.page

# ─────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────
if app_mode == "Overview":
    st.markdown('<div class="page-title">Speed vs. <span class="title-accent">Safety</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Benchmarking object detectors for autonomous vehicle perception on BDD100K</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("100K", "Images"),
        ("10", "Object Classes"),
        ("2", "Models Evaluated"),
        ("40×", "Speed Delta"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">The Challenge</div>', unsafe_allow_html=True)
    st.markdown("""
    Autonomous vehicles demand perception systems that satisfy two competing goals simultaneously:

    - **Accuracy** — correctly detect every vulnerable road user to prevent collisions
    - **Speed** — operate below 30 ms latency to enable real-time reaction

    This project systematically quantifies that trade-off using two representative architectures trained on the Berkeley DeepDrive (BDD100K) dataset.
    """)

    st.markdown('<div class="section-header">Models</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="info-box">
        <strong style="color:#00d4ff;">YOLOv8-Small</strong><br>
        Single-stage detector. Designed for edge devices with real-time constraints.
        Trades some precision for ~700 FPS throughput.
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="info-box">
        <strong style="color:#7c3aed;">Faster R-CNN (ResNet-50)</strong><br>
        Two-stage detector. Region-proposal network followed by classification.
        Higher precision on small/distant objects at ~18 FPS.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
    st.markdown("**BDD100K** — 100,000 driving images captured across diverse conditions: day/night, clear/rainy/foggy, highways and urban streets. Annotations cover 10 object categories relevant to autonomous driving.")

# ─────────────────────────────────────────────
# PAGE 2: INFERENCE
# ─────────────────────────────────────────────
elif app_mode == "Inference":
    st.markdown('<div class="page-title">Interactive <span class="title-accent">Inference</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload a street-view image and run detection with either model</div>', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        model_choice = st.radio("Model", ["YOLOv8-Small", "Faster R-CNN"], horizontal=True)
    with col_ctrl2:
        conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    if model_choice == "Faster R-CNN":
        st.markdown("""
        <div class="info-box" style="border-color:rgba(245,158,11,0.3); background:rgba(245,158,11,0.05);">
        ⚠️ <strong>CPU Inference:</strong> Faster R-CNN runs on CPU here and typically takes 5–20 seconds.
        Input is downscaled to max 800px for speed. Results are re-projected to full resolution.
        </div>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a Street-View Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        st.markdown('<div class="section-header">Input Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=False)

        if st.button("▶ Detect Objects", type="primary"):
            with st.spinner("Running detection…"):
                if model_choice == "YOLOv8-Small":
                    model = load_yolo()
                    result_img, lat = run_yolo(model, image, conf_thresh)
                else:
                    model = load_rcnn()
                    result_img, lat = run_rcnn(model, image, conf_thresh)

            st.markdown('<div class="section-header">Detection Results</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="latency-badge">⚡ {lat:.1f} ms — {model_choice}</div>', unsafe_allow_html=True)

            left, right = st.columns(2)
            with left:
                st.image(image, use_container_width=True)
                st.markdown('<div class="img-caption">Original</div>', unsafe_allow_html=True)
            with right:
                st.image(result_img, use_container_width=True)
                st.markdown(f'<div class="img-caption">{model_choice} Detections</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 3: RESULTS
# ─────────────────────────────────────────────
elif app_mode == "Results":
    st.markdown('<div class="page-title">Research <span class="title-accent">Results</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Quantitative benchmark on BDD100K validation set</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Speed vs. Accuracy Trade-off</div>', unsafe_allow_html=True)
    st.markdown("YOLOv8 is ~40× faster but Faster R-CNN achieves higher precision for vulnerable road users such as pedestrians.")

    data = {
        "Metric":        ["mAP@50 (Overall)", "Pedestrian AP", "Latency (GPU)", "Throughput (FPS)"],
        "Faster R-CNN":  ["0.41",             "0.601",         "54 ms",         "18"],
        "YOLOv8-Small":  ["0.62",             "0.441",         "1.3 ms",        "700+"],
    }
    st.table(data)

    st.markdown('<div class="section-header">Robustness — Day vs. Night</div>', unsafe_allow_html=True)
    st.markdown("Both models degrade in low-light conditions. Add your day/night AP drop-off table here.")

    st.markdown('<div class="section-header">Failure Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box" style="border-color:rgba(124,58,237,0.3); background:rgba(124,58,237,0.05);">
        <strong style="color:#7c3aed;">Faster R-CNN Confusion Matrix</strong><br><br>
        Strong diagonal coherence. Rare inter-class confusion.
        Main failure mode: missed detections on distant objects.
        </div>
        """, unsafe_allow_html=True)
        # st.image("assets/rcnn_confusion.png")
    with col2:
        st.markdown("""
        <div class="info-box" style="border-color:rgba(0,212,255,0.3); background:rgba(0,212,255,0.05);">
        <strong style="color:#00d4ff;">YOLOv8 Confusion Matrix</strong><br><br>
        Slightly higher background confusion, especially for small riders
        and partially occluded vehicles.
        </div>
        """, unsafe_allow_html=True)
        # st.image("assets/yolo_confusion.png")

    st.markdown('<div class="section-header">Qualitative "Golden Image"</div>', unsafe_allow_html=True)
    st.markdown("Faster R-CNN detecting a distant vehicle that YOLOv8 missed. Uncomment `st.image` below once assets are added.")
    # st.image("assets/qualitative_comparison.png")ader("4. Qualitative 'Golden Image'")
    st.markdown("Faster R-CNN detecting a distant vehicle that YOLOv8 missed.")
    # st.image("assets/qualitative_comparison.png")
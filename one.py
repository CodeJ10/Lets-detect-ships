import streamlit as st
import numpy as np
from PIL import Image
import time
from huggingface_hub import hf_hub_download

# ─── Page Config ─────────────────────────────
st.set_page_config(
    page_title="ShipSight — YOLOv8 Detection",
    page_icon="🚢",
    layout="wide",
)

# ─── CSS (Clean Dark UI) ─────────────────────
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* Header */
.title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #38bdf8;
}
.subtitle {
    color: #94a3b8;
    margin-bottom: 20px;
}

/* Card */
.card {
    background: #0f1629;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #1e293b;
}

/* Buttons */
.stButton button {
    background: #38bdf8;
    color: black;
    font-weight: bold;
    border-radius: 8px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1629;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model ─────────────────────────────
@st.cache_resource(show_spinner="Downloading model...")
def load_model():
    from ultralytics import YOLO
    weights = hf_hub_download(
        repo_id="CodeJ10/shipsight-yolov8",
        filename="best.pt"
    )
    return YOLO(weights)

model = load_model()

# ─── Sidebar ─────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    conf = st.slider("Confidence", 0.1, 1.0, 0.25)
    iou = st.slider("IoU Threshold", 0.1, 1.0, 0.45)

    st.success("Model loaded ✅")

# ─── Header ─────────────────────────────
st.markdown('<div class="title">🚢 ShipSight</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YOLOv8 Ship Detection from Satellite Images</div>', unsafe_allow_html=True)

# ─── Upload Section ─────────────────────────────
uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Detect Ships"):

        with st.spinner("Running detection..."):
            start = time.time()
            results = model.predict(image_np, conf=conf, iou=iou, verbose=False)
            end = time.time()

        result = results[0]
        annotated = result.plot()

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detection Result")
            st.image(annotated, use_container_width=True)

            st.markdown("---")

            # Metrics
            detections = len(result.boxes) if result.boxes else 0
            st.metric("Ships Detected", detections)
            st.metric("Time", f"{(end-start)*1000:.0f} ms")

            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Upload an image to begin 🚀")
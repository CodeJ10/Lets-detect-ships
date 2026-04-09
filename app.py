import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

# ─── Auto-download weights from Hugging Face ────────────────────────────────
@st.cache_resource(show_spinner="Downloading model weights...")
def get_weights():
    return hf_hub_download(
        repo_id="CodeJ10/shipsight-yolov8",  # ← change this
        filename="best.pt",
    )

HF_WEIGHTS_PATH = get_weights()

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShipSight — YOLOv8 Detection",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark navy theme */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1629 !important;
    border-right: 1px solid #1e2d4a;
}

/* Header */
.main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: -1px;
    margin-bottom: 0;
    line-height: 1.1;
}
.sub-header {
    font-size: 1rem;
    color: #64748b;
    font-weight: 300;
    margin-top: 4px;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #38bdf8; }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Detection box list */
.det-item {
    background: #0f1629;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.det-class { font-weight: 500; color: #e2e8f0; }
.det-conf {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #38bdf8;
    background: #172033;
    padding: 2px 8px;
    border-radius: 20px;
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed #1e2d4a !important;
    border-radius: 12px !important;
    background: #0f1629 !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8 !important;
}

/* Buttons */
.stButton > button {
    background: #38bdf8 !important;
    color: #0a0e1a !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Sliders */
[data-testid="stSlider"] > div > div { background: #38bdf8 !important; }

/* Section divider */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #38bdf8;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 8px;
    margin: 1.5rem 0 1rem;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-success { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.badge-warning { background: #1c1003; color: #fbbf24; border: 1px solid #854d0e; }
</style>
""", unsafe_allow_html=True)


# ─── Model loader ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    """Load YOLOv8 model — cached so it only loads once."""
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        return model, None
    except Exception as e:
        return None, str(e)


# ─── Inference helper ────────────────────────────────────────────────────────
def run_inference(model, image: np.ndarray, conf_thresh: float, iou_thresh: float):
    """Run YOLOv8 inference and return annotated image + detections list."""
    results = model.predict(
        source=image,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False,
    )
    result = results[0]

    # Annotated image
    annotated = result.plot()
    annotated_rgb = annotated

    # Parse detections
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": xyxy,
            })

    return annotated_rgb, detections


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown('<div class="section-title">Model Weights</div>', unsafe_allow_html=True)

    weights_source = st.radio(
        "Weights source",
        ["Upload .pt file", "Enter path"],
        label_visibility="collapsed",
    )

    model = None
    model_err = None

    if weights_source == "Upload .pt file":
        weights_file = st.file_uploader("Upload model weights (.pt)", type=["pt"])
        if weights_file:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            tmp.write(weights_file.read())
            tmp.flush()
            with st.spinner("Loading model..."):
                model, model_err = load_model(tmp.name)
    else:
        weights_path = st.text_input("Weights path", value="best.pt", placeholder="e.g. runs/detect/train/weights/best.pt")
        if st.button("Load model"):
            if os.path.exists(weights_path):
                with st.spinner("Loading model..."):
                    model, model_err = load_model(weights_path)
            else:
                model_err = f"File not found: {weights_path}"

    if model:
        st.markdown('<span class="badge badge-success">✓ Model loaded</span>', unsafe_allow_html=True)
        st.caption(f"Classes: {', '.join(model.names.values())}")
    elif model_err:
        st.markdown('<span class="badge badge-warning">⚠ Load error</span>', unsafe_allow_html=True)
        st.error(model_err, icon="🚨")

    st.markdown('<div class="section-title">Detection Settings</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05,
                            help="Minimum confidence to show a detection")
    iou_thresh = st.slider("IoU threshold (NMS)", 0.1, 1.0, 0.45, 0.05,
                           help="Higher = fewer overlapping boxes")

    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.caption("YOLOv8 Ship Detection — Final Year Project")
    st.caption("Model trained on satellite/aerial ship imagery.")


# ─── Main UI ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🚢 ShipSight</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">YOLOv8 Object Detection · Ships from Satellite Imagery</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "🎬 Video Detection", "📊 Model Info"])

# ── Tab 1: Image ──────────────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Drop a satellite / aerial image here",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supports JPG, PNG, BMP, TIFF",
    )

    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        image_rgb = np.array(pil_image)

        col_img, col_det = st.columns([3, 2])

        with col_img:
            st.markdown('<div class="section-title">Input Image</div>', unsafe_allow_html=True)
            st.image(pil_image, use_container_width=True)
            h, w = image_rgb.shape[:2]
            st.caption(f"Resolution: {w} × {h} px  ·  Size: {uploaded.size / 1024:.1f} KB")

        if not model:
            with col_det:
                st.warning("⬅️ Please load your model weights in the sidebar first.", icon="🔧")
        else:
            run_btn = st.button("🔍 Run Detection", use_container_width=False)
            if run_btn:
                with st.spinner("Detecting ships..."):
                    t0 = time.time()
                    annotated, detections = run_inference(model, image_rgb, conf_thresh, iou_thresh)
                    elapsed = (time.time() - t0) * 1000

                with col_img:
                    st.markdown('<div class="section-title">Detection Result</div>', unsafe_allow_html=True)
                    st.image(annotated, use_container_width=True)

                    # Download button
                    result_pil = Image.fromarray(annotated)
                    import io
                    buf = io.BytesIO()
                    result_pil.save(buf, format="PNG")
                    st.download_button(
                        "⬇️ Download result",
                        data=buf.getvalue(),
                        file_name="detection_result.png",
                        mime="image/png",
                    )

                with col_det:
                    st.markdown('<div class="section-title">Detection Summary</div>', unsafe_allow_html=True)

                    # Metric cards
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(detections)}</div><div class="metric-label">Ships found</div></div>', unsafe_allow_html=True)
                    with m2:
                        avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_conf:.0%}</div><div class="metric-label">Avg confidence</div></div>', unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{elapsed:.0f}ms</div><div class="metric-label">Inference time</div></div>', unsafe_allow_html=True)

                    st.markdown("")

                    if detections:
                        st.markdown('<div class="section-title">Detections</div>', unsafe_allow_html=True)
                        for i, det in enumerate(sorted(detections, key=lambda x: x["confidence"], reverse=True)):
                            st.markdown(
                                f'<div class="det-item">'
                                f'<span class="det-class">#{i+1} &nbsp; {det["class"].title()}</span>'
                                f'<span class="det-conf">{det["confidence"]:.1%}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        # Class breakdown
                        from collections import Counter
                        counts = Counter(d["class"] for d in detections)
                        if len(counts) > 1:
                            st.markdown('<div class="section-title">Class Breakdown</div>', unsafe_allow_html=True)
                            for cls, cnt in counts.most_common():
                                st.markdown(f"**{cls.title()}**: {cnt} detection{'s' if cnt > 1 else ''}")
                    else:
                        st.info("No ships detected at current confidence threshold. Try lowering it in the sidebar.")
    else:
        st.info("👆 Upload a satellite or aerial image to get started.", icon="🛰️")


# ── Tab 2: Video ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Video / Frame-by-frame Detection")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if video_file and model:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix)
        tfile.write(video_file.read())

        cap = Image.VideoCapture(tfile.name)
        total_frames = int(cap.get(Image.CAP_PROP_FRAME_COUNT))
        fps = cap.get(Image.CAP_PROP_FPS)
        cap.release()

        st.info(f"Video: {total_frames} frames · {fps:.1f} FPS · {total_frames/fps:.1f}s")
        frame_step = st.slider("Process every N frames", 1, 10, 2,
                                help="Higher = faster but may miss some detections")

        if st.button("▶️ Process Video"):
            cap = Image.VideoCapture(tfile.name)
            frame_placeholder = st.empty()
            progress = st.progress(0)
            stats_placeholder = st.empty()

            frame_idx = 0
            total_det = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_step == 0:
                    frame_rgb = Image.cvtColor(frame, Image.COLOR_BGR2RGB)
                    annotated, dets = run_inference(model, frame_rgb, conf_thresh, iou_thresh)
                    total_det += len(dets)
                    frame_placeholder.image(annotated, use_container_width=True, caption=f"Frame {frame_idx}")
                    stats_placeholder.caption(f"Frame {frame_idx}/{total_frames} · {total_det} total detections so far")
                progress.progress(min(frame_idx / max(total_frames, 1), 1.0))
                frame_idx += 1

            cap.release()
            st.success(f"✅ Done! Processed {frame_idx} frames · {total_det} total detections.")
    elif video_file and not model:
        st.warning("Please load your model weights in the sidebar first.")
    else:
        st.info("Upload a video file to run frame-by-frame detection.", icon="🎬")


# ── Tab 3: Model Info ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Model Information")
    if model:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Architecture**")
            st.code(str(type(model.model).__name__))
            st.markdown("**Detected Classes**")
            for idx, name in model.names.items():
                st.markdown(f"- `{idx}` → {name}")
        with col2:
            st.markdown("**Model Parameters**")
            try:
                params = sum(p.numel() for p in model.model.parameters())
                st.metric("Total parameters", f"{params/1e6:.1f}M")
            except:
                st.caption("Parameter count unavailable")
            st.markdown("**Inference Settings**")
            st.json({"conf_threshold": conf_thresh, "iou_threshold": iou_thresh})
    else:
        st.info("Load a model in the sidebar to see its details here.", icon="🔧")

    st.markdown("---")
    st.markdown("#### Project Details")
    st.markdown("""
| Field | Detail |
|---|---|
| Model | YOLOv8 (Ultralytics) |
| Task | Object Detection |
| Domain | Satellite / Aerial Ship Detection |
| Framework | Streamlit + OpenCV |
| Dataset | Ships dataset (Kaggle) |
    """)
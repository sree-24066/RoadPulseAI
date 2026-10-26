import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

from utils.model_loader import load_model, CLASSES, CLASS_COLORS
from utils.detector import run_detection

st.set_page_config(
    page_title="Image Detection Â· RoadPulse AI",
    page_icon="ðŸ“·",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€ Shared CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background-color: #0A0A0F; color: #E8E8F0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0F0F1A,#12121A); border-right:1px solid #1E1E2E; }
.sidebar-brand { font-size:1.3rem;font-weight:800;background:linear-gradient(90deg,#00D4FF,#7B2FBE);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;padding:0.8rem 0 0.5rem 0; }
.page-title { font-size:2.2rem;font-weight:800;background:linear-gradient(90deg,#00D4FF,#7B2FBE);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }
.page-sub { color:#6666AA;font-size:0.95rem;margin-bottom:1.5rem; }
.glow-divider { border:none;height:1px;background:linear-gradient(90deg,transparent,#00D4FF44,transparent);margin:1.2rem 0; }
.det-table { width:100%;border-collapse:collapse;font-size:0.85rem; }
.det-table th { background:#1A1A2A;color:#8888AA;padding:8px 12px;text-align:left;border-bottom:1px solid #1E1E2E; }
.det-table td { padding:8px 12px;border-bottom:1px solid #14141E;color:#E8E8F0; }
.det-table tr:hover td { background:#14142A; }
.badge { display:inline-block;border-radius:6px;padding:2px 10px;font-size:0.78rem;font-weight:600; }
.upload-hint { background:#12121A;border:1.5px dashed #1E1E2E;border-radius:12px;padding:1.2rem;color:#444466;text-align:center;font-size:0.88rem; }
</style>
""", unsafe_allow_html=True)

# â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.markdown('<div class="sidebar-brand">âš¡ RoadPulse AI</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.page_link("ðŸ _Home.py",                       label="ðŸ   Home")
    st.page_link("pages/1_ðŸ“·_Image_Detection.py",    label="ðŸ“·  Image Detection")
    st.page_link("pages/2_ðŸŽ¬_Video_Detection.py",    label="ðŸŽ¬  Video Detection")
    st.page_link("pages/3_ðŸ“¡_Realtime_Detection.py", label="ðŸ“¡  Realtime Detection")
    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;color:#444466;">Model: YOLOv8-Small Â· CRDDC2022</div>', unsafe_allow_html=True)

# â”€â”€ Page header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown('<div class="page-title">ðŸ“· Image Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Upload a road image to detect and classify damage zones instantly.</div>', unsafe_allow_html=True)
st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

# â”€â”€ Controls â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
col_ctrl1, col_ctrl2 = st.columns([3, 1])
with col_ctrl1:
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if image_file is None:
        st.markdown('<div class="upload-hint">ðŸ“‚ Drag & drop or click to upload a road image (PNG / JPG)</div>', unsafe_allow_html=True)
with col_ctrl2:
    score_threshold = st.slider("Confidence", min_value=0.05, max_value=1.0, value=0.25, step=0.05, help="Lower = more detections, Higher = fewer but more confident")

# â”€â”€ Inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if image_file is not None:
    with st.spinner("ðŸ” Analyzing imageâ€¦"):
        net = load_model()
        image = Image.open(image_file).convert("RGB")
        _image = np.array(image)
        detections, annotated = run_detection(net, _image, score_threshold)

    st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

    # â”€â”€ Side-by-side results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("**ðŸ–¼ï¸ Original Image**")
        st.image(_image, use_container_width=True)
    with col2:
        st.markdown("**ðŸŽ¯ Predictions**")
        st.image(annotated, use_container_width=True)

        # Download
        buffer = BytesIO()
        Image.fromarray(annotated).save(buffer, format="PNG")
        st.download_button(
            label="â¬‡ï¸ Download Prediction",
            data=buffer.getvalue(),
            file_name="RoadPulseAI_Prediction.png",
            mime="image/png",
            use_container_width=True,
        )

    st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

    # â”€â”€ Detection table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if detections:
        st.markdown(f"**ðŸ”Ž {len(detections)} Detection(s) Found**")
        rows = ""
        for i, d in enumerate(detections, 1):
            color = CLASS_COLORS.get(d.label, "#00D4FF")
            badge = f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}55;">{d.label}</span>'
            rows += f"<tr><td>{i}</td><td>{badge}</td><td>{d.score:.1%}</td><td>{d.box[0]}, {d.box[1]}, {d.box[2]}, {d.box[3]}</td></tr>"
        st.markdown(f"""
        <table class="det-table">
            <thead><tr><th>#</th><th>Class</th><th>Confidence</th><th>BBox (x1,y1,x2,y2)</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
    else:
        st.info("âœ… No damage detected at this confidence level. Try lowering the threshold.")


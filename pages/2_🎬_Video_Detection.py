import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import cv2
import numpy as np
import streamlit as st

from utils.model_loader import load_model, CLASSES, CLASS_COLORS
from utils.detector import run_detection

st.set_page_config(
    page_title="Video Detection · RoadPulse AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background-color: #0A0A0F; color: #E8E8F0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0F0F1A,#12121A); border-right:1px solid #1E1E2E; }
.sidebar-brand { font-size:1.3rem;font-weight:800;background:linear-gradient(90deg,#00D4FF,#7B2FBE);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;padding:0.8rem 0 0.5rem 0; }
.page-title { font-size:2.2rem;font-weight:800;background:linear-gradient(90deg,#00D4FF,#7B2FBE);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }
.page-sub { color:#6666AA;font-size:0.95rem;margin-bottom:1.5rem; }
.glow-divider { border:none;height:1px;background:linear-gradient(90deg,transparent,#00D4FF44,transparent);margin:1.2rem 0; }
.info-card { background:#12121A;border:1px solid #1E1E2E;border-radius:10px;padding:0.8rem 1.2rem;font-size:0.88rem;color:#8888AA; }
.info-card b { color:#E8E8F0; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚡ RoadPulse AI</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.page_link("🏠_Home.py",                       label="🏠  Home")
    st.page_link("pages/1_📷_Image_Detection.py",    label="📷  Image Detection")
    st.page_link("pages/2_🎬_Video_Detection.py",    label="🎬  Video Detection")
    st.page_link("pages/3_📡_Realtime_Detection.py", label="📡  Realtime Detection")
    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;color:#444466;">Model: YOLOv8-Small · CRDDC2022</div>', unsafe_allow_html=True)

st.markdown('<div class="page-title">🎬 Video Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Upload an MP4 video to run frame-by-frame road damage detection and download the annotated output.</div>', unsafe_allow_html=True)
st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

# ── Temp folder ────────────────────────────────────────────────────────────────
TEMP_DIR = Path(__file__).parent.parent / "temp"
TEMP_DIR.mkdir(exist_ok=True)
TEMP_INPUT  = str(TEMP_DIR / "video_input.mp4")
TEMP_OUTPUT = str(TEMP_DIR / "video_output.mp4")

# ── State ──────────────────────────────────────────────────────────────────────
if "video_running" not in st.session_state:
    st.session_state.video_running = False

# ── Controls ───────────────────────────────────────────────────────────────────
col_upload, col_conf = st.columns([3, 1])
with col_upload:
    video_file = st.file_uploader("Upload Video (.mp4)", type=["mp4"], disabled=st.session_state.video_running)
with col_conf:
    score_threshold = st.slider("Confidence", 0.05, 1.0, 0.25, 0.05, disabled=st.session_state.video_running)

# ── Process ────────────────────────────────────────────────────────────────────
if video_file is not None and not st.session_state.video_running:
    if st.button("▶️ Process Video", use_container_width=True, type="primary"):
        st.session_state.video_running = True
        st.rerun()

if video_file is not None and st.session_state.video_running:
    net = load_model()

    # Write to disk
    with open(TEMP_INPUT, "wb") as f:
        f.write(video_file.getbuffer())

    cap = cv2.VideoCapture(TEMP_INPUT)
    if not cap.isOpened():
        st.error("❌ Could not open the video file.")
        st.session_state.video_running = False
    else:
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = int(total / fps) if fps > 0 else 0

        st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)
        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.markdown(f'<div class="info-card"><b>{W}×{H}</b><br>Resolution</div>', unsafe_allow_html=True)
        info_col2.markdown(f'<div class="info-card"><b>{fps:.1f}</b><br>FPS</div>', unsafe_allow_html=True)
        info_col3.markdown(f'<div class="info-card"><b>{duration_s}s</b><br>Duration</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        progress_bar = st.progress(0, text="🔍 Processing frames…")
        frame_preview = st.empty()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(TEMP_OUTPUT, fourcc, fps, (W, H))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, annotated = run_detection(net, frame_rgb, score_threshold)
            out_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            writer.write(out_bgr)

            frame_count += 1
            pct = frame_count / total if total > 0 else 1
            progress_bar.progress(min(pct, 1.0), text=f"🔍 Frame {frame_count}/{total}")

            if frame_count % 5 == 0:
                frame_preview.image(annotated, caption=f"Frame {frame_count}", use_container_width=True)

        cap.release()
        writer.release()
        progress_bar.empty()
        frame_preview.empty()
        st.session_state.video_running = False

        st.success("✅ Video processing complete!")
        st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            with open(TEMP_OUTPUT, "rb") as f:
                st.download_button("⬇️ Download Annotated Video", f, "RoadPulseAI_Video.mp4", "video/mp4", use_container_width=True, type="primary")
        with btn_col2:
            if st.button("🔄 Process Another Video", use_container_width=True):
                st.rerun()

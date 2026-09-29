import sys
import queue
import logging
from pathlib import Path
from typing import List
sys.path.insert(0, str(Path(__file__).parent.parent))

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from utils.model_loader import load_model, CLASSES, CLASS_COLORS
from utils.detector import Detection

st.set_page_config(
    page_title="Realtime Detection · RoadPulse AI",
    page_icon="📡",
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
.live-badge { display:inline-block;background:#FF3B3B22;color:#FF3B3B;border:1px solid #FF3B3B55;border-radius:20px;padding:3px 14px;font-size:0.78rem;font-weight:700;letter-spacing:0.05em; }
.det-count { font-size:2.5rem;font-weight:800;color:#00D4FF;text-align:center; }
.det-label { font-size:0.75rem;color:#6666AA;text-transform:uppercase;letter-spacing:0.1em;text-align:center; }
.badge { display:inline-block;border-radius:6px;padding:2px 10px;font-size:0.78rem;font-weight:600; }
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

st.markdown('<div class="page-title">📡 Realtime Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Stream your webcam for live road damage detection. Works great for field inspection and on-the-go monitoring.</div>', unsafe_allow_html=True)
st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

logger = logging.getLogger(__name__)
net = load_model()
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

score_threshold = st.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05,
                            help="Adjust before starting the stream")

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    h_ori, w_ori = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold, verbose=False)

    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
            Detection(
                class_id=int(_box.cls.item()),
                label=CLASSES[int(_box.cls.item())],
                score=float(_box.conf.item()),
                box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]
        result_queue.put(detections)

    annotated = results[0].plot()
    annotated_bgr = cv2.cvtColor(
        cv2.resize(annotated, (w_ori, h_ori), interpolation=cv2.INTER_AREA),
        cv2.COLOR_RGB2BGR
    )
    return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)
left_col, right_col = st.columns([3, 1], gap="large")

with left_col:
    st.markdown('<span class="live-badge">● LIVE</span>', unsafe_allow_html=True)
    webrtc_ctx = webrtc_streamer(
        key="roadpulse-realtime",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": {"width": {"ideal": 1280}}, "audio": False},
        async_processing=True,
    )

with right_col:
    st.markdown("**Live Stats**")
    count_box   = st.empty()
    class_box   = st.empty()

    if webrtc_ctx.state.playing:
        count_box.markdown('<div class="det-count">—</div><div class="det-label">Detections</div>', unsafe_allow_html=True)
        while True:
            try:
                results_list = result_queue.get(timeout=1)
                count = len(results_list)
                count_box.markdown(f'<div class="det-count">{count}</div><div class="det-label">Detections</div>', unsafe_allow_html=True)
                if results_list:
                    badges = ""
                    for d in results_list:
                        color = CLASS_COLORS.get(d.label, "#00D4FF")
                        badges += f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}55;margin:3px 2px;">{d.label}<br><small>{d.score:.0%}</small></span>'
                    class_box.markdown(badges, unsafe_allow_html=True)
                else:
                    class_box.markdown('<div style="color:#444466;font-size:0.85rem;">No damage in frame</div>', unsafe_allow_html=True)
            except queue.Empty:
                pass
    else:
        count_box.markdown('<div class="det-count">—</div><div class="det-label">Detections</div>', unsafe_allow_html=True)
        class_box.markdown('<div style="color:#444466;font-size:0.85rem;">Start the stream to see live results.</div>', unsafe_allow_html=True)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="GPS Damage Map · RoadPulse AI",
    page_icon="🗺️",
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
.stat-card { background:#12121A;border:1px solid #1E1E2E;border-radius:12px;padding:1rem 1.2rem;text-align:center; }
.stat-num  { font-size:2rem;font-weight:800;color:#00D4FF; }
.stat-label{ font-size:0.78rem;color:#6666AA;text-transform:uppercase;letter-spacing:0.08em; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚡ RoadPulse AI</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.page_link("🏠_Home.py",                        label="🏠  Home")
    st.page_link("pages/1_📷_Image_Detection.py",     label="📷  Image Detection")
    st.page_link("pages/2_🎬_Video_Detection.py",     label="🎬  Video Detection")
    st.page_link("pages/3_📡_Realtime_Detection.py",  label="📡  Realtime Detection")
    st.page_link("pages/4_🗺️_GPS_Damage_Map.py",      label="🗺️  GPS Damage Map")
    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;color:#444466;">Model: YOLOv8-Small · CRDDC2022</div>', unsafe_allow_html=True)

st.markdown('<div class="page-title">🗺️ GPS Damage Map</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Interactive OpenStreetMap showing all road damage detections logged across sessions. Click any marker for details.</div>', unsafe_allow_html=True)
st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

from utils.geo_map import render_geo_input, render_map_section, get_records

lat = float(st.session_state.get("rp_lat", 0.0))
lon = float(st.session_state.get("rp_lon", 0.0))

lat, lon = render_geo_input(key_suffix="mappage")

st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

records = get_records()
if not records:
    st.info("🗺️ No damage records yet. Run detections on the Image, Video, or Realtime pages and click **Plot on Map** to populate this view.")

render_map_section(lat, lon, map_key="standalone_map")
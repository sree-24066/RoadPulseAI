import streamlit as st

st.set_page_config(
    page_title="RoadPulse AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Base ---- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0A0F;
    color: #E8E8F0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F0F1A 0%, #12121A 100%);
    border-right: 1px solid #1E1E2E;
}

/* ---- Hero section ---- */
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00D4FF 0%, #7B2FBE 50%, #FF6B6B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1.15rem;
    color: #8888AA;
    margin-bottom: 2rem;
}

/* ---- Feature cards ---- */
.card {
    background: #12121A;
    border: 1px solid #1E1E2E;
    border-radius: 16px;
    padding: 1.6rem 1.4rem;
    transition: border-color 0.25s, box-shadow 0.25s;
    height: 100%;
}
.card:hover {
    border-color: #00D4FF44;
    box-shadow: 0 0 24px #00D4FF18;
}
.card-icon { font-size: 2.4rem; margin-bottom: 0.6rem; }
.card-title { font-size: 1.1rem; font-weight: 700; color: #E8E8F0; margin-bottom: 0.4rem; }
.card-desc  { font-size: 0.88rem; color: #6666AA; line-height: 1.5; }

/* ---- Damage badges ---- */
.badge {
    display: inline-block;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 4px 4px 4px 0;
}

/* ---- Stat cards ---- */
.stat-card {
    background: #12121A;
    border: 1px solid #1E1E2E;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-num  { font-size: 2rem; font-weight: 800; color: #00D4FF; }
.stat-label{ font-size: 0.78rem; color: #6666AA; text-transform: uppercase; letter-spacing: 0.08em; }

/* ---- Sidebar brand ---- */
.sidebar-brand {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D4FF, #7B2FBE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding: 0.8rem 0 0.5rem 0;
}

/* ---- Divider ---- */
.glow-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00D4FF44, transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚡ RoadPulse AI</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Navigate**")
    st.page_link("🏠_Home.py",                   label="🏠  Home")
    st.page_link("pages/1_📷_Image_Detection.py", label="📷  Image Detection")
    st.page_link("pages/2_🎬_Video_Detection.py", label="🎬  Video Detection")
    st.page_link("pages/3_📡_Realtime_Detection.py", label="📡  Realtime Detection")
    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;color:#444466;">Model: YOLOv8-Small · CRDDC2022</div>', unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">RoadPulse AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Intelligent road damage detection powered by YOLOv8 deep learning</div>', unsafe_allow_html=True)
st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-num">4</div><div class="stat-label">Damage Types</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="stat-num">YOLOv8</div><div class="stat-label">Model Architecture</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-num">640px</div><div class="stat-label">Input Resolution</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-card"><div class="stat-num">Real‑time</div><div class="stat-label">Detection Speed</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature cards ─────────────────────────────────────────────────────────────
st.markdown("### ⚡ Detection Modes")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">📷</div>
        <div class="card-title">Image Detection</div>
        <div class="card-desc">Upload a road image and instantly detect damage zones with bounding boxes and confidence scores.</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🎬</div>
        <div class="card-title">Video Detection</div>
        <div class="card-desc">Process full MP4 videos frame-by-frame. Download the annotated output video when complete.</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="card">
        <div class="card-icon">📡</div>
        <div class="card-title">Realtime Detection</div>
        <div class="card-desc">Live webcam stream with real-time detection overlay. Great for field inspection use cases.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

# ── Damage types ──────────────────────────────────────────────────────────────
st.markdown("### 🔍 Detectable Damage Types")
badge_html = """
<span class="badge" style="background:#FF6B6B22;color:#FF6B6B;border:1px solid #FF6B6B55;">🔴 Longitudinal Crack</span>
<span class="badge" style="background:#FFD93D22;color:#FFD93D;border:1px solid #FFD93D55;">🟡 Transverse Crack</span>
<span class="badge" style="background:#6BCB7722;color:#6BCB77;border:1px solid #6BCB7755;">🟢 Alligator Crack</span>
<span class="badge" style="background:#00D4FF22;color:#00D4FF;border:1px solid #00D4FF55;">🔵 Potholes</span>
"""
st.markdown(badge_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
> **Model Info:** Trained on the [CRDDC2022](https://crddc2022.sekilab.global/) Japan & India dataset using YOLOv8-Small architecture.  
> Optimized for real-world road inspection scenarios.
""")

st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="font-size:0.8rem;color:#444466;text-align:center;">
    Built by <a href="https://github.com/sree-24066" style="color:#00D4FF;text-decoration:none;">@sree-24066</a> · 
    Powered by <a href="https://github.com/ultralytics/ultralytics" style="color:#7B2FBE;text-decoration:none;">Ultralytics YOLOv8</a> & 
    <a href="https://streamlit.io" style="color:#FF6B6B;text-decoration:none;">Streamlit</a>
</div>
""", unsafe_allow_html=True)

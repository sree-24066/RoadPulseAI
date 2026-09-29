from pathlib import Path
from ultralytics import YOLO
import streamlit as st

HERE = Path(__file__).parent
ROOT = HERE.parent
MODEL_LOCAL_PATH = ROOT / "models" / "YOLOv8_Small_RDD.pt"

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

CLASS_COLORS = {
    "Longitudinal Crack": "#FF6B6B",
    "Transverse Crack":   "#FFD93D",
    "Alligator Crack":    "#6BCB77",
    "Potholes":           "#00D4FF",
}

@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_LOCAL_PATH)

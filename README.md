# ⚡ RoadPulse AI

**Intelligent Road Damage Detection powered by YOLOv8**

A modern, dark-themed web application that detects and classifies road damage in real-time using deep learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)

### 🌍 **[Try the Live Web App](https://roadpulseai.streamlit.app)** 🌍

---

## 🔍 Features

| Mode | Description |
|------|-------------|
| 📷 **Image Detection** | Upload a road image and instantly detect damage zones |
| 🎬 **Video Detection** | Process MP4 videos frame-by-frame with downloadable output |
| 📡 **Realtime Detection** | Live webcam stream with real-time bounding box overlay |

## 🧠 Model

- **Architecture:** YOLOv8-Small
- **Dataset:** CRDDC2022 (Japan & India)
- **Classes:** Longitudinal Crack · Transverse Crack · Alligator Crack · Potholes

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/sree-24066/RoadPulseAI.git
cd RoadPulseAI

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run 🏠_Home.py
```

Then open **http://localhost:8501** in your browser.

## 📁 Project Structure

```
RoadPulseAI/
├── 🏠_Home.py                  # Main landing page
├── pages/
│   ├── 1_📷_Image_Detection.py
│   ├── 2_🎬_Video_Detection.py
│   └── 3_📡_Realtime_Detection.py
├── utils/
│   ├── model_loader.py         # Shared model loading
│   └── detector.py             # Detection logic
├── models/
│   └── YOLOv8_Small_RDD.pt    # Pre-trained model
├── .streamlit/
│   └── config.toml             # Dark theme config
└── requirements.txt
```

## 🙏 Credits

- Model trained on [CRDDC2022](https://crddc2022.sekilab.global/) dataset
- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) & [Streamlit](https://streamlit.io)

---

**Built by [@sree-24066](https://github.com/sree-24066)**

# streamlit_app.py

import streamlit as st
from streamlit.components.v1 import html
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="PlatIQ – Pengenalan Plat Kendaraan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# 1. Load & inject your full index.html (UI + CSS + JS) here
# ------------------------------------------------------------
with open("index.html", "r", encoding="utf-8") as f:
    page = f.read()

# Height disesuaikan agar seluruh halaman bisa di-scroll
html(page, height=800, scrolling=True)

# ------------------------------------------------------------
# 2. Persiapkan model deteksi (cache agar sekali load)
# ------------------------------------------------------------
@st.cache_resource
def load_detector():
    # Ganti path ke file best.pt sesuai lokasi model Anda
    return YOLO("best.pt")

detector = load_detector()

# ------------------------------------------------------------
# 3. Sidebar untuk memilih mode input
# ------------------------------------------------------------
st.sidebar.title("Demo Deteksi Plat")
mode = st.sidebar.radio("Pilih mode input", ["Upload Gambar", "Kamera"])

CONFIDENCE = 0.4

if mode == "Upload Gambar":
    upload = st.sidebar.file_uploader("Pilih file gambar (.jpg/.png)", type=["jpg", "jpeg", "png"])
    if upload:
        img = Image.open(upload).convert("RGB")
        img_np = np.array(img)
        
        st.image(img, caption="Gambar Asli", use_column_width=True)
        
        # inference
        res = detector.predict(img_np, conf=CONFIDENCE)[0]
        annotated = res.plot()  # numpy array
        
        st.image(annotated, caption="Hasil Deteksi", use_column_width=True)
        
        # tampilkan teks plat dan confidence setiap box
        st.markdown("*Detail Deteksi:*")
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            st.write(f"- Box: ({x1},{y1}) → ({x2},{y2}), Confidence: {conf*100:.1f}%")
            
elif mode == "Kamera":
    cam = st.sidebar.camera_input("Ambil foto dengan kamera")
    if cam:
        img = Image.open(cam).convert("RGB")
        img_np = np.array(img)
        
        st.image(img, caption="Foto Kamera", use_column_width=True)
        
        res = detector.predict(img_np, conf=CONFIDENCE)[0]
        annotated = res.plot()
        
        st.image(annotated, caption="Hasil Deteksi", use_column_width=True)
        st.markdown("*Detail Deteksi:*")
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            st.write(f"- Box: ({x1},{y1}) → ({x2},{y2}), Confidence: {conf*100:.1f}%")

# ------------------------------------------------------------
# 4. Footer kecil di bawah
# ------------------------------------------------------------
st.markdown(
    """
    ---
    *PlatIQ* – Sistem Pengenalan Plat Kendaraan  •  Powered by Streamlit & YOLOv8  
    ”Teknologi AI untuk identifikasi kendaraan dengan akurasi tinggi.”  
    """,
    unsafe_allow_html=True
)

import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from ultralytics import YOLO

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(page_title="Dashboard Pengolahan Citra", page_icon="📊", layout="wide")

# =====================
# LOAD MODELS
# =====================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Aditya Arrahman_Laporan 4.pt")
    buah_model = tf.keras.models.load_model("model/Aditya Arrahman_Laporan_2.h5")
    return yolo_model, buah_model

yolo_model, buah_model = load_models()
class_names = ["Apel", "Anggur", "Mangga", "Pisang", "Stroberi"]

# =====================
# CUSTOM CSS (warna teks sidebar putih bersih)
# =====================
st.markdown("""
<style>
/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(to right, #fdfcfb, #e2d1c3);
    font-family: 'Poppins', sans-serif;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #1e293b;
    color: #ffffff;
}

/* Judul sidebar */
section[data-testid="stSidebar"] h2 {
    color: #ffffff !important;
    text-align: center;
    font-weight: 700;
}

/* Label radio (semua teks putih) */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #ffffff !important;
    opacity: 1 !important;
    transition: color 0.3s ease, border-left 0.3s ease;
    padding-left: 8px;
}

/* Warna aktif */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"]:has(input[checked]) {
    color: #fb923c !important;
    font-weight: 600;
    border-left: 4px solid #fb923c;
}

/* Hover efek */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    color: #facc15 !important;
}

/* TITLE */
h1, h2, h3 {
    font-weight: 700;
    color: #1e293b;
}

/* FEATURE CARD */
.feature-card {
    background-color: #fff6ed;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    transition: transform 0.2s ease, box-shadow 0.3s ease;
}
.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 8px 18px rgba(0,0,0,0.25);
}

/* BUTTON */
.stButton > button {
    background-color: #fb923c;
    color: white;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    transition: 0.3s ease;
}
.stButton > button:hover {
    background-color: #f97316;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# =====================
# NAVIGASI (radio terhubung ke session_state lewat key="page")
# =====================
PAGES = ["Dashboard", "Klasifikasi Buah", "Deteksi Ikan"]

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

def go_to(page_name):
    st.session_state.page = page_name

# =====================
# SIDEBAR
# =====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/706/706164.png", width=80)
    st.markdown("<h2>Pengolahan Citra</h2>", unsafe_allow_html=True)
    st.radio("Pilih Halaman:", PAGES, key="page")

page = st.session_state.page

# =====================
# DASHBOARD
# =====================
if page == "Dashboard":
    st.markdown("<h1 style='text-align:center;'>📊 Dashboard Pengolahan Citra</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px;'>Gunakan aplikasi ini untuk melakukan klasifikasi buah dan deteksi ikan menggunakan model Machine Learning.</p>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🧭 Panduan Penggunaan")
    st.markdown("""
    1. **Klasifikasi Buah 🍎**  
       Upload gambar buah seperti Apel, Anggur, Mangga, Pisang, atau Stroberi.  
       Model CNN akan mengenali jenis buah berdasarkan gambar tersebut.

    2. **Deteksi Ikan 🐟**  
       Upload gambar ikan (misalnya Ikan Mas Koki).  
       Model YOLOv8 akan menampilkan posisi ikan dengan kotak deteksi otomatis.
    """)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🔍 Pilih Fitur yang Ingin Digunakan")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3>🍎 Klasifikasi Buah</h3>
            <p>Klasifikasikan jenis buah berdasarkan gambar yang diunggah menggunakan model CNN.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Masuk ke Klasifikasi Buah", on_click=go_to, args=("Klasifikasi Buah",))

    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3>🐟 Deteksi Ikan</h3>
            <p>Deteksi posisi ikan secara otomatis menggunakan model YOLOv8.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Masuk ke Deteksi Ikan", on_click=go_to, args=("Deteksi Ikan",))

# =====================
# KLASIFIKASI BUAH
# =====================
elif page == "Klasifikasi Buah":
    st.markdown("<h2 style='text-align:center;'>🍎 Klasifikasi Buah</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload gambar buah:", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang Diunggah", use_container_width=True)

        def klasifikasi_buah(img, model, class_names):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds, axis=1)[0]
            pred_label = class_names[pred_idx]
            conf = preds[0][pred_idx]
            return pred_label, conf

        label, conf = klasifikasi_buah(img, buah_model, class_names)
        st.success(f"Hasil Klasifikasi: **{label}** ({conf*100:.2f}%)")

# =====================
# DETEKSI IKAN
# =====================
elif page == "Deteksi Ikan":
    st.markdown("<h2 style='text-align:center;'>🐟 Deteksi Ikan</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload gambar ikan:", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang Diunggah", use_container_width=True)

        results = yolo_model.predict(img)
        result = results[0]
        annotated_img = result.plot()[:, :, ::-1]  # BGR -> RGB
        st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)

        st.markdown("### Deteksi Ditemukan:")
        if len(result.boxes) == 0:
            st.write("Tidak ada objek terdeteksi.")
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = yolo_model.names[cls]
            st.write(f"- **{label}** ({conf*100:.2f}%)")

import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from ultralytics import YOLO

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Dashboard Pengolahan Citra",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
# NAVIGASI
# =====================
PAGES = ["Dashboard", "Klasifikasi Buah", "Deteksi Ikan"]
NAV_LABELS = {"Dashboard": "Beranda", "Klasifikasi Buah": "Klasifikasi Buah", "Deteksi Ikan": "Deteksi Ikan"}

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

def go_to(page_name):
    st.session_state.page = page_name

page = st.session_state.page
active_idx = PAGES.index(page)

# =====================
# VIDEO BACKGROUND + EFEK
# =====================
VIDEO_URL = "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260405_170732_8a9ccda6-5cff-4628-b164-059c500a2b41.mp4"

st.markdown(f"""
<video class="hero-video" autoplay loop muted playsinline src="{VIDEO_URL}"></video>
<div class="hero-noise"></div>
<div class="hero-gradient"></div>
""", unsafe_allow_html=True)

# =====================
# CSS
# =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root { --cream: #E1E0CC; --cream-80: rgba(225,224,204,0.8); --cream-70: rgba(225,224,204,0.7); }

/* Dasar */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0a0a0a !important;
    font-family: 'Inter', sans-serif;
}
header[data-testid="stHeader"], footer, #MainMenu,
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }

.block-container {
    position: relative;
    z-index: 2;
    max-width: 100% !important;
    padding: 0 2.5rem 1.5rem 2.5rem !important;
}

/* Video, noise, gradient */
.hero-video {
    position: fixed; inset: 0; width: 100vw; height: 100vh;
    object-fit: cover; z-index: -3;
}
.hero-noise {
    position: fixed; inset: 0; pointer-events: none; z-index: -2;
    opacity: 0.7; mix-blend-mode: overlay;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='3' stitchTiles='stitch'/></filter><rect width='100%' height='100%' filter='url(%23n)' opacity='0.5'/></svg>");
}
.hero-gradient {
    position: fixed; inset: 0; pointer-events: none; z-index: -1;
    background: linear-gradient(to bottom, rgba(0,0,0,0.30), transparent 45%, rgba(0,0,0,0.60));
}

/* NAVBAR (pil hitam di tengah atas) */
.st-key-navbar {
    position: fixed; top: 0; left: 50%; transform: translateX(-50%);
    z-index: 50;
    display: flex !important; flex-direction: row !important; align-items: center;
    gap: 2.5rem !important;
    background: #000; padding: 0.5rem 2rem;
    border-radius: 0 0 1.5rem 1.5rem;
    width: auto !important;
}
.st-key-navbar button {
    background: transparent !important; border: none !important; box-shadow: none !important;
    color: var(--cream-80) !important; padding: 0.1rem 0 !important; min-height: 0 !important;
    font-size: 0.9rem; transition: color .25s ease;
}
.st-key-navbar button p { color: inherit !important; font-size: 0.9rem; }
.st-key-navbar button:hover { color: var(--cream) !important; }

/* Judul besar */
.pull-title {
    display: inline-flex; flex-wrap: wrap;
    font-weight: 500; line-height: 0.85; letter-spacing: -0.07em;
    color: var(--cream);
}
.pull-title.hero { font-size: clamp(5rem, 20vw, 22rem); }
.pull-title.page { font-size: clamp(2.6rem, 7vw, 6.5rem); padding-top: 5.5rem; }
.pull-word {
    display: inline-block; position: relative; opacity: 0;
    margin-right: 0.25em;
    animation: pullUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
.pull-word:last-child { margin-right: 0; }
.pull-word .ast { position: absolute; top: 0.65em; right: -0.3em; font-size: 0.31em; }
@keyframes pullUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }

.hero-desc {
    color: var(--cream-70); font-size: 1rem; line-height: 1.2; margin: 0 0 1.25rem 0;
    animation: pullUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.5s both;
}

/* Tata letak hero: rata bawah seperti grid 12 kolom */
div[data-testid="stHorizontalBlock"] { align-items: flex-end; }

/* Tombol CTA pil dengan lingkaran panah */
.st-key-cta_buah button, .st-key-cta_ikan button {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: var(--cream) !important; color: #000 !important;
    border: none !important; border-radius: 999px !important;
    padding: 4px 4px 4px 1.25rem !important; font-weight: 500;
    transition: gap .25s ease;
}
.st-key-cta_buah button p, .st-key-cta_ikan button p { color: #000 !important; font-weight: 500; }
.st-key-cta_buah button::after, .st-key-cta_ikan button::after {
    content: "→"; display: grid; place-items: center;
    width: 2.5rem; height: 2.5rem; border-radius: 50%;
    background: #000; color: var(--cream); font-size: 1.1rem;
    transition: transform .25s ease;
}
.st-key-cta_buah button:hover, .st-key-cta_ikan button:hover { gap: 0.9rem; transform: none !important; }
.st-key-cta_buah button:hover::after, .st-key-cta_ikan button:hover::after { transform: scale(1.1); }
.st-key-cta_ikan button { background: transparent !important; box-shadow: inset 0 0 0 1px var(--cream-70) !important; }
.st-key-cta_ikan button p { color: var(--cream) !important; }

/* Halaman dalam */
.stApp p, .stApp label, .stApp li, .stApp span, .stApp h1, .stApp h2, .stApp h3 { color: var(--cream); }
.page-desc { color: var(--cream-70); max-width: 42rem; line-height: 1.3; margin: 1rem 0 1.5rem 0; }
[data-testid="stFileUploaderDropzone"] {
    background: rgba(0,0,0,0.55) !important; backdrop-filter: blur(8px);
    border: 1px dashed rgba(225,224,204,0.4) !important; border-radius: 1.25rem !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--cream) !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--cream) !important; border: none !important; border-radius: 999px !important; }
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button * { color: #000 !important; }
[data-testid="stImage"] img { border-radius: 1.25rem; }
[data-testid="stAlert"] { background: rgba(0,0,0,0.6) !important; border-radius: 1rem; backdrop-filter: blur(8px); }
.result-card {
    background: rgba(0,0,0,0.6); backdrop-filter: blur(8px);
    border-radius: 1.25rem; padding: 1rem 1.5rem; color: var(--cream);
}
</style>
""", unsafe_allow_html=True)

# CSS dinamis: tandai menu yang sedang aktif
st.markdown(f"""
<style>
.st-key-nav_{active_idx} button, .st-key-nav_{active_idx} button p {{ color: var(--cream) !important; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# =====================
# NAVBAR
# =====================
with st.container(key="navbar"):
    for i, p in enumerate(PAGES):
        st.button(NAV_LABELS[p], key=f"nav_{i}", on_click=go_to, args=(p,))

# =====================
# HELPER: animasi kata naik
# =====================
def pull_up(text, css_class="page", asterisk=False):
    words = text.split(" ")
    spans = ""
    for i, w in enumerate(words):
        ast = "<span class='ast'>*</span>" if asterisk and i == len(words) - 1 else ""
        spans += f"<span class='pull-word' style='animation-delay:{i*0.08:.2f}s'>{w}{ast}</span>"
    return f"<div class='pull-title {css_class}'>{spans}</div>"

# =====================
# HALAMAN
# =====================
if page == "Dashboard":
    st.markdown("<div style='height:34vh'></div>", unsafe_allow_html=True)
    col_title, col_info = st.columns([8, 4], gap="large")

    with col_title:
        st.markdown(pull_up("Citra", "hero", asterisk=True), unsafe_allow_html=True)

    with col_info:
        st.markdown(
            "<p class='hero-desc'>Dashboard pengolahan citra untuk mengenali jenis buah dengan model CNN "
            "dan mendeteksi posisi ikan dengan YOLOv8, cukup dari gambar yang kamu unggah.</p>",
            unsafe_allow_html=True,
        )
        st.button("Klasifikasi Buah", key="cta_buah", on_click=go_to, args=("Klasifikasi Buah",))
        st.button("Deteksi Ikan", key="cta_ikan", on_click=go_to, args=("Deteksi Ikan",))
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

elif page == "Klasifikasi Buah":
    st.markdown(pull_up("Klasifikasi Buah"), unsafe_allow_html=True)
    st.markdown(
        "<p class='page-desc'>Upload gambar buah (Apel, Anggur, Mangga, Pisang, atau Stroberi). "
        "Model CNN akan mengenali jenis buahnya.</p>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload gambar buah:", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")

        def klasifikasi_buah(img, model, class_names):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds, axis=1)[0]
            return class_names[pred_idx], preds[0][pred_idx]

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.image(img, caption="Gambar yang Diunggah", use_container_width=True)
        with c2:
            label, conf = klasifikasi_buah(img, buah_model, class_names)
            st.markdown(
                f"<div class='result-card'><p style='margin:0;opacity:.7'>Hasil Klasifikasi</p>"
                f"<p style='margin:0;font-size:3rem;font-weight:500;letter-spacing:-0.04em'>{label}</p>"
                f"<p style='margin:0;opacity:.7'>Keyakinan {conf*100:.2f}%</p></div>",
                unsafe_allow_html=True,
            )

elif page == "Deteksi Ikan":
    st.markdown(pull_up("Deteksi Ikan"), unsafe_allow_html=True)
    st.markdown(
        "<p class='page-desc'>Upload gambar ikan (misalnya Ikan Mas Koki). "
        "Model YOLOv8 akan menampilkan posisi ikan dengan kotak deteksi otomatis.</p>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload gambar ikan:", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        results = yolo_model.predict(img)
        result = results[0]
        annotated_img = result.plot()[:, :, ::-1]  # BGR -> RGB

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.image(img, caption="Gambar yang Diunggah", use_container_width=True)
        with c2:
            st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)

        items = ""
        if len(result.boxes) == 0:
            items = "<p style='margin:0'>Tidak ada objek terdeteksi.</p>"
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            items += f"<p style='margin:0.25rem 0'><b>{yolo_model.names[cls]}</b> ({conf*100:.2f}%)</p>"
        st.markdown(
            f"<div class='result-card'><p style='margin:0 0 .5rem 0;opacity:.7'>Deteksi Ditemukan</p>{items}</div>",
            unsafe_allow_html=True,
        )

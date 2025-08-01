import streamlit as st
from ui.main import show_main_page
from ui.guide import show_guide
from ui.detection import show_detection_page
from ui.history import show_history
from config import configure_gemini_api
from function.models import ObjectDetectionModel
from function.database import init_db
import os

# --- Load External CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Make sure your style.css is in an 'assets' folder relative to your main script
# For example, if your main script is in the root, and style.css is in 'assets/style.css'
css_file_path = os.path.join(os.getcwd(), 'assets', 'style.css')
load_css(css_file_path)
# --- End Load External CSS ---


# Konfigurasi API Gemini
GEMINI_CONFIGURED = configure_gemini_api()

# Setup Model YOLO
model = ObjectDetectionModel(os.path.join(os.getcwd(), 'weights', 'best.pt'))

# Setup database
conn = init_db()

# Streamlit UI Setup
st.set_page_config(
    page_title="Deteksi Penyakit Daun Singkong",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar UI for navigation
st.sidebar.title("Navigasi")

# Tombol untuk setiap halaman
if st.sidebar.button('Halaman Utama', key='home', use_container_width=True):
    st.session_state.page = "home"

if st.sidebar.button('Halaman Panduan', key='guide', use_container_width=True):
    st.session_state.page = "guide"

if st.sidebar.button('Halaman Deteksi', key='detection', use_container_width=True):
    st.session_state.page = "detection"

if st.sidebar.button('Histori Deteksi', key='history', use_container_width=True):
    st.session_state.page = "history"

# Pastikan jika tidak ada page yang dipilih, default ke "home"
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Menampilkan halaman sesuai dengan pilihan di sidebar
if st.session_state.page == "home":
    show_main_page()
elif st.session_state.page == "guide":
    show_guide()
elif st.session_state.page == "detection":
    show_detection_page(conn, model, GEMINI_CONFIGURED)
elif st.session_state.page == "history":
    show_history(conn)
    
    
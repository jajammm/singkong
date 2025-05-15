import streamlit as st
import sqlite3
from datetime import datetime
import io
import cv2
import av
import PIL.Image as Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import settings
from ultralytics import YOLO
import google.generativeai as genai
import os
from google import generativeai as genai
from fpdf import FPDF
from datetime import datetime
import tempfile

# Konfigurasi logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_markdown(text):
    """Membersihkan format markdown dari teks untuk output PDF."""
    # Menghapus format bold (tanda bintang ganda)
    text = text.replace('**', '')
    
    # Menghapus format italic (tanda bintang tunggal)
    text = text.replace('*', '')
    
    # Menghapus format markdown lainnya jika diperlukan
    text = text.replace('#', '')  # Menghapus tanda pagar untuk heading
    text = text.replace('`', '')  # Menghapus backtick untuk code
    
    return text

def create_detection_pdf(image, label, confidence, explanation):
    try:
        # Inisialisasi FPDF
        pdf = FPDF()
        pdf.add_page()
        
        # Judul PDF
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(190, 10, 'Hasil Deteksi Penyakit Daun Singkong', 0, 1, 'C')
        pdf.ln(10)
        
        # Tambahkan waktu deteksi
        pdf.set_font('Arial', '', 12)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        pdf.cell(190, 10, f'Waktu Deteksi: {current_time}', 0, 1)
        pdf.ln(5)
        
        # Informasi deteksi
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, f'Penyakit Terdeteksi: {label}', 0, 1)
        pdf.cell(190, 10, f'Tingkat Kepercayaan: {confidence:.2f}', 0, 1)
        pdf.ln(5)
        
        # Simpan gambar sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_filename = temp_file.name
            image.save(temp_filename)
        
        # Tambahkan gambar ke PDF
        pdf.cell(190, 10, 'Gambar Daun Singkong:', 0, 1)
        pdf.image(temp_filename, x=10, y=None, w=180)
        
        # Hapus file gambar sementara
        os.unlink(temp_filename)
        
        # Tambahkan penjelasan
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(190, 10, 'Analisis dan Rekomendasi:', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        explanation_lines = explanation.split('\n')

        # Proses penjelasan baris per baris
        current_mode = 'normal'
        for line in explanation_lines:
            # Bersihkan format markdown
            clean_line = clean_markdown(line)
            
            # Deteksi judul section (misalnya "PENJELASAN:", "DAMPAK:")
            if "PENJELASAN:" in clean_line or "DAMPAK:" in clean_line or "REKOMENDASI" in clean_line:
                pdf.ln(5)
                pdf.set_font('Arial', 'B', 12)  # Bold untuk judul
                current_mode = 'title'
            elif clean_line.strip() == "":
                pdf.ln(5)
                pdf.set_font('Arial', '', 11)
                current_mode = 'normal'
            else:
                if current_mode == 'title':
                    pdf.set_font('Arial', '', 11)
                    current_mode = 'normal'
            
            # Tulis baris bersih ke PDF
            pdf.multi_cell(0, 6, clean_line)
            
        # Simpan PDF ke file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
            temp_pdf_filename = temp_pdf_file.name
            pdf.output(temp_pdf_filename)
        
        # Baca file PDF sebagai bytes
        with open(temp_pdf_filename, 'rb') as f:
            pdf_data = f.read()
        
        # Hapus file PDF sementara
        os.unlink(temp_pdf_filename)
        
        return pdf_data
        
    except Exception as e:
        st.error(f"Error saat membuat PDF: {str(e)}")
        return None

try:
    # Ambil API key dari Streamlit secrets
    gemini_api_key = st.secrets["gemini"]["api_key"]
    genai.configure(api_key=gemini_api_key)
    GEMINI_CONFIGURED = True
except Exception as e:
    GEMINI_CONFIGURED = False
    logger.error(f"Error konfigurasi Gemini API: {str(e)}")

# Fungsi untuk menggunakan Gemini API
def get_disease_explanation(disease_label):
    if not GEMINI_CONFIGURED:
        return "API Gemini belum terkonfigurasi dengan benar. Periksa file secrets.toml Anda."
    
    try:
        # Buat prompt untuk Gemini
        prompt = f"""
        Berikan penjelasan detail tentang penyakit daun singkong "{disease_label}" dengan format berikut (langsung jelaskan saja tanpa harus mengiyakan perintah saya):
        
        PENJELASAN:
        [Jelaskan gejala dan penyebab penyakit pada daun singkong tersebut secara detail. Jika daun singkong sehat/healty tidak perlu menjelaskan gejala dan penybabnya, namun jelaskan ciri-cirinya saja]
        
        DAMPAK:
        [Jelaskan dampak penyakit ini terhadap tanaman singkong]
        
        REKOMENDASI PENANGANAN:
        [Berikan 3-5 rekomendasi penanganan yang bisa dilakukan petani]
        """
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        return f"Error mendapatkan penjelasan: {str(e)}"
    
if 'detection_boxes' not in st.session_state:
    st.session_state.detection_boxes = None
if 'detection_model' not in st.session_state:
    st.session_state.detection_model = None
if 'detection_confidence' not in st.session_state:
    st.session_state.detection_confidence = None

# Function to check login credentials
def check_login(username, password):
    return username == "admin" and password == "password"

# Function to save detection result to database
def save_detection(image):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    conn.execute("INSERT INTO detections (timestamp, image) VALUES (?, ?)", (timestamp, img_byte_arr))
    conn.commit()

# Function to load detection history from database
def load_detection_history():
    c = conn.cursor()
    c.execute("SELECT id, timestamp, image FROM detections ORDER BY timestamp DESC")
    return c.fetchall()

# Function to delete all detection history
def delete_all_detections():
    c = conn.cursor()
    c.execute("DELETE FROM detections")
    conn.commit()

# Initialize SQLite database
conn = sqlite3.connect('detection_cassava_leaves.db', check_same_thread=False)
conn.execute('''CREATE TABLE IF NOT EXISTS detections
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp TEXT,
              image BLOB)''')
conn.commit()

# Model class for object detection
class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        try:
            self.model = self.load_model()
            self.confidence = 0.3
            self.detected_labels = []
        except Exception as e:
            logger.error(f"Error loading model in VideoTransformer: {str(e)}")
            st.error(f"Failed to load model: {str(e)}")
            self.model = None
    
    def load_model(self):
        try:
            # Log the model path for debugging
            model_path = settings.DETECTION_MODEL
            logger.info(f"Loading model from path: {model_path}")
            
            # Check if file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at: {model_path}")
                raise FileNotFoundError(f"Model file not found at: {model_path}")
                
            # Monkey patch torch.load to handle the weights_only issue
            import torch
            from functools import partial
            
            # Store original torch.load function
            original_torch_load = torch.load
            
            # Create a patched version that explicitly sets weights_only=False
            def patched_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            # Temporarily replace torch.load with our patched version
            torch.load = patched_torch_load
            
            try:
                # Attempt to load model with our patched torch.load
                model = YOLO(model_path, task='detect')
                logger.info("Model loaded successfully with patched torch.load")
                return model
            finally:
                # Restore original torch.load function
                torch.load = original_torch_load
                
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        if self.model is None:
            # Return original frame if model loading failed
            return frame
            
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Perform object detection
            results = self.model(img, stream=True)
            
            # Clear previous detected labels
            self.detected_labels = []
            
            # Draw bounding boxes on the frame
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0].cpu().numpy()  # get box coordinates in (top, left, bottom, right) format
                    c = box.cls
                    conf = box.conf.item()
                    if conf >= self.confidence:
                        x1, y1, x2, y2 = map(int, b)
                        label = f"{self.model.names[int(c)]} {conf:.2f}"
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Add to detected labels list
                        if self.model.names[int(c)] not in self.detected_labels:
                            self.detected_labels.append(self.model.names[int(c)])
                            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            # Draw error text on the frame
            cv2.putText(img, f"Detection error: {str(e)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Setting page layout
st.set_page_config(
    page_title="Deteksi Penyakit Daun Singkong",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Login interface
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.success("Login successful")
        else:
            st.error("Invalid username or password")
else:
    # Main page heading
    st.title("Deteksi Penyakit Daun Singkong")

    # Placeholder for detection history
    history_placeholder = st.empty()

    # Sidebar
    st.sidebar.header("Detection")
    
    st.sidebar.header("API Status")
    if GEMINI_CONFIGURED:
        st.sidebar.success("✅ Gemini API terkonfigurasi dengan benar")
    else:
        st.sidebar.error("❌ Gemini API tidak terkonfigurasi. Periksa file secrets.toml")

    # Model Options
    confidence = float(st.sidebar.slider(
        "Select Model Confidence (%)", 25, 100, 30)) / 100

    # Model path for Detection
    model_path = settings.DETECTION_MODEL

    # Load Pre-trained ML Model
    try:
        st.sidebar.text(f"Loading model from: {model_path}")
        
        # Verifikasi model ada
        if not os.path.exists(model_path):
            st.sidebar.error(f"Model file tidak ditemukan di path: {model_path}")
            model = None
        else:
            # Gunakan safe_load untuk memuat model
            model = YOLO(model_path)
            st.sidebar.success("Model berhasil dimuat!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        model = None
        logger.error(f"Failed to load YOLO model: {str(e)}")

    st.sidebar.header("Image/Video Config")
    source_radio = st.sidebar.radio(
        "Select Source", [settings.IMAGE, settings.WEBCAM])

    source_img = None
    # If image is selected
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
        
        # Tombol deteksi tetap di sidebar
        detect_button = st.sidebar.button('Detect Objects')

        # Buat dua kolom untuk gambar input dan hasil deteksi
        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    if os.path.exists(default_image_path):
                        default_image = Image.open(default_image_path)
                        st.image(default_image_path, caption="Default Image",
                                use_column_width=True)
                    else:
                        st.error(f"Default image not found at: {default_image_path}")
                else:
                    uploaded_image = Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image",
                            use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)
                logger.error(f"Error opening image: {ex}")

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                if os.path.exists(default_detected_image_path):
                    default_detected_image = Image.open(default_detected_image_path)
                    st.image(default_detected_image_path, caption='Detected Image',
                            use_column_width=True)
                else:
                    st.error(f"Default detected image not found at: {default_detected_image_path}")
            else:
                # Proses deteksi ketika tombol di sidebar diklik
                if detect_button:
                    if model is not None:
                        try:
                            res = model.predict(uploaded_image,
                                            conf=confidence
                                            )
                            boxes = res[0].boxes
                            res_plotted = res[0].plot()[:, :, ::-1]
                            detected_image = Image.fromarray(res_plotted)
                            st.image(res_plotted, caption='Detected Image',
                                    use_column_width=True)
                            save_detection(detected_image)
                            
                            # Simpan hasil deteksi untuk ditampilkan di luar kolom
                            st.session_state.detection_boxes = boxes
                            st.session_state.detection_model = model
                            st.session_state.detection_confidence = confidence
                        except Exception as e:
                            st.error(f"Error during detection: {str(e)}")
                            logger.error(f"Detection error: {str(e)}")
                    else:
                        st.error("Model belum dimuat dengan benar. Tidak dapat melakukan deteksi.")
        
        # Buat kontainer baru dengan lebar penuh untuk hasil deteksi
        if source_img is not None and detect_button and 'detection_boxes' in st.session_state and st.session_state.detection_boxes is not None:
            st.markdown("---")
            st.header("Hasil Deteksi")
            
            boxes = st.session_state.detection_boxes
            model = st.session_state.detection_model
            confidence = st.session_state.detection_confidence
            
            for box in boxes:
                label = model.names[int(box.cls)]
                conf = box.conf.item()
                if conf >= confidence:
                    with st.container():
                        st.subheader(f"Deteksi: {label} (Confidence: {conf:.2f})")
                        
                        # Dapatkan penjelasan dari Gemini
                        if GEMINI_CONFIGURED:
                            with st.spinner(f"Mendapatkan penjelasan untuk {label}..."):
                                explanation = get_disease_explanation(label)
                                st.markdown(explanation)
                                
                                # Tambahkan tombol download PDF
                                col1, col2 = st.columns([1, 6])
                                with col1:
                                    pdf_data = create_detection_pdf(detected_image, label, conf, explanation)
                                    if pdf_data:
                                        filename = f"deteksi_{label.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                        st.download_button(
                                            label="📥 Download Hasil Deteksi",
                                            data=pdf_data,
                                            file_name=filename,
                                            mime="application/pdf",
                                            key=f"download_{label}_{conf}"
                                        )
                        else:
                            st.warning("Gemini API tidak terkonfigurasi. Periksa file secrets.toml")
                        
                        # Tambahkan pemisah untuk setiap hasil deteksi
                        st.markdown("---")

    elif source_radio == settings.WEBCAM:
        st.header("WebRTC Object Detection")
        
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_processor_factory=VideoTransformer,
            async_processing=True,
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.confidence = confidence
            
            # Tampilkan penjelasan untuk label yang terdeteksi
            if hasattr(webrtc_ctx.video_processor, 'detected_labels') and webrtc_ctx.video_processor.detected_labels:
                st.markdown("---")
                st.header("Hasil Deteksi")
                
                if GEMINI_CONFIGURED:
                    for label in webrtc_ctx.video_processor.detected_labels:
                        # Gunakan container dengan lebar penuh untuk hasil deteksi
                        with st.container():
                            st.subheader(f"Deteksi: {label}")
                            with st.spinner(f"Mendapatkan penjelasan untuk {label}..."):
                                explanation = get_disease_explanation(label)
                                st.markdown(explanation)
                            
                            # Opsi untuk menyimpan gambar dan penjelasan
                            col1, col2, col3 = st.columns([1, 1, 4])
                            
                            with col1:
                                if st.button(f"Simpan ke database", key=f"save_{label}"):
                                    if webrtc_ctx.video_frame_buffer and len(webrtc_ctx.video_frame_buffer) > 0:
                                        img_array = webrtc_ctx.video_frame_buffer[-1].to_ndarray(format="bgr24")
                                        pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                                        save_detection(pil_img)
                                        st.success("Hasil deteksi berhasil disimpan!")
                            
                            with col2:
                                # Tambahkan tombol download PDF untuk webcam
                                if webrtc_ctx.video_frame_buffer and len(webrtc_ctx.video_frame_buffer) > 0:
                                    img_array = webrtc_ctx.video_frame_buffer[-1].to_ndarray(format="bgr24")
                                    pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                                    
                                    pdf_data = create_detection_pdf(pil_img, label, 0.0, explanation)  # Confidence tidak diketahui untuk webcam
                                    if pdf_data:
                                        filename = f"deteksi_webcam_{label.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                        st.download_button(
                                            label="📥 Download PDF",
                                            data=pdf_data,
                                            file_name=filename,
                                            mime="application/pdf",
                                            key=f"download_webcam_{label}"
                                        )
                            
                            # Tambahkan pemisah untuk setiap hasil deteksi
                            st.markdown("---")

    else:
        st.error("Please select a valid source type!")

    # Detection history and delete button
    if st.sidebar.button('Detection History'):
        with history_placeholder.container():
            st.header("Detection History")
            if st.button("Close History"):
                history_placeholder.empty()
            
            # Ambil history dari database
            history = load_detection_history()
            
            # Periksa apakah history kosong
            if not history or len(history) == 0:
                st.info("Belum ada history deteksi. Silakan lakukan deteksi terlebih dahulu.")
            else:
                # Tampilkan history jika ada
                st.success(f"Ditemukan {len(history)} hasil deteksi.")
                for id, timestamp, image in history:
                    try:
                        image = Image.open(io.BytesIO(image))
                        with st.expander(f"ID: {id}, Time: {timestamp}"):
                            st.image(image, caption="Detected Image", use_column_width=False, width=500)
                    except Exception as e:
                        st.error(f"Error menampilkan gambar ID: {id}: {str(e)}")

    # Delete all history
    if st.sidebar.button('Delete All History'):
        delete_all_detections()
        st.sidebar.success("All detection history has been deleted.")
        # Update the history display
        history_placeholder.empty()
        with history_placeholder.container():
            st.header("Detection History")
            st.info("No detection history available.")
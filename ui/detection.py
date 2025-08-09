import streamlit as st
from function.database import save_detection, load_detection_history
from config import get_disease_explanation
from function.pdf_utils import create_detection_pdf
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
import settings

def show_detection_page(conn, model, GEMINI_CONFIGURED):
    """Halaman Deteksi dan Menampilkan Histori Deteksi"""
    st.title("Deteksi Penyakit Daun Singkong")

    # Pilihan sumber input: hanya unggah gambar
    st.sidebar.markdown("### Unggah Gambar")
    source_img = st.sidebar.file_uploader("Pilih Gambar...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    confidence = st.sidebar.slider("Pilih Tingkat Kepercayaan Model (Opsional)", 0.0, 1.0, 0.3)
    detect_button = st.sidebar.button('Deteksi Objek')
    

    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_img is None:
                st.markdown('<h3>Gambar Asli</h3>', unsafe_allow_html=True)
                default_image_path = "images/original.png"
                default_image = Image.open(default_image_path)
                st.image(default_image, use_container_width=True)
            else:
                st.markdown('<h3>Gambar Asli</h3>', unsafe_allow_html=True)
                uploaded_image = Image.open(source_img)
                st.image(source_img, caption="Uploaded Image", use_container_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            st.markdown('<h3>Gambar Hasil Deteksi</h3>', unsafe_allow_html=True)
            default_detected_image_path = "images/hasil.png"
            default_detected_image = Image.open(default_detected_image_path)
            st.image(default_detected_image, use_container_width=True)
        else:
            if detect_button:
                st.markdown('<h3>Gambar Hasil Deteksi</h3>', unsafe_allow_html=True)
                results = model.predict(uploaded_image)
                
                # Ambil semua kotak deteksi dari hasil
                all_boxes = results[0].boxes
                
                # Filter kotak berdasarkan confidence threshold
                detected_boxes_above_confidence = [box for box in all_boxes if box.conf.item() >= confidence]
                
                # Plot semua hasil, termasuk yang di bawah confidence, untuk visualisasi awal
                res_plotted = results[0].plot()[:, :, ::-1]
                detected_image = Image.fromarray(res_plotted)
                st.image(res_plotted, caption='Detected Image', use_container_width=True)
                
                # Simpan hasil deteksi ke database
                save_detection(conn, detected_image) 
                
                st.session_state.detection_boxes = detected_boxes_above_confidence
                st.session_state.detection_model = model
                st.session_state.detection_confidence = confidence

    if source_img is not None and detect_button and 'detection_boxes' in st.session_state:
        st.markdown("---")
        st.header("Hasil Deteksi")
        
        boxes = st.session_state.detection_boxes
        model = st.session_state.detection_model
        
        # Periksa apakah ada kotak deteksi yang memenuhi ambang kepercayaan
        if len(boxes) == 0:
            st.markdown("Tidak ada penyakit yang terdeteksi/bukan merupakan daun singkong.")
        else:
            for box in boxes:
                label_index = int(box.cls)
                label = model.names[label_index]
                conf = box.conf.item()
                
                with st.container():
                    st.subheader(f"Deteksi: {label} (Tingkat Kepercayaan: {conf:.2f})")
                    
                    if GEMINI_CONFIGURED:
                        with st.spinner(f"Mendapatkan penjelasan hasil deteksi..."):
                            explanation = get_disease_explanation(label)
                            st.markdown(explanation)
                            
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
                    
                    st.markdown("---")
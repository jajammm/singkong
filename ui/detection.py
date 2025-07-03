import streamlit as st
from function.database import save_detection, load_detection_history
from config import get_disease_explanation
from function.pdf_utils import create_detection_pdf
from PIL import Image
from datetime import datetime
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import settings
from streamlit_webrtc import WebRtcMode, RTCConfiguration

def show_detection_page(conn, model, GEMINI_CONFIGURED):
    """Halaman Deteksi dan Menampilkan Histori Deteksi"""
    st.title("Deteksi Penyakit Daun Singkong")

    # Pilihan sumber input: gambar atau webcam
    source_radio = st.sidebar.radio("Pilih Sumber", ["Unggah Gambar", "Webcam"])
    
    class VideoTransformer(VideoProcessorBase):
        def __init__(self):
            self.model = YOLO(settings.DETECTION_MODEL)
            self.confidence = 0.3
            self.detected_labels = [] # Tambahkan atribut untuk menyimpan label yang terdeteksi

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            
            # Perform object detection
            results = self.model(img, stream=True)
            
            # Clear previously detected labels
            self.detected_labels = []

            # Draw bounding boxes on the frame
            found_detection = False # Flag untuk menandai apakah ada deteksi yang memenuhi confidence

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = box.conf.item()
                    if conf >= self.confidence: # Hanya pertimbangkan deteksi di atas confidence threshold
                        b = box.xyxy[0].cpu().numpy()  # get box coordinates in (top, left, bottom, right) format
                        c = box.cls
                        label = f"{self.model.names[int(c)]} {conf:.2f}"
                        self.detected_labels.append(self.model.names[int(c)]) # Simpan label yang terdeteksi
                        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                        cv2.putText(img, label, (int(b[0]), int(b[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        found_detection = True # Set flag jika ada deteksi
            
            # Jika tidak ada deteksi yang ditemukan di frame ini, tambahkan label 'Tidak Terdeteksi'
            if not found_detection:
                self.detected_labels.append("Tidak Terdeteksi")

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    source_img = None
    confidence = st.sidebar.slider("Pilih Tingkat Kepercayaan Model", 0.0, 1.0, 0.3)

    if source_radio == "Unggah Gambar":
        source_img = st.sidebar.file_uploader("Piih Gambar...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
        
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
                    
                    # Simpan hasil deteksi ke database (bisa disesuaikan apakah hanya yang terdeteksi signifikan yang disimpan)
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
                        
    elif source_radio == settings.WEBCAM:
        st.header("Webcam Deteksi Daun Singkong")

        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_processor_factory=VideoTransformer,
            async_processing=True,
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.confidence = confidence

            # Periksa apakah 'Tidak Terdeteksi' adalah satu-satunya atau ada label lain
            if "Tidak Terdeteksi" in webrtc_ctx.video_processor.detected_labels and len(set(webrtc_ctx.video_processor.detected_labels)) == 1:
                st.markdown("Tidak ada penyakit yang terdeteksi/bukan merupakan daun singkong.")
            elif webrtc_ctx.video_processor.detected_labels: # Jika ada label selain "Tidak Terdeteksi"
                st.markdown("---")
                st.header("Hasil Deteksi dari Webcam")

                # Hapus duplikat dan "Tidak Terdeteksi" jika ada label lain
                unique_detected_labels = list(set(webrtc_ctx.video_processor.detected_labels))
                if "Tidak Terdeteksi" in unique_detected_labels:
                    unique_detected_labels.remove("Tidak Terdeteksi")
                
                if not unique_detected_labels: # Jika setelah filter ternyata kosong (hanya "Tidak Terdeteksi")
                    st.markdown("Tidak ada penyakit yang terdeteksi/bukan merupakan daun singkong dari webcam.")
                else:
                    for label in unique_detected_labels:
                        with st.container():
                            st.subheader(f"Deteksi: {label}")
                            
                            if GEMINI_CONFIGURED:
                                with st.spinner(f"Mendapatkan penjelasan untuk {label}..."):
                                    explanation = get_disease_explanation(label)
                                    st.markdown(explanation)

                                    if webrtc_ctx.video_frame_buffer and len(webrtc_ctx.video_frame_buffer) > 0:
                                        img_array = webrtc_ctx.video_frame_buffer[-1].to_ndarray(format="bgr24")
                                        pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

                                        col1, col2 = st.columns([1, 2])
                                        with col1:
                                            if st.button(f"Simpan Gambar", key=f"save_{label}"):
                                                save_detection(pil_img)
                                                st.success("Gambar berhasil disimpan.")

                                        with col2:
                                            pdf_data = create_detection_pdf(pil_img, label, 0.0, explanation)
                                            if pdf_data:
                                                filename = f"deteksi_webcam_{label.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                                st.download_button(
                                                    label="📥 Download PDF",
                                                    data=pdf_data,
                                                    file_name=filename,
                                                    mime="application/pdf",
                                                    key=f"download_webcam_{label}"
                                                )
                            else:
                                st.warning("Gemini API tidak terkonfigurasi dengan benar.")
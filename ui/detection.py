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

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            
            # Perform object detection
            results = self.model(img, stream=True)
            
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
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    source_img = None
    confidence = st.sidebar.slider("Pilih Tingkat Kepercayaan Model", 0.0, 1.0, 0.3)  # Definisikan confidence di sini

    if source_radio == "Unggah Gambar":
        source_img = st.sidebar.file_uploader("Piih Gambar...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
        
        detect_button = st.sidebar.button('Deteksi Objek')

        col1, col2 = st.columns(2)
        with col1:
            try:
                if source_img is None:
                    st.markdown('<h3>Gambar Asli</h3>', unsafe_allow_html=True)
                    default_image_path = "images/original.png"  # Ganti path dengan gambar default
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
                default_detected_image_path = "images/hasil.png"  # Ganti path dengan gambar default hasil
                default_detected_image = Image.open(default_detected_image_path)
                st.image(default_detected_image, use_container_width=True)
            else:
                if detect_button:
                    # Proses deteksi gambar
                    st.markdown('<h3>Gambar Hasil Deteksi</h3>', unsafe_allow_html=True)
                    results = model.predict(uploaded_image)  # Prediksi tanpa 'conf' sebagai keyword
                    boxes = results[0].boxes
                    res_plotted = results[0].plot()[:, :, ::-1]
                    detected_image = Image.fromarray(res_plotted)
                    st.image(res_plotted, caption='Detected Image', use_container_width=True)
                    save_detection(conn, detected_image)
                    
                    # Filter berdasarkan confidence
                    boxes = [box for box in boxes if box.conf.item() >= confidence]  # Hanya yang lebih besar dari threshold
                    st.session_state.detection_boxes = boxes
                    st.session_state.detection_model = model
                    st.session_state.detection_confidence = confidence  # Menyimpan nilai confidence

    if source_img is not None and detect_button and 'detection_boxes' in st.session_state:
        st.markdown("---")
        st.header("Hasil Deteksi")
        
        boxes = st.session_state.detection_boxes
        model = st.session_state.detection_model
        confidence = st.session_state.detection_confidence
        
        for box in boxes:
            label_index = int(box.cls)
            label = model.names[label_index]  # Mengakses nama kelas dari model
            conf = box.conf.item()
            if conf >= confidence:
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

            # Jika deteksi berhasil, tampilkan hasilnya
            if hasattr(webrtc_ctx.video_processor, 'detected_labels') and webrtc_ctx.video_processor.detected_labels:
                st.markdown("---")
                st.header("Hasil Deteksi dari Webcam")

                for label in webrtc_ctx.video_processor.detected_labels:
                    with st.container():
                        st.subheader(f"Deteksi: {label}")
                        
                        # Penjelasan dengan Gemini
                        if GEMINI_CONFIGURED:
                            with st.spinner(f"Mendapatkan penjelasan untuk {label}..."):
                                explanation = get_disease_explanation(label)
                                st.markdown(explanation)

                                # Simpan gambar terakhir dari webcam
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

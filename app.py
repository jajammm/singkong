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
conn = sqlite3.connect('detection_tea.db', check_same_thread=False)
conn.execute('''CREATE TABLE IF NOT EXISTS detections
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp TEXT,
              image BLOB)''')
conn.commit()

# Model class for object detection
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
    # Setting page layout
    st.set_page_config(
        page_title="Object Detection using YOLOv8",
        page_icon="🍃",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Main page heading
    st.title("Deteksi Penyakit Pada Tanaman Daun Teh")

    # Placeholder for detection history
    history_placeholder = st.empty()

    # Sidebar
    st.sidebar.header("Detection")

    # Model Options
    confidence = float(st.sidebar.slider(
        "Select Model Confidence (%)", 25, 100, 30)) / 100

    # Model path for Detection
    model_path = settings.DETECTION_MODEL

    # Load Pre-trained ML Model
    model = YOLO(model_path)

    st.sidebar.header("Image/Video Config")
    source_radio = st.sidebar.radio(
        "Select Source", [settings.IMAGE, settings.WEBCAM])

    source_img = None
    # If image is selected
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image",
                             use_column_width=True)
                else:
                    uploaded_image = Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image",
                             use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = Image.open(
                    default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image',
                         use_column_width=True)
            else:
                if st.sidebar.button('Detect Objects'):
                    res = model.predict(uploaded_image,
                                        conf=confidence
                                        )
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    detected_image = Image.fromarray(res_plotted)
                    st.image(res_plotted, caption='Detected Image',
                             use_column_width=True, width=400)  # Set width to 400 pixels
                    save_detection(detected_image)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")

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

    else:
        st.error("Please select a valid source type!")

    # Detection history and delete button
    if st.sidebar.button('Detection History'):
        with history_placeholder.container():
            st.header("Detection History")
            if st.button("Close History"):
                history_placeholder.empty()
            history = load_detection_history()
            for id, timestamp, image in history:
                image = Image.open(io.BytesIO(image))
                with st.expander(f"ID: {id}, Time: {timestamp}"):
                    st.image(image, caption="Detected Image", use_column_width=False, width=500)

    # Delete all history
    if st.sidebar.button('Delete All History'):
        delete_all_detections()
        st.sidebar.success("All detection history has been deleted.")
        # Update the history display
        history_placeholder.empty()
        with history_placeholder.container():
            st.header("Detection History")
            st.info("No detection history available.")
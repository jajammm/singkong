import av
import cv2
from ultralytics import YOLO
import settings

class VideoTransformer:
    def __init__(self, confidence: float = 0.3):
        self.model = YOLO(settings.DETECTION_MODEL)
        self.confidence = confidence

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        results = self.model(img, stream=True)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].cpu().numpy()  # Get box coordinates in (top, left, bottom, right)
                conf = box.conf.item()
                if conf >= self.confidence:
                    x1, y1, x2, y2 = map(int, b)
                    label = f"{self.model.names[int(box.cls)]} {conf:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

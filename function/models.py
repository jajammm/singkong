from ultralytics import YOLO

class ObjectDetectionModel:
    def __init__(self, model_path: str, confidence: float = 0.3):
        # Memuat model YOLO
        self.model = YOLO(model_path)  
        self.confidence = confidence
        # Menyimpan 'names' setelah model dimuat
        self.names = self.model.names  # Ini harus mengakses names dari model YOLO

    def predict(self, image):
        # Melakukan prediksi dengan model
        results = self.model(image, conf=self.confidence)
        return results

import av
import cv2
from ultralytics import YOLO
import settings
import time # Import module time untuk logging performa (opsional)

class VideoTransformer:
    def __init__(self, confidence: float = 0.3):
        self.model = YOLO(settings.DETECTION_MODEL)
        self.confidence = confidence

        # --- Pengaturan Optimasi ---
        # 1. Resolusi Inferensi: Ukuran frame yang akan diproses oleh YOLO.
        #    Lebih kecil = lebih cepat, tapi akurasi mungkin sedikit menurun.
        #    Coba: (640, 480), (480, 360), (320, 240)
        self.infer_width, self.infer_height = 480, 360 

        # 2. Proses Setiap N Frame: Hanya lakukan deteksi YOLO setiap N frame.
        #    Nilai 1 = setiap frame (tidak ada skip).
        #    Nilai 2 = setiap frame kedua. Lebih tinggi = lebih cepat, tapi deteksi kurang real-time.
        self.process_every_n_frames = 2 
        
        self.frame_count = 0
        self.last_results = None # Untuk menyimpan hasil deteksi dari frame terakhir yang diproses

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        original_height, original_width, _ = img.shape # Dapatkan dimensi asli

        self.frame_count += 1
        
        # --- Lakukan Inferensi YOLO Hanya Pada Setiap N Frame ---
        if self.frame_count % self.process_every_n_frames == 0:
            start_time_inference = time.time() # Untuk logging performa
            
            # Ubah ukuran frame untuk inferensi (resolusi lebih rendah)
            img_for_inference = cv2.resize(img, (self.infer_width, self.infer_height))
            
            # Lakukan inferensi pada gambar beresolusi rendah
            # Tambahkan verbose=False untuk mengurangi log konsol dari YOLO
            self.last_results = self.model(img_for_inference, stream=True, verbose=False)
            
            end_time_inference = time.time() # Untuk logging performa
            # print(f"YOLO Inference Time: {end_time_inference - start_time_inference:.4f} seconds")

            # Reset frame_count jika sudah terlalu besar untuk menghindari overflow
            if self.frame_count >= 100000: 
                self.frame_count = 0
        
        # --- Gambar Bounding Box Menggunakan Hasil Deteksi Terakhir ---
        # Gunakan self.last_results agar bounding box tetap muncul
        # bahkan pada frame yang tidak melalui proses inferensi
        if self.last_results:
            for r in self.last_results:
                boxes = r.boxes
                for box in boxes:
                    conf = box.conf.item()
                    if conf >= self.confidence:
                        # Koordinat bounding box dari hasil inferensi (resolusi rendah)
                        x1_infer, y1_infer, x2_infer, y2_infer = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Skala kembali koordinat ke dimensi frame asli
                        x1 = int(x1_infer * (original_width / self.infer_width))
                        y1 = int(y1_infer * (original_height / self.infer_height))
                        x2 = int(x2_infer * (original_width / self.infer_width))
                        y2 = int(y2_infer * (original_height / self.infer_height))

                        label = f"{self.model.names[int(box.cls)]} {conf:.2f}"
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
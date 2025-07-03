import sqlite3
from datetime import datetime
import io

def init_db():
    conn = sqlite3.connect('cassava_leaves.db', check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  image BLOB)''')
    conn.commit()
    return conn

def save_detection(conn, image):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    conn.execute("INSERT INTO detections (timestamp, image) VALUES (?, ?)", (timestamp, img_byte_arr))
    conn.commit()

def load_detection_history(conn):
    c = conn.cursor()
    c.execute("SELECT id, timestamp, image FROM detections ORDER BY timestamp DESC")
    return c.fetchall()

def delete_all_detections(conn):
    c = conn.cursor()
    c.execute("DELETE FROM detections")
    conn.commit()

import streamlit as st
from function.database import load_detection_history, delete_all_detections
from PIL import Image
import io

def show_history(conn):
    """Menampilkan histori deteksi dan tombol hapus semua histori"""
    st.title("Histori Deteksi")

    # Ambil histori deteksi dari database
    history = load_detection_history(conn)

    if not history or len(history) == 0:
        st.info("Belum ada histori deteksi.")
    else:
        for id, timestamp, image in history:
            try:
                img = Image.open(io.BytesIO(image))
                with st.expander(f"ID: {id}, Time: {timestamp}"):
                    st.image(img, caption=f"Deteksi pada {timestamp}", use_container_width=True)
            except Exception as e:
                st.error(f"Error menampilkan gambar ID: {id}: {str(e)}")

    # Tombol untuk menghapus semua histori deteksi
    if st.button("Hapus Semua Histori Deteksi"):
        delete_all_detections(conn)
        st.success("Semua histori deteksi telah dihapus.")

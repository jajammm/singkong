import streamlit as st
from function.database import load_detection_history, delete_all_detections
from PIL import Image
import io
import base64

def show_history(conn):
    st.title("Histori Deteksi")

    # --- CSS Internal Responsif ---
    st.markdown("""
    <style>
    /* Gaya untuk kontainer gambar - berlaku untuk semua ukuran layar */
    .detection-image-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 100%;
        text-align: center;
    }

    /* Gaya untuk gambar default (layar desktop besar) */
    .detection-image {
        max-width: 50%; 
        height: 50vh;
        object-fit: contain;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* Gaya untuk teks caption */
    .detection-caption {
        font-size: 0.9em;
        color: grey;
        margin-top: 10px;
    }

    /* --- Media Queries untuk Responsif --- */

    /* Untuk tablet dan layar yang lebih kecil (maksimal 768px lebar) */
    @media (max-width: 768px) {
        .detection-image {
            max-width: 70%; /* Lebih lebar di tablet, misalnya 70% */
            height: 70vh;  /* Tinggi lebih proporsional untuk tablet */
        }
    }

    /* Untuk mobile dan layar yang sangat kecil (maksimal 480px lebar) */
    @media (max-width: 480px) {
        .detection-image {
            max-width: 40%; /* Hampir memenuhi layar di mobile, misalnya 90% */
            height: 40vh;  /* Tinggi yang lebih moderat untuk mobile */
        }
    }
    </style>
    """, unsafe_allow_html=True)
    # --- Akhir CSS Internal Responsif ---

    history = load_detection_history(conn)

    if not history or len(history) == 0:
        st.info("Belum ada histori deteksi.")
    else:
        for id, timestamp, image_data in history:
            try:
                img = Image.open(io.BytesIO(image_data))

                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                with st.expander(f"ID: {id}, Waktu: {timestamp}"):
                    st.markdown(f"""
                    <div class="detection-image-container">
                        <img src="data:image/png;base64,{img_str}" class="detection-image" alt="Deteksi">
                        <p class="detection-caption">Deteksi pada {timestamp}</p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error menampilkan gambar ID: {id}: {str(e)}")

    if st.button("Hapus Semua Histori Deteksi"):
        delete_all_detections(conn)
        st.success("Semua histori deteksi telah dihapus.")
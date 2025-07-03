import streamlit as st
from PIL import Image
import io
import os

def show_guide():
    """Halaman Panduan untuk penggunaan aplikasi, dengan langkah-langkah terstruktur, penjelasan, dan gambar."""

    # --- CSS untuk Halaman Panduan ---
    # PENTING: Untuk konsistensi seluruh aplikasi, sangat disarankan untuk MEMINDAHKAN SEMUA kode <style> ini
    # ke file assets/style.css Anda.
    st.markdown(
        """
        <style>
        /* Gaya judul halaman panduan */
        .guide-title {
            font-size: 3.5em;
            color: #3C6A3C; 
            text-align: center;
            margin-bottom: 0.5em;
        }

        /* Gaya untuk header setiap langkah */
        .guide-step-header {
            font-size: 1.5em;
            color: #3C6A3C; /* Hijau gelap */
            margin-top: 1.5em; /* Memberikan jarak atas untuk memisahkan langkah */
            margin-bottom: 0.8em;
            padding-bottom: 5px;
        }

        .guide-step-text {
            font-size: 1em;
            line-height: 1.6;
            color: #333333;
            margin-bottom: 1em;
            text-align: left;
        }

        /* --- Gaya Gambar (max-height dan width 100%) --- */
        /* Menargetkan elemen <img> di dalam komponen st.image */
        [data-testid="stImage"] img {
            max-height: 20vh; /* Ketinggian maksimum yang diinginkan */
            width: 100%; /* Memastikan gambar mengisi lebar kolom */
            object-fit: contain; /* Sangat penting: mencegah distorsi. Gambar akan diskala ke bawah agar pas tanpa terpotong. */
            display: block; /* Memastikan properti width/height bekerja dengan benar */
            margin-left: auto; /* Memusatkan gambar jika tidak mengisi lebar penuh karena max-height */
            margin-right: auto;
        }
        /* Opsional: Tambahkan sedikit margin/padding di sekitar komponen gambar itu sendiri */
        [data-testid="stImage"] {
            margin-top: 1em;
            margin-bottom: 1em;
        }

        /* --- Penyesuaian Responsif --- */
        @media (max-width: 768px) {
            .guide-title { font-size: 2.5em; }
            .guide-step-header { font-size: 1.3em; margin-top: 1em;}
            .guide-step-text { font-size: 0.95em; }
            [data-testid="stImage"] img { max-height: 100vh; } /* Sesuaikan tinggi maksimum untuk tablet */
        }
        @media (max-width: 480px) {
            .guide-title { font-size: 2em; }
            .guide-step-header { font-size: 1.1em; margin-top: 0.8em;}
            .guide-step-text { font-size: 0.9em; }
            [data-testid="stImage"] img { max-height: 100vh; } /* Sesuaikan tinggi maksimum untuk ponsel */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h2 class="guide-title">📖 Panduan Penggunaan Aplikasi 📖</h2>', unsafe_allow_html=True)

    # Definisi langkah-langkah panduan beserta teks penjelasan dan path gambar
    guide_steps = [
        {
            "title": "Membuka Halaman Deteksi",
            "text": "Langkah pertama adalah mengakses halaman deteksi pada aplikasi. Ini dilakukan dengan mengklik tombol yang sesuai untuk menuju halaman deteksi dari halaman utama aplikasi. Tombol terlihat pada menu samping kiri atau sidebar",
            "images": ["assets/guide/step_1.png"]
        },
        {
            "title": "Memilih Metode Deteksi",
            "text": "Anda memiliki dua pilihan untuk mengunggah gambar: melalui 'Upload Gambar' dari perangkat Anda atau menggunakan 'Webcam' secara langsung. Pilih metode yang paling sesuai dengan kebutuhan Anda.",
            "images": ["assets/guide/step_2.png"]
        },
        {
            "title": "Memilih Tingkat Kepercayaan Model (Opsional)",
            "text": "Anda dapat menyesuaikan 'Tingkat Kepercayaan Model' menggunakan slider. Ini menentukan seberapa yakin model harus dalam mendeteksi objek sebelum menampilkannya. Nilai yang lebih tinggi berarti deteksi yang lebih ketat.",
            "images": ["assets/guide/step_3.png"]
        },
        {
            "title": "Mengunggah Gambar (Jika Menggunakan Metode Upload Gambar)",
            "text": "Jika Anda memilih 'Upload Gambar', klik tombol 'Browse files' dan pilih gambar daun singkong dari galeri perangkat Anda. Pastikan gambar jelas dan fokus pada daun.",
            "images": ["assets/guide/step_4.png"]
        },
        {
            "title": "Melakukan Deteksi",
            "text": "Setelah gambar diunggah atau webcam diaktifkan, klik tombol 'Deteksi Objek' untuk memproses gambar. Model YOLO akan menganalisis daun untuk mengidentifikasi potensi penyakit.",
            "images": ["assets/guide/step_5.png"]
        },
        {
            "title": "Melihat Hasil Deteksi",
            "text": "Aplikasi akan menampilkan gambar daun dengan kotak pembatas di sekitar area yang terdeteksi penyakit. Anda juga akan melihat label penyakit dan tingkat kepercayaan model untuk setiap deteksi.",
            "images": ["assets/guide/step_6.png"]
        },
        {
            "title": "Melihat Detail Penyakit",
            "text": "Untuk setiap penyakit yang terdeteksi, aplikasi akan memberikan penjelasan singkat tentang gejala dan karakteristiknya, membantu Anda memahami kondisi daun singkong lebih lanjut.",
            "images": ["assets/guide/step_7.png"]
        },
        {
            "title": "Mengelola Riwayat Deteksi",
            "text": "Semua deteksi yang Anda lakukan akan disimpan di 'Histori Deteksi'. Anda dapat mengunjungi halaman ini kapan saja untuk meninjau deteksi sebelumnya atau menghapus seluruh riwayat.",
            "images": ["assets/guide/step_8.png"]
        }
    ]

    for i, step in enumerate(guide_steps):
        # TIDAK menggunakan st.container() di sini, hanya menampilkan elemen secara langsung
        st.markdown(f'<h3 class="guide-step-header">{i+1}. {step["title"]}</h3>', unsafe_allow_html=True)
        st.markdown(f'<p class="guide-step-text">{step["text"]}</p>', unsafe_allow_html=True)

        if step["images"]:
            num_cols = min(len(step["images"]), 3)
            cols = st.columns(num_cols)

            for idx, img_path in enumerate(step["images"]):
                try:
                    img = Image.open(img_path)
                    with cols[idx]:
                        # use_container_width=True akan mengatur width: 100% dari kolom
                        st.image(img, caption=f"Langkah {i+1}.{idx+1}", use_container_width=True)
                except FileNotFoundError:
                    with cols[idx]:
                        st.warning(f"Gambar tidak ditemukan: {img_path}. Pastikan file ada di folder 'assets/guide'.")
                except Exception as e:
                    with cols[idx]:
                        st.error(f"Error memuat gambar {img_path}: {e}")
        st.markdown("<hr>", unsafe_allow_html=True) # Tambahkan pemisah setelah setiap langkah
import streamlit as st
import os
from PIL import Image

def show_main_page():
    """Enhanced Main Page with application background, disease explanations, and responsiveness,
    all with a consistent styling."""
    
    st.markdown(
        """
        <style>
        .main-title {
            text-align: center;
        }
        
        .disease-explanation-title {
            font-size: 2em; /* Ukuran judul utama */
            color: #3C6A3C;
            text-align: center;
            margin-bottom: 1em;
        }

        .disease-step-header { /* Menggunakan nama yang mirip dengan guide-step-header untuk konsistensi */
            font-size: 1.5em;
            color: #3C6A3C;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
            padding-bottom: 5px;
        }

        .disease-explanation-text { /* Menggunakan nama yang mirip dengan guide-step-text */
            font-size: 1em;
            line-height: 1.6;
            color: #333333;
            margin-bottom: 1em;
            text-align: left;
        }

        /* --- Gaya Gambar (max-height dan width 100%) --- */
        /* Menargetkan elemen <img> di dalam komponen st.image */
        [data-testid="stImage"] img {
            max-height: 50vh; /* Atur tinggi maksimum gambar di sini, misalnya 200px */
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
            .disease-explanation-title { font-size: 1.8em; }
            .disease-step-header { font-size: 1.3em; margin-top: 1em;}
            .disease-explanation-text { font-size: 0.95em; }
            [data-testid="stImage"] img { max-height: 150px; } /* Sesuaikan tinggi maksimum untuk tablet */
        }
        @media (max-width: 480px) {
            .disease-explanation-title { font-size: 1.5em; }
            .disease-step-header { font-size: 1.1em; margin-top: 0.8em;}
            .disease-explanation-text { font-size: 0.9em; }
            [data-testid="stImage"] img { max-height: 100vh; } /* Sesuaikan tinggi maksimum untuk ponsel */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h2 class="main-title">Selamat Datang di Aplikasi Deteksi Penyakit Daun Singkong</h2>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <p class="app-description">
        Aplikasi ini dibuat dengan menggunkan YOLOv11
        untuk membantu dalam mengidentifikasi berbagai penyakit pada daun singkong
        dengan cepat dan akurat.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h3>Latar Belakang Pembuatan Aplikasi:</h3>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="background-text">
        Singkong merupakan komoditas pertanian penting yang menjadi sumber pangan dan pendapatan bagi jutaan orang, terutama di daerah tropis.
        Namun, produktivitas singkong seringkali terancam oleh berbagai penyakit daun yang dapat menyebabkan kerugian hasil panen yang signifikan.
        Aplikasi ini dikembangkan dengan tujuan untuk membantu ahli pangan dan petani singkong dalam melakukan
        deteksi dini penyakit daun. Dengan identifikasi masalah yang cepat, diharapkan langkah-langkah penanganan yang tepat
        dapat segera diambil, sehingga menjaga kesehatan tanaman, meningkatkan kualitas dan kuantitas panen, serta mendukung ketahanan pangan.
        </p>
        """,
        unsafe_allow_html=True
    )

    # --- New Section for Disease Explanations ---
    st.markdown('<h3 class="section-header">Mengenal Penyakit Umum Daun Singkong:</h3>', unsafe_allow_html=True)
    disease_info = [
        {
            "title": "Penyakit Garis Coklat Singkong (Cassava Brown Streak Disease - CBSD)",
            "text": """
            <b>CBSD</b> adalah penyakit virus serius yang terutama menyerang umbi singkong,
            menyebabkan pembusukan coklat pada jaringan akar yang tidak dapat dimakan.
            Gejala pada daun meliputi urat daun yang menguning (klorosis) atau nekrosis
            bergaris, seringkali pada daun tua. Penyakit ini dapat menyebabkan kerugian panen yang sangat besar.
            """,
            "images": [
                "assets/diseases/cbsd.png"
            ]
        },
        {
            "title": "Hawar Bakteri Singkong (Cassava Green Mite - CGM)",
            "text": """
            Meskipun disebut "green mite," <b>CGM</b> sebenarnya adalah tungau hijau
            (<i>Mononychellus tanajoa</i>) yang menyerang daun singkong.
            Gejala yang terlihat adalah daun menguning, pertumbuhan kerdil, dan daun yang
            terdistorsi atau mengeriting, terutama pada pucuk tanaman. Serangan parah dapat
            mengurangi area fotosintetik dan hasil panen secara signifikan.
            """,
            "images": [
                "assets/diseases/cgm.png"
            ]
        },
        {
            "title": "Bercak Daun Bakteri Singkong (Cassava Bacterial Blight - CBB)",
            "text": """
            <b>CBB</b> disebabkan oleh bakteri <i>Xanthomonas axonopodis pv. manihotis</i>.
            Gejala awal sering muncul sebagai bercak kecil berair yang berkembang menjadi
            lesi angular berwarna coklat atau kehitaman pada daun. Pada kondisi lembap,
            area yang terinfeksi dapat meluas dan menyebabkan daun layu serta rontok.
            """,
            "images": [
                "assets/diseases/cbb.png"
            ]
        },
        {
            "title": "Penyakit Mosaik Singkong (Cassava Mosaic Disease - CMD)",
            "text": """
            <b>CMD</b> adalah penyakit virus yang paling luas dan merusak pada singkong di Afrika.
            Gejala khasnya adalah pola mosaik yang jelas pada daun, di mana area hijau normal
            diselingi dengan area kekuningan atau pucat. Daun juga bisa mengeriting, terdistorsi,
            dan berukuran kecil, menyebabkan tanaman kerdil dan hasil panen sangat rendah.
            """,
            "images": [
                "assets/diseases/cmd.png"
            ]
        }
    ]

    for i, disease in enumerate(disease_info):
        st.markdown(f'<h5 class="disease-step-header">{i+1}. {disease["title"]}</h5>', unsafe_allow_html=True)
        st.markdown(f'<p class="disease-explanation-text">{disease["text"]}</p>', unsafe_allow_html=True)

        if disease["images"]:
            # Menentukan jumlah kolom berdasarkan jumlah gambar, maksimal 3 kolom
            num_cols = min(len(disease["images"]), 3)
            cols = st.columns(num_cols)

            for idx, img_path in enumerate(disease["images"]):
                try:
                    # Memeriksa apakah file gambar ada sebelum mencoba membukanya
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        with cols[idx]:
                            st.image(img, caption=f"Gambar {disease['title']}", use_container_width=True)
                    else:
                        with cols[idx]:
                            st.warning(f"Gambar tidak ditemukan: {img_path}. Pastikan file ada di folder yang benar.")
                except Exception as e:
                    with cols[idx]:
                        st.error(f"Error memuat gambar {img_path}: {e}")

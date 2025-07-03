import streamlit as st

def show_main_page():
    """Enhanced Main Page with application background, disease explanations, and responsiveness,
    all with a consistent styling."""
    
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 3.5em; /* Larger on big screens */
            color: #2e8b57; /* SeaGreen */
            text-align: center;
            margin-bottom: 0.5em;
        }
        .section-header {
            font-size: 1.8em;
            color: #3cb371; /* MediumSeaGreen */
            margin-top: 1.5em;
            margin-bottom: 0.8em;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 5px;
        }
        .app-description, .background-text{
            font-size: 1em;
            line-height: 1.6;
            color: #333333;
            margin-bottom: 2em;
        }
        .disease-explanation-text {
            font-size: 1em;
            color: #333333;
        }
        .call-to-action {
            font-size: 1.2em;
            color: #1e8449; /* Darker Green */
            text-align: center;
            margin-top: 2.5em;
            padding: 10px;
            border: 1px solid #1e8449;
            border-radius: 8px;
            background-color: #e6ffe6; /* Lightest Green */
        }
        
        /* Specific styling for disease titles within the explanation */
        .disease-title-bold {
            font-size: 1em; /* Slightly larger than surrounding text */
            font-weight: bold;
            margin-top: 1em; /* Space before each new disease explanation */
            display: block; /* Ensures it acts like a block element for margin */
        }

        /* --- Responsive Adjustments --- */
        @media (max-width: 768px) { /* For tablets and smaller */
            .main-title {
                font-size: 2.5em; /* Smaller font for smaller screens */
            }
            .section-header {
                font-size: 1.5em;
            }
            .app-description, .background-text, .disease-explanation-text, .call-to-action {
                font-size: 1em;
            }
            .disease-title-bold {
                font-size: 1.1em;
            }
        }

        @media (max-width: 480px) { /* For mobile phones */
            .main-title {
                font-size: 1.8em;
            }
            .section-header {
                font-size: 1.3em;
            }
            .app-description, .background-text, .disease-explanation-text, .call-to-action {
                font-size: 0.9em;
            }
            .disease-title-bold {
                font-size: 1em;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h2 class="main-title">🌱 Selamat Datang di Aplikasi Deteksi Penyakit Daun Singkong 🌱</h2>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <p class="app-description">
        Aplikasi inovatif ini memanfaatkan kekuatan teknologi <b>deteksi objek YOLO</b>
        untuk membantu dalam mengidentifikasi berbagai penyakit pada daun singkong
        dengan cepat dan akurat.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h3 class="section-header">🌍 Latar Belakang Pembuatan Aplikasi:</h3>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="background-text">
        Singkong merupakan komoditas pertanian penting yang menjadi sumber pangan dan pendapatan bagi jutaan orang, terutama di daerah tropis.
        Namun, produktivitas singkong seringkali terancam oleh berbagai penyakit daun yang dapat menyebabkan kerugian hasil panen yang signifikan.
        Aplikasi ini dikembangkan dengan tujuan untuk membantu ahli pangan dan petani singkong dalam melakukan
        deteksi dini penyakit daun. Dengan identifikasi masalah yang cepat, diharapkan langkah-langkah penanganan yang tepat
        dapat segera diambil, sehingga menjaga kesehatan tanaman, meningkatkan kualitas dan kuantitas panen, serta mendukung ketahanan pangan.
        Kami berkomitmen untuk menyediakan alat yang praktis dan efektif untuk kemajuan pertanian singkong.
        </p>
        """,
        unsafe_allow_html=True
    )

    # --- New Section for Disease Explanations ---
    st.markdown('<h3 class="section-header">🌿 Mengenal Penyakit Umum Daun Singkong:</h3>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="disease-explanation-text">
        Untuk memberikan pemahaman yang lebih baik tentang apa yang mungkin Anda deteksi, berikut adalah penjelasan singkat mengenai
        beberapa penyakit umum yang sering menyerang daun singkong:

        <span class="disease-title-bold">1. Penyakit Busuk Batang dan Akar Singkong (Cassava Brown Streak Disease - CBSD)</span>
        <b>CBSD</b> adalah penyakit virus serius yang terutama menyerang umbi singkong,
        menyebabkan pembusukan coklat pada jaringan akar yang tidak dapat dimakan.
        Gejala pada daun meliputi urat daun yang menguning (klorosis) atau nekrosis
        bergaris, seringkali pada daun tua. Penyakit ini dapat menyebabkan kerugian panen yang sangat besar.

        <span class="disease-title-bold">2. Hawar Bakteri Singkong (Cassava Green Mite - CGM)</span>
        Meskipun disebut "green mite," <b>CGM</b> sebenarnya adalah tungau hijau
        (<i>Mononychellus tanajoa</i>) yang menyerang daun singkong.
        Gejala yang terlihat adalah daun menguning, pertumbuhan kerdil, dan daun yang
        terdistorsi atau mengeriting, terutama pada pucuk tanaman. Serangan parah dapat
        mengurangi area fotosintetik dan hasil panen secara signifikan.

        <span class="disease-title-bold">3. Bercak Daun Bakteri Singkong (Cassava Bacterial Blight - CBB)</span>
        <b>CBB</b> disebabkan oleh bakteri <i>Xanthomonas axonopodis pv. manihotis</i>.
        Gejala awal sering muncul sebagai bercak kecil berair yang berkembang menjadi
        lesi angular berwarna coklat atau kehitaman pada daun. Pada kondisi lembap,
        area yang terinfeksi dapat meluas dan menyebabkan daun layu serta rontok.

        <span class="disease-title-bold">4. Penyakit Mosaik Singkong (Cassava Mosaic Disease - CMD)</span>
        <b>CMD</b> adalah penyakit virus yang paling luas dan merusak pada singkong di Afrika.
        Gejala khasnya adalah pola mosaik yang jelas pada daun, di mana area hijau normal
        diselingi dengan area kekuningan atau pucat. Daun juga bisa mengeriting, terdistorsi,
        dan berukuran kecil, menyebabkan tanaman kerdil dan hasil panen sangat rendah.
        </p>
        """, unsafe_allow_html=True
    )

import google.generativeai as genai
import streamlit as st

def configure_gemini_api():
    try:
        # Ambil API key dari file secrets.toml
        gemini_api_key = st.secrets["gemini"]["api_key"]
        
        # Konfigurasi API Gemini
        genai.configure(api_key=gemini_api_key)
        return True
    except Exception as e:
        # Jika terjadi kesalahan, tampilkan pesan error
        print(f"Error konfigurasi Gemini API: {str(e)}")
        return False

def get_disease_explanation(disease_label):
    """Mengambil penjelasan tentang penyakit daun singkong menggunakan API Gemini"""
    # Memeriksa apakah API key tersedia di st.secrets
    if "gemini" not in st.secrets or "api_key" not in st.secrets["gemini"]:
        return "API Gemini tidak terkonfigurasi dengan benar. Periksa file secrets.toml Anda."
    
    try:
        # Membuat prompt untuk API Gemini
        prompt = f"""
        Berikan penjelasan detail tentang penyakit daun singkong "{disease_label}" dengan format berikut (langsung jelaskan saja tanpa harus mengiyakan perintah saya):
        
        PENJELASAN:
        [Jelaskan gejala dan penyebab penyakit pada daun singkong tersebut secara detail.]
        
        DAMPAK:
        [Jelaskan dampak penyakit ini terhadap tanaman singkong]
        
        REKOMENDASI PENANGANAN:
        [Berikan 3-5 rekomendasi penanganan yang bisa dilakukan petani]
        """
        
        # Menggunakan model Gemini untuk menghasilkan konten
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        # Mengembalikan hasil penjelasan
        return response.text.strip()
    except Exception as e:
        return f"Error mendapatkan penjelasan: {str(e)}"

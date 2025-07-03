from fpdf import FPDF
from datetime import datetime
import os
import tempfile

def clean_markdown(text):
    text = text.replace('**', '').replace('*', '').replace('#', '').replace('`', '')
    return text

def create_detection_pdf(image, label, confidence, explanation):
    try:
        pdf = FPDF()
        pdf.add_page()

        # Menambahkan konten ke PDF (sama seperti sebelumnya)
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(190, 10, 'Hasil Deteksi Penyakit Daun Singkong', 0, 1, 'C')
        pdf.ln(10)
        
        # Menambahkan waktu deteksi dan penjelasan lainnya
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        pdf.set_font('Arial', '', 12)
        pdf.cell(190, 10, f'Waktu Deteksi: {current_time}', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, f'Penyakit Terdeteksi: {label}', 0, 1)
        pdf.cell(190, 10, f'Tingkat Kepercayaan: {confidence:.2f}', 0, 1)
        pdf.ln(5)

        # Simpan gambar sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_filename = temp_file.name
            image.save(temp_filename)

        pdf.cell(190, 10, 'Gambar Daun Singkong:', 0, 1)
        pdf.image(temp_filename, x=10, y=None, w=180)

        os.unlink(temp_filename)
        
        # Menambahkan penjelasan
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(190, 10, 'Analisis dan Rekomendasi:', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        explanation_lines = explanation.split('\n')

        current_mode = 'normal'
        for line in explanation_lines:
            clean_line = clean_markdown(line)
            
            if "Penjelasan:" in clean_line or "Dampak:" in clean_line or "Rekomendasi" in clean_line:
                pdf.ln(5)
                pdf.set_font('Arial', 'B', 12)
                current_mode = 'title'
            elif clean_line.strip() == "":
                pdf.ln(5)
                pdf.set_font('Arial', '', 11)
                current_mode = 'normal'
            else:
                if current_mode == 'title':
                    pdf.set_font('Arial', '', 11)
                    current_mode = 'normal'
            
            pdf.multi_cell(0, 6, clean_line)

        return pdf.output(dest='S').encode('latin1')
    except Exception as e:
        return None

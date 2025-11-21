# TomatoDoc AI — Deteksi Penyakit Daun Tomat

Aplikasi web untuk mendeteksi penyakit daun tomat menggunakan Kecerdasan Buatan. Sistem memadukan model CNN (TensorFlow/Keras) dan fallback Traditional ML (ekstraksi fitur OpenCV) dengan antarmuka web modern berbasis Flask + Bootstrap.

## Fitur Utama
- Deteksi otomatis 10 kelas penyakit daun tomat + kondisi sehat.
- Dua mode prediksi:
  - `CNN (Deep Learning)` jika model tersedia (`models/tomato_cnn_model.h5`).
  - `Traditional ML (Fallback)` jika CNN tidak tersedia (menggunakan fitur HSV, statistik warna, tekstur gradien).
- Antarmuka web modern:
  - Upload gambar dengan drag & drop dan pratinjau.
  - Indikator tingkat kepercayaan (progress bar + label low/medium/high).
  - Ringkasan diagnosis: deskripsi, gejala, penyebab, penanganan, pencegahan.
  - Rekomendasi tindakan, badge tingkat keparahan, dan informasi urgensi.
  - Grafik Top-5 probabilitas penyakit.
- Halaman Tentang berisi teknologi, jenis penyakit yang dideteksi, cara kerja, dan performa model.
- API JSON untuk integrasi aplikasi lain.

## Teknologi
- Backend: `Flask (Python)`
- Model: `TensorFlow/Keras` untuk CNN, `scikit-learn` untuk fallback
- CV & utilitas: `OpenCV`, `NumPy`, `Pillow`
- Frontend: `Bootstrap 5`, `HTML`, `CSS`, `JavaScript`

## Prasyarat
- Python 3.8–3.10 (disarankan Python 3.10 untuk kompatibilitas TensorFlow 2.13)
- Pip dan virtual environment (opsional namun disarankan)

## Instalasi
1) Clone atau unduh repo ini.
2) Buat environment dan instal dependensi:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3) Pastikan folder `models/` berisi file berikut (untuk mode CNN):
   - `models/tomato_cnn_model.h5`
   - `models/cnn_class_names.pkl`
   - `models/cnn_metadata.pkl`
   Jika file CNN tidak tersedia, aplikasi akan mencoba fallback Traditional ML. (Opsional: siapkan `models/tomato_disease_model.pkl` dan `models/class_names.pkl` jika Anda punya model tradisional.)

## Menjalankan Aplikasi
```bash
python app_cnn.py
```
- Server berjalan di `http://localhost:5000`.
- Upload tersimpan di `static/uploads` (maks. 16MB; ekstensi: png, jpg, jpeg, gif).

## Cara Pakai (UI)
- Buka `http://localhost:5000`.
- Tarik & lepas atau pilih gambar daun tomat.
- Klik tombol analisis untuk melihat hasil diagnosis, rekomendasi, dan grafik probabilitas.

## API
- Endpoint: `POST /upload`
- Content-Type: `multipart/form-data`
- Field: `file` (gambar)

Contoh request:
```bash
curl -F "file=@/path/to/daun-tomat.jpg" http://localhost:5000/upload
```

Contoh respons (ringkas):
```json
{
  "disease": "Hawar Daun Awal",
  "confidence": 0.82,
  "confidence_level": "High",
  "description": "...",
  "symptoms": ["..."],
  "causes": ["..."],
  "treatment": ["..."],
  "prevention": ["..."],
  "severity": "Sedang-Tinggi",
  "urgency": "Tangani dalam 1-2 hari",
  "all_probabilities": {"Hawar Daun Awal": 0.82, "Bercak Bakteri": 0.12, "...": 0.06},
  "top_3_diseases": [["Hawar Daun Awal", 0.82], ["Bercak Bakteri", 0.12], ["Jamur Daun", 0.06]],
  "recommendations": ["..."],
  "is_healthy": false,
  "model_type": "CNN (Deep Learning)",
  "filename": "20241121_120301_daun.jpg",
  "file_path": "static/uploads/20241121_120301_daun.jpg",
  "image_path": "uploads/20241121_120301_daun.jpg",
  "file_size": 123456,
  "timestamp": "20241121_120301"
}
```

Penjelasan field penting:
- `disease`: Nama penyakit (Indonesia) atau `Sehat`.
- `confidence` + `confidence_level`: Skor dan kategori kepercayaan.
- `all_probabilities`: Probabilitas per kelas (diterjemahkan ke bahasa Indonesia).
- `top_3_diseases`: Tiga kandidat teratas beserta probabilitasnya.
- `recommendations`: Saran tindakan berdasarkan hasil.
- `severity` dan `urgency`: Tingkat keparahan dan urgensi penanganan.
- `model_type`: Jenis model yang digunakan (`CNN` atau `Traditional ML`).
- `filename`/`file_path`/`image_path`/`file_size`/`timestamp`: Info file terunggah.

## Rute
- `/` — Halaman utama upload & analisis.
- `/about` — Informasi teknologi, jenis penyakit, cara kerja, performa.
- `/upload` — Endpoint API untuk analisis gambar (JSON).

## Struktur Proyek
```text
.
├── app_cnn.py                # Aplikasi Flask + logika prediksi
├── requirements.txt          # Dependensi Python
├── models/                   # Model & metadata
│   ├── tomato_cnn_model.h5
│   ├── cnn_class_names.pkl
│   └── cnn_metadata.pkl
├── templates/                # HTML (Jinja2)
│   ├── base.html
│   ├── index.html
│   └── about.html
├── static/                   # Asset web & uploads
│   ├── css/
│   ├── js/
│   └── uploads/             # Dibuat otomatis saat runtime
├── dataset/                  # Dataset (opsional, untuk referensi)
├── dataset_info.py           # Info dataset & kelas
└── download_dataset.py       # Script unduh dataset (opsional)
```

## Dataset & Kelas Penyakit
- Kelas yang dideteksi:
  - Bercak Bakteri, Hawar Daun Awal, Hawar Daun Lanjut, Jamur Daun,
    Bercak Daun Septoria, Tungau Laba-laba, Bercak Target,
    Virus Keriting Daun Kuning, Virus Mozaik Tomat, Sehat.
- Lihat `dataset_info.py` untuk ringkasan kategori dan jumlah gambar.

## Konfigurasi Penting
- Folder upload: `static/uploads` (dibuat otomatis).
- Maksimal ukuran file: `16MB`.
- Ekstensi diizinkan: `png`, `jpg`, `jpeg`, `gif`.

## Catatan & Batasan
- Angka akurasi yang ditampilkan (mis. 73%) adalah gambaran performa model saat pengujian awal dan dapat berubah tergantung kualitas foto.
- Sistem adalah alat bantu diagnosis, bukan pengganti konsultasi ahli agronomi.
- Untuk hasil terbaik, gunakan foto tajam, pencahayaan baik, dan fokus pada daun yang terindikasi.

## Pengembangan & Testing Cepat
- Jalankan server: `python app_cnn.py` lalu buka `http://localhost:5000`.
- Uji API cepat: `curl -F "file=@/path/to/image.jpg" http://localhost:5000/upload`.
- Lihat log detail di terminal untuk proses preprocessing dan prediksi.

## Lisensi
Belum ditentukan. Silakan tambahkan lisensi sesuai kebutuhan proyek Anda.
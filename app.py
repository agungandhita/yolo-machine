from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import numpy as np
import pickle
from PIL import Image
from werkzeug.utils import secure_filename
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import colorsys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# =====================================================================
# Mapping: Nama kelas output model CNN → kunci disease_database.json
# =====================================================================
CLASS_TO_DB = {
    "Tomato_Bacterial_spot":                       "Bercak Bakteri",
    "Tomato_Early_blight":                         "Hawar Daun Awal",
    "Tomato_Late_blight":                          "Hawar Daun Lanjut",
    "Tomato_Leaf_Mold":                            "Jamur Daun",
    "Tomato_Septoria_leaf_spot":                   "Bercak Daun Septoria",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tungau Laba-laba",
    "Tomato__Target_Spot":                         "Bercak Target",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":       "Virus Keriting Daun Kuning",
    "Tomato__Tomato_mosaic_virus":                 "Virus Mozaik Tomat",
    "Tomato_healthy":                              "Sehat",
}

# =====================================================================
# Threshold validasi gambar
#
# MIN_CONFIDENCE_THRESHOLD: ambang batas minimum confidence model.
#   Model dengan akurasi ~51% wajar memberikan confidence 35-55%,
#   terutama untuk gambar daun yang rusak/menguning.
#   Threshold diturunkan dari 0.55 → 0.30 agar tidak menolak
#   gambar daun tomat yang memang valid tapi kondisinya parah.
#
# MAX_ENTROPY_RATIO: tolak jika distribusi prediksi terlalu merata.
#   Entropy maksimum untuk 10 kelas = ln(10) ≈ 2.303
#   Naikkan dari 0.72 → 0.90 untuk lebih toleran.
#
# COLOR_LEAF_MIN_RATIO: minimal % pixel yang memiliki warna daun
#   (hijau, kuning, coklat). Jika < threshold ini, kemungkinan
#   bukan foto daun sama sekali.
# =====================================================================
MIN_CONFIDENCE_THRESHOLD = 0.30   # turun dari 0.55 → 0.30 (model ~51% akurasi)
MAX_ENTROPY_RATIO        = 0.90   # naik dari 0.72 → 0.90 (lebih toleran)
COLOR_LEAF_MIN_RATIO     = 0.10   # minimal 10% pixel berwarna daun
COLOR_LEAF_MAX_RATIO     = 0.98   # tolak jika hampir semua pixel putih/hitam solid

# Global state
MODEL       = None
CLASS_NAMES = None
DISEASE_DB  = None


def load_resources():
    global MODEL, CLASS_NAMES, DISEASE_DB
    print("🚀 Memulai TomatoDoc AI...")

    # ── Model ────────────────────────────────────────────────────────
    model_path = "models/mobilenetv2_tomato.h5"
    if os.path.exists(model_path):
        MODEL = keras.models.load_model(model_path)
        print("✅ Model MobileNetV2 berhasil dimuat.")
    else:
        print("⚠️  Model tidak ditemukan! Jalankan train_cnn.py terlebih dahulu.")

    # ── Class Names ───────────────────────────────────────────────────
    classes_path = "models/cnn_class_names.pkl"
    if os.path.exists(classes_path):
        with open(classes_path, "rb") as f:
            CLASS_NAMES = pickle.load(f)
        print(f"✅ Nama kelas dimuat ({len(CLASS_NAMES)} kelas).")

    # ── Disease DB ────────────────────────────────────────────────────
    db_path = "disease_database.json"
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            DISEASE_DB = json.load(f)
        print("✅ Database penyakit dimuat.")
    else:
        print("⚠️  Database penyakit tidak ditemukan!")


load_resources()


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocessing konsisten dengan train_cnn.py:
    - RGB, resize 224×224, normalisasi [0, 1]
    """
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def analyze_leaf_color(image_path: str) -> dict:
    """
    Analisis warna berbasis HSV untuk memvalidasi apakah gambar
    mengandung warna khas daun (hijau, kuning, coklat, oranye).

    Daun tomat sehat  → dominan hijau (H: 60°–150°)
    Daun menguning    → kuning/hijau-kuning (H: 40°–80°)
    Daun berpenyakit  → coklat/oranye/merah (H: 0°–40°, 150°–30°)
    Background tanah  → coklat (H: 10°–30°, S rendah)

    Bukan daun sama sekali → tidak ada pixel dalam rentang ini,
    atau pixel sangat seragam (foto dinding, kertas, wajah, dsb.).

    Returns:
        dict berisi:
            - leaf_ratio (float)  : rasio pixel berwarna daun
            - is_likely_leaf (bool): apakah kemungkinan daun
            - dominant_color (str) : deskripsi warna dominan
            - reason (str)         : alasan penolakan jika bukan daun
    """
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Downsample untuk efisiensi (tidak perlu full resolution)
    img_small = img.resize((64, 64), Image.Resampling.LANCZOS)
    rgb_array = np.array(img_small, dtype=np.float32) / 255.0

    total_pixels = 64 * 64
    leaf_pixels   = 0
    green_pixels  = 0
    yellow_pixels = 0
    brown_pixels  = 0
    gray_pixels   = 0  # pixel abu/putih/hitam (background netral)

    for row in rgb_array:
        for pixel in row:
            r, g, b = float(pixel[0]), float(pixel[1]), float(pixel[2])
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h_deg = h * 360  # 0–360

            # Pixel sangat gelap atau sangat terang = background/netral
            if v < 0.08 or v > 0.97:
                gray_pixels += 1
                continue

            # Saturasi sangat rendah → abu-abu/putih/hitam
            if s < 0.08:
                gray_pixels += 1
                continue

            # ── Kategorisasi warna daun ──────────────────────────
            # Hijau (H: 80°–165°) — daun sehat
            if 80 <= h_deg <= 165 and s >= 0.12:
                green_pixels += 1
                leaf_pixels  += 1

            # Kuning-hijau (H: 40°–80°) — daun menguning
            elif 40 <= h_deg < 80 and s >= 0.15:
                yellow_pixels += 1
                leaf_pixels   += 1

            # Coklat/oranye kemerahan (H: 0°–40° atau 330°–360°) — bercak/daun layu
            elif (h_deg <= 40 or h_deg >= 330) and s >= 0.20 and v >= 0.20:
                brown_pixels += 1
                leaf_pixels  += 1

    leaf_ratio  = leaf_pixels  / total_pixels
    green_ratio = green_pixels / total_pixels
    gray_ratio  = gray_pixels  / total_pixels

    # Tentukan warna dominan
    if green_pixels >= yellow_pixels and green_pixels >= brown_pixels:
        dominant = "Hijau (daun sehat)"
    elif yellow_pixels > green_pixels:
        dominant = "Kuning (daun menguning)"
    elif brown_pixels > green_pixels:
        dominant = "Coklat/oranye (bercak atau layu)"
    else:
        dominant = "Campuran tidak jelas"

    # Keputusan validasi
    is_likely_leaf = True
    reason = ""

    if leaf_ratio < COLOR_LEAF_MIN_RATIO:
        is_likely_leaf = False
        reason = (
            f"Gambar tidak mengandung cukup warna daun "
            f"(hanya {leaf_ratio*100:.1f}% pixel berwarna daun, "
            f"minimum {COLOR_LEAF_MIN_RATIO*100:.0f}%). "
            f"Gambar terlihat seperti bukan foto daun tanaman."
        )
    elif gray_ratio > 0.85:
        is_likely_leaf = False
        reason = (
            f"Gambar terlalu seragam/netral ({gray_ratio*100:.1f}% pixel abu/putih/hitam). "
            f"Pastikan foto menampilkan daun tomat yang jelas."
        )

    return {
        "leaf_ratio":      leaf_ratio,
        "green_ratio":     green_ratio,
        "is_likely_leaf":  is_likely_leaf,
        "dominant_color":  dominant,
        "reason":          reason,
    }


def validate_tomato_leaf(predictions: np.ndarray, class_names: list, image_path: str = None):
    """
    Validasi TIGA lapis apakah gambar merupakan daun tomat valid:

    Lapis 0 – Color/Texture Check (Heuristik Pra-Model):
        Analisis distribusi warna HSV pada gambar.
        Daun tomat (sehat/sakit) harus mengandung warna hijau,
        kuning, atau coklat yang cukup.
        Ini mendeteksi gambar non-tanaman (wajah, bangunan, dll.)
        SEBELUM diserahkan ke model CNN.

    Lapis 1 – Confidence Check:
        Max softmax probability harus ≥ MIN_CONFIDENCE_THRESHOLD (30%).
        Threshold diturunkan dari 55% ke 30% karena model memiliki
        akurasi ~51%; confidence tinggi untuk gambar yang sudah rusak
        memang tidak realistis.

    Lapis 2 – Entropy Check:
        Entropy distribusi softmax > 90% dari entropy maksimum
        → model sangat bingung → kemungkinan bukan daun tomat.

    Returns:
        (is_valid: bool, max_conf: float, entropy_ratio: float,
         pred_class: str, reason: str, color_info: dict)
    """
    max_conf  = float(np.max(predictions))
    pred_idx  = int(np.argmax(predictions))
    pred_cls  = class_names[pred_idx]
    n_classes = len(predictions)

    # Entropy
    eps = 1e-10
    entropy = -np.sum(predictions * np.log(predictions + eps))
    max_entropy = np.log(n_classes)  # ln(10) ≈ 2.303
    entropy_ratio = float(entropy / max_entropy)

    # ── Lapis 0: Validasi warna (heuristik) ─────────────────────────
    color_info = {"leaf_ratio": 0, "is_likely_leaf": True, "dominant_color": "N/A", "reason": ""}
    if image_path:
        try:
            color_info = analyze_leaf_color(image_path)
            if not color_info["is_likely_leaf"]:
                return False, max_conf, entropy_ratio, pred_cls, color_info["reason"], color_info
        except Exception as e:
            print(f"⚠️  Color analysis error: {e}")

    # ── Lapis 1: Confidence terlalu rendah ──────────────────────────
    if max_conf < MIN_CONFIDENCE_THRESHOLD:
        reason = (
            f"Tingkat keyakinan model terlalu rendah ({max_conf*100:.1f}% < "
            f"{MIN_CONFIDENCE_THRESHOLD*100:.0f}%). "
            "Gambar kemungkinan bukan daun tomat atau kualitas foto terlalu buruk."
        )
        return False, max_conf, entropy_ratio, pred_cls, reason, color_info

    # ── Lapis 2: Distribusi terlalu merata (model sangat bingung) ───
    if entropy_ratio > MAX_ENTROPY_RATIO:
        reason = (
            f"Distribusi prediksi terlalu merata (entropy {entropy_ratio*100:.1f}% > "
            f"{MAX_ENTROPY_RATIO*100:.0f}% threshold). "
            "Model tidak mengenali pola daun tomat pada gambar ini."
        )
        return False, max_conf, entropy_ratio, pred_cls, reason, color_info

    return True, max_conf, entropy_ratio, pred_cls, "OK", color_info


# ─────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/evaluation')
def evaluation():
    metrics = {}
    metrics_path = os.path.join('evaluation_results', 'metrics_summary.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    return render_template('evaluation.html', metrics=metrics)


@app.route('/evaluation_results/<path:filename>')
def serve_evaluation_file(filename):
    """Serve file dari folder evaluation_results (confusion matrix dll)."""
    return send_from_directory('evaluation_results', filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    # ── Guard: pastikan sistem siap ──────────────────────────────────
    if MODEL is None or CLASS_NAMES is None or DISEASE_DB is None:
        return jsonify({'error': 'Sistem belum siap. Hubungi administrator.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih.'}), 400

    # ── Validasi tipe file ───────────────────────────────────────────
    allowed_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({
            'error': f'Format file tidak didukung ({ext}). Gunakan JPG, PNG, atau BMP.'
        }), 400

    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Simpan file sementara
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = secure_filename(f"{timestamp}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess & prediksi
        img_array   = preprocess_image(file_path)
        predictions = MODEL.predict(img_array, verbose=0)[0]

        # ── VALIDASI GAMBAR (3 lapis) ─────────────────────────────────
        is_valid, max_conf, entropy_ratio, pred_cls_raw, reason, color_info = validate_tomato_leaf(
            predictions, CLASS_NAMES, file_path
        )

        if not is_valid:
            # Simpan file untuk ditampilkan di preview (tidak dihapus)
            # agar pengguna bisa melihat gambar yang mereka upload
            return jsonify({
                'error': (
                    f'{reason}\n\n'
                    'Pastikan gambar yang diunggah adalah:\n'
                    '• Foto daun tomat yang jelas dan fokus\n'
                    '• Pencahayaan yang cukup (tidak terlalu gelap/terang)\n'
                    '• Latar belakang tidak terlalu ramai\n'
                    '• Resolusi minimal 100×100 piksel'
                ),
                'is_not_leaf': True,
                'confidence': max_conf,
                'entropy_ratio': entropy_ratio,
                'leaf_ratio': color_info.get('leaf_ratio', 0),
                'dominant_color': color_info.get('dominant_color', 'N/A'),
                'image_path': f'uploads/{filename}',
            }), 400

        # ── Mapping kelas → database ─────────────────────────────────
        db_key       = CLASS_TO_DB.get(pred_cls_raw, pred_cls_raw)
        disease_info = DISEASE_DB.get(db_key, {})
        confidence   = float(np.max(predictions))

        # Semua probabilitas untuk chart
        all_probabilities = {}
        for i, prob in enumerate(predictions):
            raw_name      = CLASS_NAMES[i]
            display_name  = CLASS_TO_DB.get(raw_name, raw_name)
            all_probabilities[display_name] = float(prob)

        # ── Bangun response ──────────────────────────────────────────
        # Pisahkan treatment (obat spesifik) dari prevention (pencegahan)
        # agar rekomendasi per penyakit tidak tertukar
        is_healthy = (db_key == "Sehat")

        result = {
            # ─ Info dasar penyakit ─
            "disease":          disease_info.get("nama_penyakit", db_key),
            "disease_key":      db_key,
            "is_healthy":       is_healthy,
            "confidence":       confidence,
            "confidence_pct":   f"{confidence*100:.1f}%",
            "confidence_level": "Tinggi" if confidence > 0.80 else "Sedang" if confidence > 0.60 else "Rendah",
            "entropy_ratio":    entropy_ratio,
            "severity":         disease_info.get("tingkat_keparahan", "Tidak diketahui"),

            # ─ Penyebab & gejala ─
            "causes":   disease_info.get("penyebab", "").split(" / "),
            "symptoms": disease_info.get("gejala", []),

            # ─ Pencegahan (umum, berlaku sebelum sakit) ─
            "prevention": disease_info.get("pencegahan", []),

            # ─ Penanganan (obat & tindakan, berlaku setelah terdeteksi sakit) ─
            # Diambil dari obat_details yang UNIK PER PENYAKIT di disease_database.json
            "recommended_medicines": disease_info.get("obat_details", []),
            "organic_alternatives":  disease_info.get("alternatif_organik", []),
            "medicine_notes":        disease_info.get("referensi", ""),
            "obat_list":             disease_info.get("obat", []),

            # ─ Deskripsi & urgensi ─
            "description": (
                "Tanaman tomat Anda dalam kondisi SEHAT. Pertahankan perawatan rutin!"
                if is_healthy else
                f"Tanaman terdeteksi mengidap {disease_info.get('nama_penyakit', db_key)}. "
                f"Segera lakukan penanganan sesuai rekomendasi di bawah."
            ),
            "urgency": (
                "Pantau & pertahankan perawatan rutin"
                if is_healthy else
                "⚠️ Segera tangani (dalam 24-48 jam)"
                if disease_info.get("tingkat_keparahan", "") in ["Tinggi", "Sangat Tinggi"]
                else "Tangani dalam 1-3 hari"
            ),

            # ─ Probabilitas semua kelas ─
            "all_probabilities": all_probabilities,
            "image_path":        f"uploads/{filename}",
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Terjadi kesalahan internal: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

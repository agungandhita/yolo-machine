from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import cv2
import numpy as np
import pickle
from PIL import Image
import base64
from io import BytesIO
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables untuk model
cnn_model = None
class_names = None
model_metadata = None

def load_cnn_model():
    """Load CNN model dan metadata"""
    global cnn_model, class_names, model_metadata
    
    try:
        # Load CNN model
        print("🔄 Loading CNN model...")
        cnn_model = keras.models.load_model('models/tomato_cnn_model.h5')
        print("✅ CNN model loaded successfully!")
        
        # Load class names
        with open('models/cnn_class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        print(f"✅ Class names loaded: {len(class_names)} classes")
        
        # Load metadata
        with open('models/cnn_metadata.pkl', 'rb') as f:
            model_metadata = pickle.load(f)
        print(f"✅ Model metadata loaded: {model_metadata['model_type']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading CNN model: {e}")
        print("🔄 Falling back to traditional ML model...")
        
        # Fallback ke model lama
        try:
            with open('models/tomato_disease_model.pkl', 'rb') as f:
                global fallback_model
                fallback_model = pickle.load(f)
            
            with open('models/class_names.pkl', 'rb') as f:
                class_names = pickle.load(f)
                
            print("✅ Fallback model loaded successfully!")
            return False
            
        except Exception as e2:
            print(f"❌ Error loading fallback model: {e2}")
            return False

def preprocess_image_for_cnn(image_path, target_size=(224, 224)):
    """
    Preprocessing gambar untuk CNN model dengan validasi yang lebih baik
    """
    try:
        print(f"🔄 Preprocessing image: {image_path}")
        
        # Validasi file exists
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None
            
        # Baca gambar menggunakan PIL untuk konsistensi
        img = Image.open(image_path)
        print(f"📷 Original image mode: {img.mode}, size: {img.size}")
        
        # Convert ke RGB jika perlu
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print(f"🔄 Converted to RGB mode")
        
        # Resize gambar dengan resampling yang baik
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        print(f"📏 Resized to: {target_size}")
        
        # Convert ke numpy array
        img_array = np.array(img)
        print(f"📊 Array shape before normalization: {img_array.shape}")
        print(f"📊 Pixel value range before normalization: {img_array.min()} - {img_array.max()}")
        
        # Normalisasi pixel values ke [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        print(f"📊 Pixel value range after normalization: {img_array.min():.3f} - {img_array.max():.3f}")
        
        # Tambahkan batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        print(f"📊 Final array shape: {img_array.shape}")
        
        return img_array
        
    except Exception as e:
        print(f"❌ Error preprocessing image for CNN: {e}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")
        return None

def extract_features_traditional(image_path, target_size=(64, 64)):
    """
    Ekstraksi fitur tradisional untuk fallback model
    """
    try:
        # Baca gambar
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Resize gambar
        img = cv2.resize(img, target_size)
        
        # Konversi ke HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Ekstraksi fitur statistik
        features = []
        
        # Fitur dari setiap channel HSV
        for channel in cv2.split(hsv):
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.min(channel),
                np.max(channel)
            ])
        
        # Fitur tekstur sederhana (gradien)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        features.extend([
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y)),
            np.std(grad_x),
            np.std(grad_y)
        ])
        
        return np.array(features).reshape(1, -1)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_disease_cnn(image_path):
    """
    Prediksi penyakit menggunakan CNN model dengan debug logging
    """
    global cnn_model, class_names, model_metadata
    
    print(f"🧠 Starting CNN prediction for: {image_path}")
    
    if cnn_model is None or class_names is None:
        print("❌ CNN model or class names not loaded")
        return {"error": "CNN model not loaded"}
    
    print(f"✅ Model loaded: {type(cnn_model)}")
    print(f"✅ Class names: {len(class_names)} classes")
    
    # Preprocessing gambar untuk CNN
    # Extract target size from input_shape (height, width, channels)
    target_size = model_metadata['input_shape'][:2]  # (224, 224)
    print(f"🎯 Target size from metadata: {target_size}")
    
    img_array = preprocess_image_for_cnn(image_path, target_size=target_size)
    if img_array is None:
        print("❌ Image preprocessing failed")
        return {"error": "Failed to process image"}
    
    try:
        print(f"🔮 Making prediction with input shape: {img_array.shape}")
        
        # Prediksi menggunakan CNN
        predictions = cnn_model.predict(img_array, verbose=0)[0]
        print(f"📊 Raw predictions shape: {predictions.shape}")
        print(f"📊 Raw predictions: {predictions}")
        print(f"📊 Predictions sum: {np.sum(predictions):.6f}")
        
        # Validasi dimensi prediksi
        if len(predictions) != len(class_names):
            error_msg = f"Model prediction dimension mismatch: {len(predictions)} vs {len(class_names)} classes"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions)
        print(f"🎯 Predicted class index: {predicted_class_idx}")
        
        # Validasi index prediksi
        if predicted_class_idx >= len(class_names):
            error_msg = f"Predicted class index out of range: {predicted_class_idx} >= {len(class_names)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
            
        predicted_class = class_names[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])
        print(f"🎯 Predicted class: {predicted_class}")
        print(f"🎯 Confidence: {confidence:.6f}")
        
        # Get top 3 predictions dengan validasi index
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        print(f"🏆 Top 3 indices: {top_3_indices}")
        
        top_3_predictions = []
        for i in top_3_indices:
            if i < len(class_names) and i < len(predictions):
                class_name = class_names[i]
                prob = float(predictions[i])
                top_3_predictions.append((class_name, prob))
                print(f"   {i}: {class_name} = {prob:.6f}")
            else:
                print(f"⚠️ Warning: Skipping invalid index {i} (class_names: {len(class_names)}, predictions: {len(predictions)})")
        
        # Format nama penyakit untuk display dalam bahasa Indonesia
        disease_translations = {
            "Tomato_Bacterial_spot": "Bercak Bakteri",
            "Tomato_Early_blight": "Hawar Daun Awal", 
            "Tomato_Late_blight": "Hawar Daun Lanjut",
            "Tomato_Leaf_Mold": "Jamur Daun",
            "Tomato_Septoria_leaf_spot": "Bercak Daun Septoria",
            "Tomato_Spider_mites_Two_spotted_spider_mite": "Tungau Laba-laba",
            "Tomato__Target_Spot": "Bercak Target",
            "Tomato__Tomato_YellowLeaf__Curl_Virus": "Virus Keriting Daun Kuning",
            "Tomato__Tomato_mosaic_virus": "Virus Mozaik Tomat",
            "Tomato_healthy": "Sehat"
        }
        
        disease_name = disease_translations.get(predicted_class, predicted_class)
        print(f"🏥 Disease name (Indonesian): {disease_name}")

        # Informasi lengkap untuk setiap penyakit
        disease_info = {
            "Bercak Bakteri": {
                "description": "Penyakit bakteri yang menyebabkan bercak-bercak kecil berwarna coklat dengan halo kuning pada daun.",
                "symptoms": ["Bercak kecil coklat dengan tepi kuning", "Daun menguning dan layu", "Buah dapat terinfeksi"],
                "causes": ["Kelembaban tinggi", "Suhu hangat 24-30°C", "Percikan air hujan", "Alat pertanian terkontaminasi"],
                "treatment": [
                    "Semprotkan fungisida berbahan tembaga",
                    "Perbaiki drainase untuk mengurangi kelembaban",
                    "Buang dan musnahkan daun yang terinfeksi",
                    "Hindari penyiraman dari atas",
                    "Rotasi tanaman dengan tanaman non-solanaceae"
                ],
                "prevention": ["Jaga kebersihan kebun", "Hindari kelembaban berlebih", "Sterilisasi alat pertanian"],
                "severity": "Sedang",
                "urgency": "Segera tangani dalam 1-2 hari"
            },
            "Hawar Daun Awal": {
                "description": "Penyakit jamur yang menyebabkan bercak coklat dengan pola lingkaran konsentris pada daun.",
                "symptoms": ["Bercak coklat dengan pola lingkaran", "Daun menguning dan gugur", "Dapat menyerang buah"],
                "causes": ["Kelembaban tinggi", "Suhu 24-29°C", "Daun yang lembab", "Sirkulasi udara buruk"],
                "treatment": [
                    "Aplikasi fungisida sistemik",
                    "Buang daun yang terinfeksi",
                    "Perbaiki sirkulasi udara",
                    "Kurangi kelembaban dengan mulsa",
                    "Penyiraman di pagi hari"
                ],
                "prevention": ["Jarak tanam yang cukup", "Pemangkasan untuk sirkulasi udara", "Hindari penyiraman malam"],
                "severity": "Sedang-Tinggi",
                "urgency": "Tangani dalam 1-2 hari"
            },
            "Hawar Daun Lanjut": {
                "description": "Penyakit jamur serius yang dapat merusak seluruh tanaman dengan cepat.",
                "symptoms": ["Bercak coklat kehitaman", "Daun layu dan mati", "Buah membusuk", "Pertumbuhan jamur putih"],
                "causes": ["Kelembaban sangat tinggi", "Suhu dingin 15-20°C", "Hujan berkepanjangan", "Kondisi lembab"],
                "treatment": [
                    "Fungisida sistemik segera",
                    "Buang semua bagian terinfeksi",
                    "Perbaiki drainase",
                    "Kurangi kelembaban drastis",
                    "Isolasi tanaman terinfeksi"
                ],
                "prevention": ["Varietas tahan penyakit", "Drainase yang baik", "Hindari kelembaban berlebih"],
                "severity": "Sangat Tinggi",
                "urgency": "DARURAT - Tangani segera!"
            },
            "Jamur Daun": {
                "description": "Jamur yang menyebabkan bercak kuning pada permukaan atas daun dan pertumbuhan jamur di bawah daun.",
                "symptoms": ["Bercak kuning pada daun", "Jamur abu-abu di bawah daun", "Daun mengering dan gugur"],
                "causes": ["Kelembaban tinggi", "Sirkulasi udara buruk", "Suhu hangat", "Kondisi lembab berkepanjangan"],
                "treatment": [
                    "Fungisida khusus jamur daun",
                    "Perbaiki ventilasi",
                    "Kurangi kelembaban",
                    "Buang daun terinfeksi",
                    "Penyiraman dari bawah"
                ],
                "prevention": ["Sirkulasi udara baik", "Jarak tanam cukup", "Hindari kelembaban berlebih"],
                "severity": "Sedang",
                "urgency": "Tangani dalam 2-3 hari"
            },
            "Bercak Daun Septoria": {
                "description": "Penyakit jamur yang menyebabkan bercak kecil dengan titik hitam di tengah pada daun.",
                "symptoms": ["Bercak kecil dengan titik hitam", "Daun menguning", "Gugur daun dari bawah ke atas"],
                "causes": ["Kelembaban tinggi", "Percikan air", "Suhu hangat", "Daun yang lembab"],
                "treatment": [
                    "Fungisida preventif",
                    "Buang daun bawah yang terinfeksi",
                    "Mulsa untuk mencegah percikan",
                    "Penyiraman dari bawah",
                    "Perbaiki sirkulasi udara"
                ],
                "prevention": ["Mulsa yang baik", "Hindari penyiraman dari atas", "Jarak tanam cukup"],
                "severity": "Sedang",
                "urgency": "Tangani dalam 2-3 hari"
            },
            "Tungau Laba-laba": {
                "description": "Hama tungau yang menyebabkan bercak kuning dan jaring halus pada daun.",
                "symptoms": ["Bercak kuning kecil", "Jaring halus di daun", "Daun mengering", "Pertumbuhan terhambat"],
                "causes": ["Cuaca kering", "Kelembaban rendah", "Suhu tinggi", "Kurang air"],
                "treatment": [
                    "Semprotkan air untuk menghilangkan tungau",
                    "Mitisida jika serangan parah",
                    "Tingkatkan kelembaban",
                    "Predator alami seperti kepik",
                    "Penyiraman teratur"
                ],
                "prevention": ["Jaga kelembaban", "Penyiraman teratur", "Monitoring rutin"],
                "severity": "Sedang",
                "urgency": "Tangani dalam 3-5 hari"
            },
            "Bercak Target": {
                "description": "Penyakit jamur yang menyebabkan bercak dengan pola target atau lingkaran konsentris.",
                "symptoms": ["Bercak dengan pola lingkaran", "Warna coklat dengan tepi gelap", "Daun menguning"],
                "causes": ["Kelembaban tinggi", "Suhu hangat", "Luka pada tanaman", "Kondisi stress"],
                "treatment": [
                    "Fungisida sistemik",
                    "Buang bagian terinfeksi",
                    "Perbaiki kondisi tanaman",
                    "Kurangi stress pada tanaman",
                    "Perbaiki nutrisi"
                ],
                "prevention": ["Hindari luka pada tanaman", "Nutrisi seimbang", "Jaga kesehatan tanaman"],
                "severity": "Sedang",
                "urgency": "Tangani dalam 2-3 hari"
            },
            "Virus Keriting Daun Kuning": {
                "description": "Virus yang menyebabkan daun mengkerut, menguning, dan pertumbuhan terhambat.",
                "symptoms": ["Daun mengkerut dan menggulung", "Warna kuning", "Pertumbuhan kerdil", "Produksi buah menurun"],
                "causes": ["Serangga vektor (kutu kebul)", "Tanaman terinfeksi", "Alat pertanian terkontaminasi"],
                "treatment": [
                    "Tidak ada obat langsung untuk virus",
                    "Buang tanaman terinfeksi",
                    "Kendalikan serangga vektor",
                    "Isolasi tanaman sehat",
                    "Gunakan varietas tahan virus"
                ],
                "prevention": ["Kendalikan kutu kebul", "Gunakan bibit sehat", "Isolasi tanaman baru"],
                "severity": "Tinggi",
                "urgency": "Segera isolasi dan buang tanaman"
            },
            "Virus Mozaik Tomat": {
                "description": "Virus yang menyebabkan pola mozaik hijau terang dan gelap pada daun.",
                "symptoms": ["Pola mozaik pada daun", "Daun keriting", "Pertumbuhan tidak normal", "Buah cacat"],
                "causes": ["Kontak langsung", "Alat pertanian", "Serangga", "Tangan yang terkontaminasi"],
                "treatment": [
                    "Buang tanaman terinfeksi",
                    "Sterilisasi alat pertanian",
                    "Cuci tangan sebelum menyentuh tanaman",
                    "Isolasi tanaman sehat",
                    "Gunakan varietas tahan"
                ],
                "prevention": ["Sterilisasi alat", "Cuci tangan", "Gunakan bibit bersertifikat"],
                "severity": "Tinggi",
                "urgency": "Segera buang tanaman terinfeksi"
            },
            "Sehat": {
                "description": "Tanaman dalam kondisi sehat tanpa tanda-tanda penyakit atau hama.",
                "symptoms": ["Daun hijau segar", "Pertumbuhan normal", "Tidak ada bercak atau kerusakan"],
                "causes": ["Perawatan yang baik", "Kondisi lingkungan optimal", "Nutrisi cukup"],
                "treatment": [
                    "Lanjutkan perawatan rutin",
                    "Monitoring berkala",
                    "Jaga kebersihan lingkungan",
                    "Pemupukan sesuai jadwal",
                    "Penyiraman teratur"
                ],
                "prevention": ["Perawatan preventif", "Monitoring rutin", "Sanitasi lingkungan"],
                "severity": "Tidak ada",
                "urgency": "Perawatan rutin"
            }
        }
        
        # Ambil informasi lengkap penyakit
        info = disease_info.get(disease_name, {})
        description = info.get("description", "Informasi tidak tersedia untuk penyakit ini.")
        
        # Tentukan level kepercayaan
        if confidence > 0.9:
            confidence_level = "Sangat Tinggi"
        elif confidence > 0.8:
            confidence_level = "Tinggi"
        elif confidence > 0.6:
            confidence_level = "Sedang"
        else:
            confidence_level = "Rendah"
        
        # Translate all probabilities to Indonesian
        translated_probabilities = {}
        for i, prob in enumerate(predictions):
            original_name = class_names[i]
            translated_name = disease_translations.get(original_name, original_name)
            translated_probabilities[translated_name] = float(prob)
        
        # Top 3 diseases dengan nama Indonesia
        top_3_diseases = [(disease_translations.get(name, name), prob) for name, prob in top_3_predictions]
        
        # Buat rekomendasi berdasarkan probabilitas
        recommendations = []
        if confidence > 0.8:
            recommendations.append(f"Diagnosis dengan CNN: {disease_name} (tingkat kepercayaan {confidence_level.lower()})")
        else:
            recommendations.append(f"Kemungkinan: {disease_name} - disarankan pemeriksaan lebih lanjut")
            
        if len(top_3_diseases) > 1 and top_3_diseases[1][1] > 0.1:
            recommendations.append(f"Kemungkinan alternatif: {top_3_diseases[1][0]} ({top_3_diseases[1][1]*100:.1f}%)")
        
        return {
            "disease": disease_name,
            "confidence": float(confidence),
            "confidence_level": confidence_level,
            "description": description,
            "symptoms": info.get("symptoms", []),
            "causes": info.get("causes", []),
            "treatment": info.get("treatment", []),
            "prevention": info.get("prevention", []),
            "severity": info.get("severity", "Tidak diketahui"),
            "urgency": info.get("urgency", "Konsultasi dengan ahli"),
            "all_probabilities": translated_probabilities,
            "top_3_diseases": top_3_diseases,
            "recommendations": recommendations,
            "is_healthy": disease_name == "Sehat",
            "model_type": "CNN (Deep Learning)"
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ CNN prediction error: {str(e)}")
        print(f"📋 Error details: {error_details}")
        return {"error": f"CNN prediction failed: {str(e)}", "details": error_details}

def predict_disease_fallback(image_path):
    """
    Prediksi penyakit menggunakan traditional ML model sebagai fallback dengan debug logging
    """
    global traditional_model, class_names
    
    print(f"🔄 Starting traditional model prediction for: {image_path}")
    
    if traditional_model is None or class_names is None:
        print("❌ Traditional model or class names not loaded")
        return {"error": "Traditional model not loaded"}
    
    print(f"✅ Traditional model loaded: {type(traditional_model)}")
    print(f"✅ Class names: {len(class_names)} classes")
    
    try:
        # Extract features menggunakan traditional method
        print("🔍 Extracting traditional features...")
        features = extract_features_traditional(image_path)
        if features is None:
            print("❌ Feature extraction failed")
            return {"error": "Failed to extract features"}
        
        print(f"📊 Extracted features shape: {features.shape}")
        print(f"📊 Feature range: {features.min():.3f} - {features.max():.3f}")
        
        # Prediksi menggunakan traditional model
        print("🔮 Making prediction with traditional model...")
        prediction_proba = traditional_model.predict_proba([features])[0]
        print(f"📊 Prediction probabilities shape: {prediction_proba.shape}")
        print(f"📊 Prediction probabilities: {prediction_proba}")
        print(f"📊 Probabilities sum: {np.sum(prediction_proba):.6f}")
        
        # Validasi dimensi prediksi
        if len(prediction_proba) != len(class_names):
            error_msg = f"Traditional model prediction dimension mismatch: {len(prediction_proba)} vs {len(class_names)} classes"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
        
        # Get predicted class
        predicted_class_idx = np.argmax(prediction_proba)
        print(f"🎯 Predicted class index: {predicted_class_idx}")
        
        # Validasi index prediksi
        if predicted_class_idx >= len(class_names):
            error_msg = f"Predicted class index out of range: {predicted_class_idx} >= {len(class_names)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
            
        predicted_class = class_names[predicted_class_idx]
        confidence = float(prediction_proba[predicted_class_idx])
        print(f"🎯 Predicted class: {predicted_class}")
        print(f"🎯 Confidence: {confidence:.6f}")
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction_proba)[-3:][::-1]
        print(f"🏆 Top 3 indices: {top_3_indices}")
        
        top_3_predictions = []
        for i in top_3_indices:
            if i < len(class_names) and i < len(prediction_proba):
                class_name = class_names[i]
                prob = float(prediction_proba[i])
                top_3_predictions.append((class_name, prob))
                print(f"   {i}: {class_name} = {prob:.6f}")
            else:
                print(f"⚠️ Warning: Skipping invalid index {i} (class_names: {len(class_names)}, predictions: {len(prediction_proba)})")
        
        # Format nama penyakit untuk display dalam bahasa Indonesia
        disease_translations = {
            "Tomato_Bacterial_spot": "Bercak Bakteri",
            "Tomato_Early_blight": "Hawar Daun Awal", 
            "Tomato_Late_blight": "Hawar Daun Lanjut",
            "Tomato_Leaf_Mold": "Jamur Daun",
            "Tomato_Septoria_leaf_spot": "Bercak Daun Septoria",
            "Tomato_Spider_mites_Two_spotted_spider_mite": "Tungau Laba-laba",
            "Tomato__Target_Spot": "Bercak Target",
            "Tomato__Tomato_YellowLeaf__Curl_Virus": "Virus Keriting Daun Kuning",
            "Tomato__Tomato_mosaic_virus": "Virus Mozaik Tomat",
            "Tomato_healthy": "Sehat"
        }
        
        disease_name = disease_translations.get(predicted_class, predicted_class)
        print(f"🏥 Disease name (Indonesian): {disease_name}")

        # Informasi lengkap untuk setiap penyakit (simplified version)
        disease_info = {
            "Bercak Bakteri": {
                "description": "Penyakit bakteri yang menyebabkan bercak-bercak kecil berwarna coklat dengan halo kuning pada daun.",
                "symptoms": ["Bercak kecil coklat dengan tepi kuning", "Daun menguning dan layu"],
                "treatment": ["Semprot fungisida berbahan tembaga", "Buang daun yang terinfeksi"],
                "severity": "Sedang"
            },
            "Hawar Daun Awal": {
                "description": "Penyakit jamur yang menyebabkan bercak coklat dengan pola lingkaran konsentris.",
                "symptoms": ["Bercak coklat dengan pola target", "Daun menguning dari bawah"],
                "treatment": ["Fungisida sistemik", "Perbaiki drainase"],
                "severity": "Sedang"
            },
            "Hawar Daun Lanjut": {
                "description": "Penyakit jamur yang sangat merusak, menyebabkan bercak coklat kehitaman.",
                "symptoms": ["Bercak coklat kehitaman", "Daun layu dengan cepat"],
                "treatment": ["Fungisida khusus", "Isolasi tanaman"],
                "severity": "Tinggi"
            },
            "Jamur Daun": {
                "description": "Infeksi jamur yang menyebabkan lapisan putih kekuningan pada permukaan daun.",
                "symptoms": ["Lapisan putih pada daun", "Daun keriting"],
                "treatment": ["Fungisida", "Tingkatkan ventilasi"],
                "severity": "Sedang"
            },
            "Bercak Daun Septoria": {
                "description": "Penyakit jamur yang menyebabkan bercak kecil dengan titik hitam di tengah.",
                "symptoms": ["Bercak kecil dengan titik hitam", "Daun menguning"],
                "treatment": ["Fungisida", "Buang daun terinfeksi"],
                "severity": "Sedang"
            },
            "Tungau Laba-laba": {
                "description": "Serangan hama tungau yang menyebabkan bintik-bintik kuning pada daun.",
                "symptoms": ["Bintik kuning kecil", "Jaring laba-laba halus"],
                "treatment": ["Mitisida", "Semprot air"],
                "severity": "Sedang"
            },
            "Bercak Target": {
                "description": "Penyakit jamur yang menyebabkan bercak dengan pola target yang jelas.",
                "symptoms": ["Bercak dengan pola lingkaran", "Daun berlubang"],
                "treatment": ["Fungisida", "Sanitasi kebun"],
                "severity": "Sedang"
            },
            "Virus Keriting Daun Kuning": {
                "description": "Infeksi virus yang menyebabkan daun menguning dan keriting.",
                "symptoms": ["Daun menguning", "Daun keriting ke atas"],
                "treatment": ["Kontrol vektor", "Buang tanaman terinfeksi"],
                "severity": "Tinggi"
            },
            "Virus Mozaik Tomat": {
                "description": "Infeksi virus yang menyebabkan pola mozaik pada daun.",
                "symptoms": ["Pola mozaik hijau-kuning", "Pertumbuhan terhambat"],
                "treatment": ["Kontrol vektor", "Sanitasi"],
                "severity": "Tinggi"
            },
            "Sehat": {
                "description": "Tanaman dalam kondisi sehat tanpa tanda-tanda penyakit.",
                "symptoms": ["Daun hijau segar", "Pertumbuhan normal"],
                "treatment": ["Perawatan rutin", "Pemupukan teratur"],
                "severity": "Tidak ada"
            }
        }
        
        info = disease_info.get(disease_name, {})
        description = info.get("description", "Informasi tidak tersedia")
        
        # Confidence level
        if confidence > 0.9:
            confidence_level = "Sangat Tinggi"
        elif confidence > 0.8:
            confidence_level = "Tinggi"
        elif confidence > 0.6:
            confidence_level = "Sedang"
        else:
            confidence_level = "Rendah"
        
        # Translate all probabilities to Indonesian
        translated_probabilities = {}
        for i, prob in enumerate(probabilities):
            original_name = class_names[i]
            translated_name = disease_translations.get(original_name, original_name)
            translated_probabilities[translated_name] = float(prob)
        
        # Top 3 diseases dengan nama Indonesia
        top_3_diseases = [(disease_translations.get(name, name), prob) for name, prob in top_3_predictions]
        
        # Buat rekomendasi berdasarkan probabilitas
        recommendations = []
        if confidence > 0.8:
            recommendations.append(f"Diagnosis dengan Traditional ML: {disease_name} (tingkat kepercayaan {confidence_level.lower()})")
        else:
            recommendations.append(f"Kemungkinan: {disease_name} - disarankan pemeriksaan lebih lanjut")
            
        if len(top_3_diseases) > 1 and top_3_diseases[1][1] > 0.1:
            recommendations.append(f"Kemungkinan alternatif: {top_3_diseases[1][0]} ({top_3_diseases[1][1]*100:.1f}%)")
        
        return {
            "disease": disease_name,
            "confidence": float(confidence),
            "confidence_level": confidence_level,
            "description": description,
            "symptoms": info.get("symptoms", []),
            "causes": info.get("causes", []),
            "treatment": info.get("treatment", []),
            "prevention": info.get("prevention", []),
            "severity": info.get("severity", "Tidak diketahui"),
            "urgency": info.get("urgency", "Konsultasi dengan ahli"),
            "all_probabilities": translated_probabilities,
            "top_3_diseases": top_3_diseases,
            "recommendations": recommendations,
            "is_healthy": disease_name == "Sehat",
            "model_type": "Traditional ML (Fallback)"
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Fallback prediction error: {str(e)}")
        print(f"📋 Error details: {error_details}")
        return {"error": f"Fallback prediction failed: {str(e)}", "details": error_details}

def predict_disease(image_path):
    """
    Fungsi utama untuk prediksi penyakit dengan debug logging
    Mencoba CNN terlebih dahulu, jika gagal menggunakan traditional model
    """
    print(f"\n{'='*60}")
    print(f"🚀 STARTING DISEASE PREDICTION")
    print(f"📁 Image path: {image_path}")
    print(f"{'='*60}")
    
    # Validasi file exists
    if not os.path.exists(image_path):
        error_msg = f"Image file not found: {image_path}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}
    
    # Coba CNN model terlebih dahulu
    print(f"\n🧠 ATTEMPTING CNN PREDICTION...")
    print(f"CNN Model loaded: {cnn_model is not None}")
    print(f"CNN Class names loaded: {class_names is not None}")
    
    if cnn_model is not None and class_names is not None:
        try:
            print("✅ CNN model available, proceeding with CNN prediction...")
            result = predict_disease_cnn(image_path)
            
            if "error" not in result:
                print(f"✅ CNN prediction successful!")
                print(f"🎯 Result: {result.get('disease', 'Unknown')} ({result.get('confidence', 0):.3f})")
                print(f"{'='*60}\n")
                return result
            else:
                print(f"❌ CNN prediction failed: {result['error']}")
                
        except Exception as e:
            print(f"❌ CNN prediction exception: {e}")
            import traceback
            print(f"📋 Traceback: {traceback.format_exc()}")
    else:
        print("❌ CNN model or class names not available")
    
    # Fallback ke traditional model
    print(f"\n🔄 FALLING BACK TO TRADITIONAL MODEL...")
    print(f"Traditional Model loaded: {traditional_model is not None}")
    
    if traditional_model is not None and class_names is not None:
        try:
            print("✅ Traditional model available, proceeding with traditional prediction...")
            result = predict_disease_fallback(image_path)
            
            if "error" not in result:
                print(f"✅ Traditional prediction successful!")
                print(f"🎯 Result: {result.get('disease', 'Unknown')} ({result.get('confidence', 0):.3f})")
                print(f"{'='*60}\n")
                return result
            else:
                print(f"❌ Traditional prediction failed: {result['error']}")
                
        except Exception as e:
            print(f"❌ Traditional prediction exception: {e}")
            import traceback
            print(f"📋 Traceback: {traceback.format_exc()}")
    else:
        print("❌ Traditional model or class names not available")
    
    # Jika semua model gagal
    error_msg = "All prediction models failed"
    print(f"❌ {error_msg}")
    print(f"{'='*60}\n")
    return {"error": error_msg}

# Load model saat aplikasi start
print("🚀 Starting TomatoDoc AI with CNN support...")
cnn_available = load_cnn_model()

if cnn_available:
    print("✅ CNN model loaded successfully!")
else:
    print("⚠️ Using fallback traditional ML model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Route untuk upload dan prediksi gambar dengan debug logging
    """
    print(f"\n🌐 NEW UPLOAD REQUEST RECEIVED")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if 'file' not in request.files:
        error_msg = 'No file part in request'
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 400
    
    file = request.files['file']
    print(f"📁 File received: {file.filename}")
    
    if file.filename == '':
        error_msg = 'No file selected'
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename dengan timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = secure_filename(file.filename)
            filename = f"{timestamp}_{original_filename}"
            
            print(f"💾 Saving file as: {filename}")
            
            # Pastikan upload directory exists
            upload_dir = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
                print(f"📁 Created upload directory: {upload_dir}")
            
            # Save file
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            print(f"✅ File saved to: {file_path}")
            
            # Verify file was saved
            if not os.path.exists(file_path):
                error_msg = f"Failed to save file to {file_path}"
                print(f"❌ {error_msg}")
                return jsonify({'error': error_msg}), 500
            
            # Get file size for logging
            file_size = os.path.getsize(file_path)
            print(f"📊 File size: {file_size} bytes")
            
            # Prediksi penyakit
            print(f"🔮 Starting disease prediction...")
            result = predict_disease(file_path)
            
            if 'error' in result:
                print(f"❌ Prediction error: {result['error']}")
                return jsonify(result), 500
            
            # Add file info to result
            result['filename'] = filename
            result['file_path'] = file_path
            result['image_path'] = f"uploads/{filename}"  # Add this for frontend
            result['file_size'] = file_size
            result['timestamp'] = timestamp
            
            print(f"✅ Prediction completed successfully!")
            print(f"🎯 Final result: {result.get('disease', 'Unknown')}")
            print(f"🌐 UPLOAD REQUEST COMPLETED\n")
            
            return jsonify(result)
            
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            print(f"📋 Traceback: {traceback.format_exc()}")
            return jsonify({'error': error_msg}), 500
    else:
        error_msg = f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 400

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/about')
def about():
    return render_template('about.html')

# Global error handlers untuk memastikan API selalu return JSON
@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error occurred'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    # Untuk request API (yang mengharapkan JSON), return JSON error
    if request.path.startswith('/upload') or request.headers.get('Content-Type', '').startswith('multipart/form-data'):
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    # Untuk request HTML biasa, biarkan Flask handle secara normal
    raise e

if __name__ == '__main__':
    # Buat direktori upload jika belum ada
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print(f"🌐 Starting server on http://0.0.0.0:5000")
    print(f"🤖 Model type: {'CNN (Deep Learning)' if cnn_available else 'Traditional ML'}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
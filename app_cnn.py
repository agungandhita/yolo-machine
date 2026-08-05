from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
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

# Import YOLO predictor
from yolo_predictor import predict_disease_yolo, load_yolo_model, is_model_loaded as is_yolo_loaded

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables untuk model
cnn_model = None
class_names = None
model_metadata = None
medicine_database = None
disease_info_database = None
yolo_available = False  # YOLO model availability flag
fallback_model = None   # Fallback traditional ML model

def load_medicine_database():
    """Load database rekomendasi obat dari JSON file"""
    global medicine_database
    
    try:
        medicine_file = 'medicine_recommendation.json'
        
        if not os.path.exists(medicine_file):
            print(f"⚠️ Warning: {medicine_file} not found. Medicine recommendations will not be available.")
            medicine_database = {}
            return False
        
        with open(medicine_file, 'r', encoding='utf-8') as f:
            medicine_database = json.load(f)
        
        print(f"✅ Medicine database loaded: {len(medicine_database)} diseases with recommendations")
        return True
        
    except Exception as e:
        print(f"❌ Error loading medicine database: {e}")
        medicine_database = {}
        return False

def load_disease_info_database():
    """Load database informasi penyakit dari JSON file"""
    global disease_info_database
    
    try:
        disease_file = 'disease_info.json'
        
        if not os.path.exists(disease_file):
            print(f"⚠️ Warning: {disease_file} not found. Disease info will not be available.")
            disease_info_database = {}
            return False
        
        with open(disease_file, 'r', encoding='utf-8') as f:
            disease_info_database = json.load(f)
        
        print(f"✅ Disease info database loaded: {len(disease_info_database)} diseases")
        return True
        
    except Exception as e:
        print(f"❌ Error loading disease info database: {e}")
        disease_info_database = {}
        return False

def get_medicine_recommendation(disease_name, confidence):
    """
    Ambil rekomendasi obat berdasarkan nama penyakit dan confidence level
    
    Args:
        disease_name: Nama penyakit dalam bahasa Indonesia
        confidence: Tingkat kepercayaan prediksi (0-1)
    
    Returns:
        Dictionary berisi rekomendasi obat dan catatan
    """
    global medicine_database
    
    print(f"🔍 Getting medicine recommendation for: {disease_name} (confidence: {confidence:.3f})")
    
    # Default response jika database tidak tersedia
    if medicine_database is None or len(medicine_database) == 0:
        return {
            "recommended_medicines": [],
            "organic_alternatives": [],
            "medicine_notes": "Database rekomendasi obat tidak tersedia. Silakan konsultasi dengan ahli pertanian.",
            "category": "Unknown"
        }
    
    # Cek apakah penyakit ada di database
    if disease_name not in medicine_database:
        print(f"⚠️ Disease '{disease_name}' not found in medicine database")
        return {
            "recommended_medicines": [],
            "organic_alternatives": [],
            "medicine_notes": f"Rekomendasi obat untuk '{disease_name}' belum tersedia. Silakan konsultasi dengan ahli pertanian atau toko pertanian terdekat.",
            "category": "Unknown"
        }
    
    disease_data = medicine_database[disease_name]
    category = disease_data.get("category", "Unknown")
    recommended_medicines = disease_data.get("recommended_medicines", [])
    organic_alternatives = disease_data.get("organic_alternatives", [])
    
    print(f"📊 Found {len(recommended_medicines)} chemical medicines and {len(organic_alternatives)} organic alternatives")
    
    # Buat catatan berdasarkan confidence level dan kategori penyakit
    medicine_notes = []
    
    # Catatan confidence
    if confidence < 0.6:
        medicine_notes.append("⚠️ PERINGATAN: Tingkat kepercayaan prediksi rendah. Sangat disarankan untuk konsultasi dengan ahli pertanian sebelum menggunakan obat apapun.")
    elif confidence < 0.8:
        medicine_notes.append("ℹ️ Tingkat kepercayaan prediksi sedang. Disarankan untuk memverifikasi diagnosa dengan ahli jika memungkinkan.")
    else:
        medicine_notes.append("✅ Tingkat kepercayaan prediksi tinggi. Rekomendasi obat di bawah ini dapat dipertimbangkan.")
    
    # Catatan khusus berdasarkan kategori
    if category == "Virus":
        medicine_notes.append("")
        medicine_notes.append("🦠 PENTING - PENYAKIT VIRUS:")
        medicine_notes.append("• Tidak ada obat yang dapat membunuh virus pada tanaman")
        medicine_notes.append("• Obat yang direkomendasikan adalah untuk MENGENDALIKAN SERANGGA VEKTOR (pembawa virus)")
        medicine_notes.append("• Tanaman yang terinfeksi berat HARUS SEGERA DIBUANG dan DIMUSNAHKAN untuk mencegah penyebaran")
        medicine_notes.append("• Fokus pada PENCEGAHAN dengan mengendalikan populasi serangga vektor")
        medicine_notes.append("• Isolasi tanaman yang terinfeksi dari tanaman sehat")
    
    elif category == "Bakteri":
        medicine_notes.append("")
        medicine_notes.append("🔬 CATATAN PENYAKIT BAKTERI:")
        medicine_notes.append("• Gunakan bakterisida berbahan tembaga sebagai pilihan utama")
        medicine_notes.append("• Hindari penyiraman dari atas yang dapat menyebarkan bakteri")
        medicine_notes.append("• Sanitasi alat pertanian sangat penting (sterilisasi dengan alkohol atau api)")
        medicine_notes.append("• Buang dan musnahkan bagian tanaman yang terinfeksi")
    
    elif category == "Jamur":
        medicine_notes.append("")
        medicine_notes.append("🍄 CATATAN PENYAKIT JAMUR:")
        medicine_notes.append("• Lakukan penyemprotan fungisida secara merata, termasuk bagian bawah daun")
        medicine_notes.append("• Rotasi bahan aktif fungisida untuk mencegah resistensi")
        medicine_notes.append("• Kurangi kelembaban di sekitar tanaman")
    
    elif category == "Hama":
        medicine_notes.append("")
        medicine_notes.append("🐛 CATATAN PENGENDALIAN HAMA:")
        medicine_notes.append("• Semprotkan insektisida/mitisida pada sore hari saat hama aktif")
        medicine_notes.append("• Pastikan seluruh bagian tanaman tercover")
        medicine_notes.append("• Gunakan perekat jika perlu")
    
    return {
        "recommended_medicines": recommended_medicines,
        "organic_alternatives": organic_alternatives,
        "medicine_notes": "\n".join(medicine_notes),
        "category": category
    }

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

        # Use global disease info
        global disease_info_database
        if disease_info_database is None:
             load_disease_info_database()
        
        # Ambil informasi lengkap penyakit
        info = disease_info_database.get(disease_name, {})
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
        
        # Get medicine recommendations
        medicine_info = get_medicine_recommendation(disease_name, confidence)
        
        return {
            "disease": disease_name,
            "confidence": float(confidence),
            "confidence_level": confidence_level,
            "description": description,
            "symptoms": info.get("symptoms", []),
            "causes": info.get("causes", []),
            "treatment": info.get("treatment", []),
            "medication": info.get("medication", []),
            "prevention": info.get("prevention", []),
            "severity": info.get("severity", "Tidak diketahui"),
            "urgency": info.get("urgency", "Konsultasi dengan ahli"),
            "all_probabilities": translated_probabilities,
            "top_3_diseases": top_3_diseases,
            "recommendations": recommendations,
            "is_healthy": disease_name == "Sehat",
            "model_type": "CNN (Deep Learning)",
            # New medicine recommendation fields
            "recommended_medicines": medicine_info["recommended_medicines"],
            "organic_alternatives": medicine_info["organic_alternatives"],
            "medicine_notes": medicine_info["medicine_notes"],
            "disease_category": medicine_info["category"]
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
    global fallback_model, class_names
    
    print(f"🔄 Starting traditional model prediction for: {image_path}")
    
    if fallback_model is None or class_names is None:
        print("❌ Traditional model or class names not loaded")
        return {"error": "Traditional model not loaded"}
    
    print(f"✅ Traditional model loaded: {type(fallback_model)}")
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
        prediction_proba = fallback_model.predict_proba([features])[0]
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

        # Use global disease info
        global disease_info_database
        if disease_info_database is None:
             load_disease_info_database()
        
        info = disease_info_database.get(disease_name, {})
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
        for i, prob in enumerate(prediction_proba):
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
        
        # Get medicine recommendations
        medicine_info = get_medicine_recommendation(disease_name, confidence)

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
            "model_type": "Traditional ML (Fallback)",
            # New medicine recommendation fields
            "recommended_medicines": medicine_info["recommended_medicines"],
            "organic_alternatives": medicine_info["organic_alternatives"],
            "medicine_notes": medicine_info["medicine_notes"],
            "disease_category": medicine_info["category"]
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Fallback prediction error: {str(e)}")
        print(f"📋 Error details: {error_details}")
        return {"error": f"Fallback prediction failed: {str(e)}", "details": error_details}

def is_leaf_image(image_path):
    """
    Validasi out-of-distribution: mengecek apakah gambar dominan memiliki
    warna hijau/kuning/coklat (khas daun) menggunakan OpenCV HSV.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Resize untuk mempercepat komputasi
        img = cv2.resize(img, (224, 224))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Range warna hijau (daun sehat)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Range warna kuning/coklat/kering (daun berpenyakit)
        lower_brown = np.array([10, 30, 30])
        upper_brown = np.array([25, 255, 255])
        
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        
        mask_combined = cv2.bitwise_or(mask_green, mask_brown)
        
        # Hitung rasio piksel daun
        leaf_pixels = cv2.countNonZero(mask_combined)
        total_pixels = img.shape[0] * img.shape[1]
        ratio = leaf_pixels / total_pixels
        
        print(f"🌿 Leaf pixel ratio: {ratio:.3f}")
        return ratio > 0.05  # Minimal 5% warna daun
        
    except Exception as e:
        print(f"❌ Error in leaf detection: {e}")
        return True # Fallback true if error

def predict_disease(image_path):
    """
    Fungsi utama untuk prediksi penyakit dengan debug logging
    Mencoba YOLO terlebih dahulu, jika gagal mencoba CNN, lalu traditional model
    """
    global yolo_available
    global disease_info_database
    
    print(f"\n{'='*60}")
    print(f"🚀 STARTING DISEASE PREDICTION")
    print(f"📁 Image path: {image_path}")
    print(f"{'='*60}")
    
    # Validasi file exists
    if not os.path.exists(image_path):
        error_msg = f"Image file not found: {image_path}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}
    
    # Coba CNN model terlebih dahulu (PRIMARY karena akurasinya mungkin lebih baik)
    print(f"\n🎯 ATTEMPTING CNN PREDICTION...")
    print(f"CNN Model loaded: {cnn_model is not None}")
    
    if cnn_model is not None and class_names is not None:
        try:
            print("✅ CNN model available, proceeding with CNN prediction...")
            result = predict_disease_cnn(image_path)
            
            if "error" not in result:
                print(f"✅ CNN prediction successful!")
                print(f"🎯 Result: {result.get('disease', 'Unknown')} ({result.get('confidence', 0):.3f})")
                
                # Get medicine and disease info using disease_key
                disease_key = result.get('disease_key', result.get('disease'))
                medicine_info = get_medicine_recommendation(disease_key, result.get('confidence', 0))
                
                if disease_info_database is None:
                    load_disease_info_database()
                
                info = disease_info_database.get(disease_key, {})
                
                # Merge with result
                result['description'] = info.get("description", "Informasi tidak tersedia.")
                result['symptoms'] = info.get("symptoms", [])
                result['causes'] = info.get("causes", [])
                result['treatment'] = info.get("treatment", [])
                result['prevention'] = info.get("prevention", [])
                result['severity'] = info.get("severity", "Tidak diketahui")
                result['urgency'] = info.get("urgency", "Konsultasi dengan ahli")
                result['is_healthy'] = (disease_key == "Sehat")
                
                result['recommended_medicines'] = medicine_info.get('recommended_medicines', [])
                result['organic_alternatives'] = medicine_info.get('organic_alternatives', [])
                result['medicine_notes'] = medicine_info.get('medicine_notes', '')
                result['disease_category'] = medicine_info.get('category', '')
                
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
    
    # Coba YOLO model (FALLBACK 1)
    print(f"\n🎯 ATTEMPTING YOLO PREDICTION...")
    print(f"YOLO Model available: {yolo_available}")
    
    if yolo_available:
        try:
            print("✅ YOLO model available, proceeding with YOLO prediction...")
            result = predict_disease_yolo(image_path)
            
            if "error" not in result:
                print(f"✅ YOLO prediction successful!")
                print(f"🎯 Result: {result.get('disease', 'Unknown')} ({result.get('confidence', 0):.3f})")
                
                # Get medicine and disease info using disease_key
                disease_key = result.get('disease_key', result.get('disease'))
                medicine_info = get_medicine_recommendation(disease_key, result.get('confidence', 0))
                
                if disease_info_database is None:
                    load_disease_info_database()
                
                info = disease_info_database.get(disease_key, {})
                
                # Merge with result
                result['description'] = info.get("description", "Informasi tidak tersedia.")
                result['symptoms'] = info.get("symptoms", [])
                result['causes'] = info.get("causes", [])
                result['treatment'] = info.get("treatment", [])
                result['prevention'] = info.get("prevention", [])
                result['severity'] = info.get("severity", "Tidak diketahui")
                result['urgency'] = info.get("urgency", "Konsultasi dengan ahli")
                result['is_healthy'] = (disease_key == "Sehat")
                
                result['recommended_medicines'] = medicine_info.get('recommended_medicines', [])
                result['organic_alternatives'] = medicine_info.get('organic_alternatives', [])
                result['medicine_notes'] = medicine_info.get('medicine_notes', '')
                result['disease_category'] = medicine_info.get('category', '')
                
                print(f"{'='*60}\n")
                return result
            else:
                print(f"❌ YOLO prediction failed: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ YOLO prediction exception: {e}")
            import traceback
            print(f"📋 Traceback: {traceback.format_exc()}")
    else:
        print("⚠️ YOLO model not available")
    
        print("❌ CNN model or class names not available")
    
    # Fallback ke traditional model
    print(f"\n🔄 FALLING BACK TO TRADITIONAL MODEL...")
    print(f"Traditional Model loaded: {fallback_model is not None}")
    
    if fallback_model is not None and class_names is not None:
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
print("🚀 Starting TomatoDoc AI with YOLO + CNN support...")
load_medicine_database()
load_disease_info_database()

# Try loading YOLO model first (primary)
yolo_available = load_yolo_model()
if yolo_available:
    print("✅ YOLO model loaded as PRIMARY classifier!")
else:
    print("⚠️ YOLO model not available")

# Load CNN as fallback
cnn_available = load_cnn_model()
if cnn_available:
    print("✅ CNN model loaded as FALLBACK classifier!")
else:
    print("⚠️ CNN model not available - using traditional ML fallback")

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
            
            # Cek apakah gambar adalah daun tomat
            if not is_leaf_image(file_path):
                error_msg = "Gambar tidak terdeteksi sebagai daun. Mohon upload foto daun tomat yang jelas."
                print(f"❌ {error_msg}")
                os.remove(file_path)
                return jsonify({'error': error_msg}), 400
            
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
        allowed_exts = ['png', 'jpg', 'jpeg', 'gif']
        error_msg = f'Invalid file type. Allowed types: {", ".join(allowed_exts)}'
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 400

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/evaluation')
def evaluation():
    metrics = {}
    metrics_path = os.path.join('static', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    return render_template('evaluation.html', metrics=metrics)

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
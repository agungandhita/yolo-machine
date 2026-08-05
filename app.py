from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
import pickle
from PIL import Image
from werkzeug.utils import secure_filename
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global state
MODEL = None
CLASS_NAMES = None
DISEASE_DB = None

def load_resources():
    global MODEL, CLASS_NAMES, DISEASE_DB
    print("🚀 Initializing TomatoDoc Application...")
    
    # Load Model
    model_path = "models/mobilenetv2_tomato.h5"
    if os.path.exists(model_path):
        MODEL = keras.models.load_model(model_path)
        print("✅ MobileNetV2 model loaded successfully!")
    else:
        print("⚠️ Model not found! Please run train_cnn.py first.")
        
    # Load Class Names
    classes_path = "models/cnn_class_names.pkl"
    if os.path.exists(classes_path):
        with open(classes_path, "rb") as f:
            CLASS_NAMES = pickle.load(f)
        print(f"✅ Class names loaded ({len(CLASS_NAMES)} classes).")
        
    # Load Disease DB
    db_path = "disease_database.json"
    if os.path.exists(db_path):
        with open(db_path, "r") as f:
            DISEASE_DB = json.load(f)
        print("✅ Disease database loaded.")
    else:
        print("⚠️ Disease database not found! Run merge_data.py first.")

# Load resources on startup
load_resources()

def preprocess_image(image_path):
    """
    100% konsisten dengan preprocessing di train_cnn.py (ImageDataGenerator)
    - Resize 224x224
    - Convert ke RGB
    - Scaling pixel ke 0-1 (dibagi 255.0)
    """
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Gunakan LANCZOS untuk resizing yang lebih baik
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Convert ke numpy array dan scaling
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Expand dims untuk batch axis -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

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
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    return render_template('evaluation.html', metrics=metrics)

@app.route('/upload', methods=['POST'])
def upload_file():
    if MODEL is None or CLASS_NAMES is None or DISEASE_DB is None:
        return jsonify({'error': 'System resources not fully loaded. Please contact administrator.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
        
    try:
        # Create upload directory
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file with secure timestamped name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"{timestamp}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess exactly like training
        img_array = preprocess_image(file_path)
        
        # Predict
        predictions = MODEL.predict(img_array, verbose=0)[0]
        pred_idx = int(np.argmax(predictions))
        confidence = float(predictions[pred_idx])
        predicted_class = CLASS_NAMES[pred_idx]
        
        # Map to disease info
        disease_info = DISEASE_DB.get(predicted_class, {})
        
        # Formatting result
        result = {
            "disease": disease_info.get("nama_penyakit", predicted_class),
            "confidence": confidence,
            "confidence_level": "Tinggi" if confidence > 0.8 else "Sedang" if confidence > 0.6 else "Rendah",
            "causes": disease_info.get("penyebab", "").split(" / "),
            "prevention": disease_info.get("pencegahan", []),
            "medication": disease_info.get("obat", []),
            "recommended_medicines": disease_info.get("obat_details", []), # fallback for frontend compatibility
            "disease_category": "Jamur/Bakteri/Virus", # simplified
            "medicine_notes": disease_info.get("referensi", ""),
            "severity": disease_info.get("tingkat_keparahan", "Tidak diketahui"),
            "image_path": f"uploads/{filename}"
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

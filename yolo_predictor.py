"""
YOLO Predictor Module for Tomato Disease Detection
Integrates trained YOLOv8 classification model with Flask application.
"""

from pathlib import Path
from PIL import Image
import os

# YOLO model and class names
yolo_model = None
yolo_class_names = None

# Mapping from YOLO training class names to exactly match disease_info.json keys
DISEASE_KEY_MAPPING = {
    "Tomato_healthy": "Sehat",
    "Tomato_Bacterial_spot": "Bercak Bakteri",
    "Tomato_Early_blight": "Hawar Daun Awal",
    "Tomato_Late_blight": "Hawar Daun Lanjut", 
    "Tomato_Leaf_Mold": "Jamur Daun",
    "Tomato_Septoria_leaf_spot": "Bercak Daun Septoria",
    "Tomato__Target_Spot": "Bercak Target",
    "Tomato__Tomato_mosaic_virus": "Virus Mozaik Tomat",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Virus Keriting Daun Kuning",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tungau Laba-laba",
}

# Indonesian translations (identity mapping since keys are already translated)
DISEASE_TRANSLATIONS = {v: v for v in DISEASE_KEY_MAPPING.values()}


def load_yolo_model():
    """Load YOLOv8 classification model"""
    global yolo_model, yolo_class_names
    
    model_path = Path(__file__).parent / "models" / "yolo_tomato_disease.pt"
    
    if not model_path.exists():
        print(f"⚠️  YOLO model not found at: {model_path}")
        return False
    
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(str(model_path))
        yolo_class_names = yolo_model.names  # Get class names from model
        print(f"✅ YOLO model loaded successfully!")
        print(f"   Classes: {list(yolo_class_names.values())}")
        return True
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        return False


def predict_disease_yolo(image_path):
    """
    Predict tomato disease using YOLO model.
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict with prediction results or error
    """
    global yolo_model, yolo_class_names
    
    if yolo_model is None:
        if not load_yolo_model():
            return {"error": "YOLO model not loaded"}
    
    try:
        # Run inference
        results = yolo_model(image_path, verbose=False)
        
        if not results or len(results) == 0:
            return {"error": "No predictions from YOLO model"}
        
        result = results[0]
        probs = result.probs
        
        # Get top prediction
        top_idx = probs.top1
        top_conf = probs.top1conf.item()
        
        # Get class name from YOLO model
        yolo_class = yolo_class_names[top_idx]
        
        # Map to disease_info.json key
        disease_key = DISEASE_KEY_MAPPING.get(yolo_class, yolo_class)
        
        # Get Indonesian translation
        disease_name_id = DISEASE_TRANSLATIONS.get(disease_key, disease_key)
        
        # Determine confidence level
        if top_conf > 0.8:
            confidence_level = "Tinggi"
        elif top_conf > 0.6:
            confidence_level = "Sedang"
        else:
            confidence_level = "Rendah"
        
        # Get top 3 predictions
        top3_indices = probs.top5[:3] if len(probs.top5) >= 3 else probs.top5
        top3_predictions = []
        for idx in top3_indices:
            cls_name = yolo_class_names[idx]
            cls_key = DISEASE_KEY_MAPPING.get(cls_name, cls_name)
            cls_name_id = DISEASE_TRANSLATIONS.get(cls_key, cls_key)
            conf = probs.data[idx].item()
            top3_predictions.append((cls_name_id, conf))
        
        # Build all probabilities dict
        all_probabilities = {}
        for idx, prob in enumerate(probs.data):
            cls_name = yolo_class_names[idx]
            cls_key = DISEASE_KEY_MAPPING.get(cls_name, cls_name)
            cls_name_id = DISEASE_TRANSLATIONS.get(cls_key, cls_key)
            all_probabilities[cls_name_id] = float(prob.item())
        
        return {
            "disease": disease_name_id,
            "disease_key": disease_key,
            "confidence": float(top_conf),
            "confidence_level": confidence_level,
            "top_3_diseases": top3_predictions,
            "all_probabilities": all_probabilities,
            "model_type": "YOLO",
            "yolo_class": yolo_class,
        }
        
    except Exception as e:
        import traceback
        return {
            "error": f"YOLO prediction failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


def is_model_loaded():
    """Check if YOLO model is loaded"""
    return yolo_model is not None


if __name__ == "__main__":
    # Test the predictor
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python yolo_predictor.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = predict_disease_yolo(image_path)
    print("\n🎯 Prediction Result:")
    for key, value in result.items():
        print(f"   {key}: {value}")

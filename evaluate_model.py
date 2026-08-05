import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle

DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EVAL_DIR = "evaluation_results"

def evaluate():
    print("📊 Evaluating Model...")
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    # Load model & class names
    model = keras.models.load_model("models/mobilenetv2_tomato.h5")
    with open("models/cnn_class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
        
    # Validation data generator (no augmentation)
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False  # Important for evaluation metrics!
    )
    
    print("⏳ Predicting on validation set...")
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    # 1. Classification Report
    print("📝 Generating Classification Report...")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    with open(os.path.join(EVAL_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
        
    # 2. Confusion Matrix
    print("🗺️ Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"))
    plt.close()
    
    # 3. Metrics Summary
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics_summary = {
        "accuracy": report_dict["accuracy"],
        "macro_avg": report_dict["macro avg"],
        "weighted_avg": report_dict["weighted avg"]
    }
    with open(os.path.join(EVAL_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"✅ Evaluation complete. Results saved in {EVAL_DIR}/")

if __name__ == "__main__":
    evaluate()

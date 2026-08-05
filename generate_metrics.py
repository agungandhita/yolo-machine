import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ultralytics import YOLO

print("Memuat model YOLOv8...")
model = YOLO('models/yolo_tomato_disease.pt')
class_names = model.names
val_dir = 'dataset_yolo/val'

y_true = []
y_pred = []

print("Menjalankan inferensi pada dataset validasi (ini mungkin memakan waktu beberapa menit)...")
total_images = sum([len(files) for r, d, files in os.walk(val_dir) if any(f.endswith(('.jpg', '.jpeg', '.png', '.JPG')) for f in files)])
print(f"Total gambar validasi: {total_images}")

processed = 0
for class_idx, class_name in class_names.items():
    class_dir = os.path.join(val_dir, class_name)
    if not os.path.exists(class_dir):
        continue
    
    for img_file in os.listdir(class_dir):
        if not img_file.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
            continue
        
        img_path = os.path.join(class_dir, img_file)
        results = model(img_path, verbose=False)
        if len(results) > 0:
            pred_idx = results[0].probs.top1
            y_true.append(class_idx)
            y_pred.append(pred_idx)
        
        processed += 1
        if processed % 50 == 0:
            print(f"Progres: {processed}/{total_images}")

# Hitung metrik
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("\n--- HASIL EVALUASI ---")
print(f"Akurasi: {accuracy:.4f}")
print(f"Presisi: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Simpan metrik ke JSON
metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1)
}

with open('static/metrics.json', 'w') as f:
    json.dump(metrics, f)

# Buat Confusion Matrix
print("Membuat grafik Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[class_names[i].replace('Tomato_', '').replace('__', '_') for i in range(len(class_names))],
            yticklabels=[class_names[i].replace('Tomato_', '').replace('__', '_') for i in range(len(class_names))])
plt.ylabel('Kelas Aktual', fontsize=12, fontweight='bold')
plt.xlabel('Kelas Prediksi', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix - Deteksi Penyakit Daun Tomat', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

os.makedirs('static/images', exist_ok=True)
plt.savefig('static/images/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Selesai! Metrik dan gambar confusion matrix telah disimpan ke dalam folder 'static'.")

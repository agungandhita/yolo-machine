

import os
import json
import pickle
import random

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ─────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────
DATASET_DIR  = "dataset"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EVAL_DIR     = "evaluation_results"
VAL_SPLIT    = 0.2          # 20% data untuk validasi
RANDOM_SEED  = 42

# Mapping nama folder dataset → label tampilan (Bahasa Indonesia)
DISPLAY_NAMES = {
    "Tomato_Bacterial_spot":                      "Bercak Bakteri",
    "Tomato_Early_blight":                        "Hawar Daun Awal",
    "Tomato_Late_blight":                         "Hawar Daun Lanjut",
    "Tomato_Leaf_Mold":                           "Jamur Daun",
    "Tomato_Septoria_leaf_spot":                  "Bercak Daun Septoria",
    "Tomato_Spider_mites_Two_spotted_spider_mite":"Tungau Laba-laba",
    "Tomato__Target_Spot":                        "Bercak Target",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":      "Virus Keriting Daun Kuning",
    "Tomato__Tomato_mosaic_virus":                "Virus Mozaik Tomat",
    "Tomato_healthy":                             "Sehat",
}


# ─────────────────────────────────────────────
# Helper: load & preprocess satu gambar
# ─────────────────────────────────────────────
def load_image(path: str) -> np.ndarray:
    """
    Load gambar dari path, resize ke 224x224, normalisasi ke [0,1].
    Konsisten dengan preprocessing di train_cnn.py.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


# ─────────────────────────────────────────────
# Helper: kumpulkan path gambar validation split
# ─────────────────────────────────────────────
def collect_validation_paths(dataset_dir: str, class_names: list[str]) -> tuple[list, list]:
    """
    Untuk setiap kelas, ambil 20% terakhir gambar sebagai validation set
    (deterministik: sorted + seed tetap → hasil selalu sama).
    Mengembalikan (image_paths, labels) di mana label adalah indeks kelas.
    """
    all_paths: list[str] = []
    all_labels: list[int] = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  ⚠️  Folder tidak ditemukan: {class_dir}")
            continue

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted([
            f for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ])

        # Ambil 20% terakhir sebagai validation set (deterministik, sorted order)
        n_val = max(1, int(len(files) * VAL_SPLIT))
        val_files = files[-n_val:]

        for fname in val_files:
            all_paths.append(os.path.join(class_dir, fname))
            all_labels.append(class_idx)

    return all_paths, all_labels


# ─────────────────────────────────────────────
# Helper: prediksi dalam batch
# ─────────────────────────────────────────────
def predict_in_batches(
    model: keras.Model,
    image_paths: list[str],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Jalankan inferensi model pada semua gambar, kembalikan probabilitas."""
    all_probs: list[np.ndarray] = []
    total = len(image_paths)

    for start in range(0, total, batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_imgs  = np.stack([load_image(p) for p in batch_paths])
        probs = model.predict(batch_imgs, verbose=0)
        all_probs.append(probs)

        done = min(start + batch_size, total)
        print(f"  Progres: {done}/{total} gambar", end="\r")

    print()
    return np.vstack(all_probs)


# ─────────────────────────────────────────────
# Fungsi utama evaluasi
# ─────────────────────────────────────────────
def evaluate() -> None:
    print("📊 Mengevaluasi Model MobileNetV2...")
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs("static/images", exist_ok=True)

    # ── 1. Load model & class names ──────────────────────────────
    model_path  = "models/mobilenetv2_tomato.h5"
    class_path  = "models/cnn_class_names.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}. Jalankan train_cnn.py terlebih dahulu.")
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Class names tidak ditemukan: {class_path}.")

    print("  Loading model...")
    model = keras.models.load_model(model_path)

    with open(class_path, "rb") as f:
        class_names: list[str] = pickle.load(f)

    display_labels = [DISPLAY_NAMES.get(c, c) for c in class_names]
    print(f"  ✅ Model dimuat. Jumlah kelas: {len(class_names)}")

    # ── 2. Kumpulkan validation set ──────────────────────────────
    print(f"\n  Mengumpulkan gambar validasi dari '{DATASET_DIR}'...")
    image_paths, y_true = collect_validation_paths(DATASET_DIR, class_names)
    print(f"  ✅ Total gambar validasi: {len(image_paths)}")

    if len(image_paths) == 0:
        raise RuntimeError(f"Tidak ada gambar ditemukan di '{DATASET_DIR}'. Pastikan folder dataset ada.")

    # ── 3. Prediksi ───────────────────────────────────────────────
    print(f"\n  Menjalankan prediksi (batch size={BATCH_SIZE})...")
    probs  = predict_in_batches(model, image_paths)
    y_pred = np.argmax(probs, axis=1)
    y_true_arr = np.array(y_true)

    # ── 4. Hitung metrik keseluruhan ─────────────────────────────
    accuracy  = float(accuracy_score(y_true_arr, y_pred))
    precision = float(precision_score(y_true_arr, y_pred, average="weighted", zero_division=0))
    recall    = float(recall_score(y_true_arr, y_pred, average="weighted", zero_division=0))
    f1        = float(f1_score(y_true_arr, y_pred, average="weighted", zero_division=0))

    print("\n" + "=" * 50)
    print("  HASIL EVALUASI MODEL")
    print("=" * 50)
    print(f"  Akurasi   : {accuracy*100:.2f}%")
    print(f"  Presisi   : {precision*100:.2f}%")
    print(f"  Recall    : {recall*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  Total sampel: {len(y_true_arr)}")
    print("=" * 50)

    # ── 5. Metrik per kelas ───────────────────────────────────────
    report_dict = classification_report(
        y_true_arr, y_pred,
        target_names=display_labels,
        output_dict=True,
        zero_division=0,
    )
    per_class_metrics = []
    for label in display_labels:
        if label in report_dict:
            per_class_metrics.append({
                "class":     label,
                "precision": round(report_dict[label]["precision"], 4),
                "recall":    round(report_dict[label]["recall"],    4),
                "f1_score":  round(report_dict[label]["f1-score"],  4),
                "support":   int(report_dict[label]["support"]),
            })

    # ── 6. Simpan metrics_summary.json (flat keys) ────────────────
    metrics_summary = {
        "accuracy":      accuracy,
        "precision":     precision,
        "recall":        recall,
        "f1_score":      f1,
        "per_class":     per_class_metrics,
        "total_samples": len(y_true_arr),
        "num_classes":   len(class_names),
    }
    summary_path = os.path.join(EVAL_DIR, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4, ensure_ascii=False)
    print(f"\n  ✅ metrics_summary.json disimpan ke: {summary_path}")

    # ── 7. Classification report (teks) ──────────────────────────
    report_text = classification_report(
        y_true_arr, y_pred,
        target_names=display_labels,
        zero_division=0,
    )
    print("\n  Classification Report:\n")
    print(report_text)
    with open(os.path.join(EVAL_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    # ── 8. Confusion Matrix ───────────────────────────────────────
    print("  🗺️  Membuat Confusion Matrix...")
    cm = confusion_matrix(y_true_arr, y_pred)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
    )
    ax.set_title(
        "Confusion Matrix — Deteksi Penyakit Daun Tomat",
        fontsize=15, fontweight="bold", pad=15,
    )
    ax.set_ylabel("Label Aktual (Ground Truth)", fontsize=12)
    ax.set_xlabel("Label Prediksi (Model)",       fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()

    for dest in [
        os.path.join(EVAL_DIR, "confusion_matrix.png"),
        "static/images/confusion_matrix.png",
    ]:
        plt.savefig(dest, dpi=150, bbox_inches="tight")
        print(f"  ✅ Confusion matrix disimpan ke: {dest}")

    plt.close()

    print("\n✅ Evaluasi selesai!")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    evaluate()

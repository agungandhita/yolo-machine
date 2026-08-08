"""
train_cnn.py — MobileNetV2 Transfer Learning untuk Deteksi Penyakit Daun Tomat
==============================================================================
Perbaikan dari versi sebelumnya:
  - EPOCHS: 1 → 20 (dengan EarlyStopping patience=5)
  - steps_per_epoch: 20 (hardcoded) → None (full dataset)
  - validation_steps: 10 (hardcoded) → None (full validation set)
  - Fine-tuning Fase 2: unfreeze 50 layer terakhir MobileNetV2
  - Augmentasi lebih kuat untuk kelas minoritas
  - Class weights untuk menangani imbalance (Virus Mozaik hanya 74 gambar)
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight

# ──────────────────────────────────────────────────────────────────────────────
# Konfigurasi
# ──────────────────────────────────────────────────────────────────────────────
DATASET_DIR     = "dataset"
BATCH_SIZE      = 32
IMG_SIZE        = (224, 224)

# Fase 1: Feature extraction (base frozen)
EPOCHS_PHASE1   = 10

# Fase 2: Fine-tuning (unfreeze layer akhir)
EPOCHS_PHASE2   = 10

# Jumlah layer MobileNetV2 yang di-unfreeze di fase 2
UNFREEZE_LAYERS = 50

MODEL_SAVE_PATH = "models/mobilenetv2_tomato.h5"
CLASS_PATH      = "models/cnn_class_names.pkl"


def create_model(num_classes: int) -> Model:
    """Buat model MobileNetV2 + custom head."""
    print("🔄 Membangun model MobileNetV2...")
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False  # Fase 1: freeze semua

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"   Total params: {model.count_params():,}")
    return model


def get_callbacks(phase: int) -> list:
    """Kembalikan daftar callbacks sesuai fase training."""
    return [
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def main():
    print("=" * 65)
    print("🚀 TomatoDoc — CNN Training Pipeline (MobileNetV2)")
    print("=" * 65)

    os.makedirs("models", exist_ok=True)

    # ── Data generators ────────────────────────────────────────────
    # Augmentasi cukup kuat untuk generalisasi pola visual penyakit
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
        validation_split=0.2,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    print("\n📂 Memuat dataset dari:", DATASET_DIR)
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)
    print(f"✅ Kelas terdeteksi ({num_classes}): {class_names}")

    # Simpan class names
    with open(CLASS_PATH, "wb") as f:
        pickle.dump(class_names, f)
    print(f"✅ Class names disimpan ke: {CLASS_PATH}")

    # ── Class weights untuk menangani imbalance ───────────────────
    print("\n⚖️  Menghitung class weights untuk imbalance handling...")
    classes_arr = train_generator.classes
    weights_arr = compute_class_weight(
        "balanced",
        classes=np.unique(classes_arr),
        y=classes_arr,
    )
    class_weights = dict(enumerate(weights_arr))
    print("   Class weights:")
    for idx, name in enumerate(class_names):
        print(f"     [{idx:2d}] {name:<50s} → {weights_arr[idx]:.3f}")

    # ── Fase 1: Feature Extraction ─────────────────────────────────
    print(f"\n{'='*65}")
    print(f"📚 FASE 1: Feature Extraction ({EPOCHS_PHASE1} epochs maks)")
    print(f"   Base model FROZEN, hanya melatih custom head")
    print(f"{'='*65}")

    model = create_model(num_classes)

    history1 = model.fit(
        train_generator,
        epochs=EPOCHS_PHASE1,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=get_callbacks(phase=1),
        verbose=1,
    )

    best_val_acc_1 = max(history1.history["val_accuracy"])
    print(f"\n✅ Fase 1 selesai. Best val_accuracy: {best_val_acc_1*100:.2f}%")

    # ── Fase 2: Fine-tuning ────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"🔧 FASE 2: Fine-tuning ({EPOCHS_PHASE2} epochs maks)")
    print(f"   Unfreeze {UNFREEZE_LAYERS} layer terakhir MobileNetV2")
    print(f"{'='*65}")

    # Load best model dari fase 1
    model = keras.models.load_model(MODEL_SAVE_PATH)

    # Unfreeze layer terakhir
    base_model = model.layers[0]  # MobileNetV2 adalah layer pertama
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"   Layer trainable: {trainable_count}/{len(model.layers)}")

    # Compile ulang dengan LR jauh lebih kecil untuk fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_generator,
        epochs=EPOCHS_PHASE2,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=get_callbacks(phase=2),
        verbose=1,
    )

    best_val_acc_2 = max(history2.history["val_accuracy"])
    print(f"\n✅ Fase 2 selesai. Best val_accuracy: {best_val_acc_2*100:.2f}%")

    # ── Simpan metadata ────────────────────────────────────────────
    all_val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    all_train_acc = history1.history["accuracy"] + history2.history["accuracy"]

    metadata = {
        "model_type": "MobileNetV2 Transfer Learning (2-Phase Fine-tuning)",
        "input_shape": [224, 224, 3],
        "classes": class_names,
        "num_classes": num_classes,
        "training_config": {
            "epochs_phase1": len(history1.history["val_accuracy"]),
            "epochs_phase2": len(history2.history["val_accuracy"]),
            "batch_size": BATCH_SIZE,
            "unfreeze_layers": UNFREEZE_LAYERS,
            "augmentation": "rotation30/shift0.2/shear0.15/zoom0.2/brightness0.7-1.3",
        },
        "metrics": {
            "best_val_accuracy": float(max(all_val_acc)),
            "final_train_accuracy": float(all_train_acc[-1]),
        },
    }
    import json
    with open("models/cnn_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*65}")
    print(f"🎉 Training selesai!")
    print(f"   Model terbaik disimpan ke: {MODEL_SAVE_PATH}")
    print(f"   Best val_accuracy overall: {max(all_val_acc)*100:.2f}%")
    print(f"   Jalankan: python evaluate_model.py")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

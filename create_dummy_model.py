#!/usr/bin/env python3

import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 1337
EPOCHS = int(os.getenv("EPOCHS", "3"))


def build_datasets():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
    )

    raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
    )

    class_names = list(raw_train_ds.class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = raw_train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = raw_val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def build_model(num_classes: int):
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))

    x = keras.layers.Rescaling(1.0 / 255)(inputs)
    x = keras.layers.RandomFlip("horizontal")(x)
    x = keras.layers.RandomRotation(0.1)(x)
    x = keras.layers.RandomZoom(0.1)(x)

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base.trainable = False

    x = base(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_save():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, class_names = build_datasets()
    num_classes = len(class_names)

    model = build_model(num_classes)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "tomato_cnn_model.h5"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    class_names_path = MODELS_DIR / "cnn_class_names.pkl"
    metadata_path = MODELS_DIR / "cnn_metadata.pkl"

    with open(class_names_path, "wb") as f:
        pickle.dump(class_names, f)

    metadata = {
        "model_type": "CNN (Transfer Learning, EfficientNetB0)",
        "input_shape": (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        "class_count": num_classes,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved model to: {MODELS_DIR / 'tomato_cnn_model.h5'}")
    print(f"Saved class names to: {class_names_path}")
    print(f"Saved metadata to: {metadata_path}")


def main():
    train_and_save()


if __name__ == "__main__":
    main()
import pickle
import numpy as np
import os

# Kelas penyakit tomat
class_names = [
    'Healthy',
    'Bacterial_spot',
    'Early_blight', 
    'Late_blight',
    'Leaf_Mold',
    'Septoria_leaf_spot',
    'Spider_mites',
    'Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus'
]

print("🚀 Creating dummy model for tomato disease detection...")

# Create a simple model structure using basic Python types
model_data = {
    'type': 'dummy_classifier',
    'n_features': 49,
    'n_classes': len(class_names),
    'class_names': class_names,
    'weights': np.random.rand(49, len(class_names)).tolist(),  # Convert to list for pickle
    'bias': np.random.rand(len(class_names)).tolist()
}

# Create simple scaler data
scaler_data = {
    'type': 'dummy_scaler',
    'mean': np.random.rand(49).tolist(),
    'scale': (np.random.rand(49) + 0.1).tolist()  # Avoid zero scale
}

# Save model
os.makedirs('models', exist_ok=True)

with open('models/tomato_dummy_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

with open('models/dummy_class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)

with open('models/dummy_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_data, f)

print("✅ Dummy model saved to models/tomato_dummy_model.pkl")
print("✅ Class names saved to models/dummy_class_names.pkl") 
print("✅ Scaler saved to models/dummy_scaler.pkl")
print(f"✅ Model features: {model_data['n_features']}")
print(f"✅ Model classes: {model_data['n_classes']}")
print("🎉 Dummy model creation completed!")
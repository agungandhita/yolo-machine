import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
import json
from pathlib import Path

# Config
DATASET_DIR = "dataset"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 1  # Reduced to 1 for quick proof of concept

def create_model(num_classes):
    print("🔄 Creating MobileNetV2 Transfer Learning Model...")
    # Base model from MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("🚀 Starting CNN Training Pipeline...")
    
    # Data Augmentation & Loading
    # Use validation_split to handle imbalanced sets slightly better and avoid writing manual split code
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    class_names = list(train_generator.class_indices.keys())
    print(f"📊 Classes detected: {len(class_names)}")
    
    # Save class names
    os.makedirs("models", exist_ok=True)
    with open("models/cnn_class_names.pkl", "wb") as f:
        pickle.dump(class_names, f)
        
    # Calculate class weights to solve the mode collapse
    print("⚖️ Calculating class weights...")
    classes = train_generator.classes
    class_weights_arr = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
    class_weights = dict(enumerate(class_weights_arr))
    print(f"Class weights: {class_weights}")
    
    # Create Model
    model = create_model(len(class_names))
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "models/mobilenetv2_tomato.h5", 
            save_best_only=True,
            monitor='val_accuracy'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]
    
    # Train
    print("⏳ Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=20, # Reduced for rapid proof of concept
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=10, # Reduced for rapid proof of concept
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    print("✅ Training complete! Model saved to models/mobilenetv2_tomato.h5")
    
    # Save metadata
    metadata = {
        "model_type": "MobileNetV2 Transfer Learning",
        "input_shape": (224, 224, 3),
        "classes": class_names,
        "metrics": {
            "val_accuracy": float(history.history['val_accuracy'][-1]),
            "val_loss": float(history.history['val_loss'][-1])
        }
    }
    with open("models/cnn_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

if __name__ == "__main__":
    main()

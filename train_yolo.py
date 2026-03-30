#!/usr/bin/env python3
"""
YOLO Tomato Disease Classification Training Script
Trains a YOLOv8 classification model on the tomato disease dataset.
"""

import os
import shutil
import random
from pathlib import Path

# Dataset configuration
DATASET_DIR = Path("dataset")
YOLO_DATASET_DIR = Path("dataset_yolo")
TRAIN_RATIO = 0.8

# Disease classes (matching folder names)
DISEASE_CLASSES = [
    "Tomato_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
]


def prepare_yolo_dataset():
    """
    Prepare dataset in YOLO classification format.
    Structure: dataset_yolo/train/<class>/*.jpg and dataset_yolo/val/<class>/*.jpg
    """
    print("🔄 Preparing YOLO classification dataset...")
    
    # Create YOLO dataset directories
    train_dir = YOLO_DATASET_DIR / "train"
    val_dir = YOLO_DATASET_DIR / "val"
    
    # Clean and create directories
    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)
    
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    
    total_train = 0
    total_val = 0
    
    for class_name in DISEASE_CLASSES:
        source_dir = DATASET_DIR / class_name
        
        if not source_dir.exists():
            print(f"⚠️  Class folder not found: {source_dir}")
            continue
        
        # Get all image files
        images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.JPG")) + \
                 list(source_dir.glob("*.jpeg")) + list(source_dir.glob("*.png"))
        
        if not images:
            print(f"⚠️  No images found in: {source_dir}")
            continue
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create class directories
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        
        # Copy images (symlink for speed, copy if symlink fails)
        for img in train_images:
            dst = train_dir / class_name / img.name
            try:
                dst.symlink_to(img.resolve())
            except:
                shutil.copy2(img, dst)
        
        for img in val_images:
            dst = val_dir / class_name / img.name
            try:
                dst.symlink_to(img.resolve())
            except:
                shutil.copy2(img, dst)
        
        total_train += len(train_images)
        total_val += len(val_images)
        
        print(f"✅ {class_name}: {len(train_images)} train, {len(val_images)} val")
    
    print(f"\n📊 Dataset prepared: {total_train} training, {total_val} validation images")
    return str(YOLO_DATASET_DIR)


def train_yolo():
    """Train YOLOv8 classification model"""
    from ultralytics import YOLO
    
    # Prepare dataset
    dataset_path = prepare_yolo_dataset()
    
    print("\n🚀 Starting YOLOv8 classification training...")
    print("⚡ Using nano model optimized for CPU training...")
    
    # Load YOLOv8 classification model (nano version for faster CPU training)
    model = YOLO("yolov8n-cls.pt")
    
    # Train the model with CPU-optimized settings
    results = model.train(
        data=dataset_path,
        epochs=10,           # Reduced for faster training
        imgsz=224,
        batch=16,            # Smaller batch for CPU
        patience=3,
        project="runs/classify",
        name="tomato_disease",
        exist_ok=True,
        verbose=True,
        workers=0,           # Avoid multiprocessing issues on CPU
        amp=False,           # Disable mixed precision on CPU
        fraction=0.5,        # Use 50% of data for faster training
    )
    
    print("\n✅ Training complete!")
    print(f"📁 Model saved to: runs/classify/tomato_disease/weights/best.pt")
    
    # Copy best model to models directory
    best_model_src = Path("runs/classify/tomato_disease/weights/best.pt")
    best_model_dst = Path("models/yolo_tomato_disease.pt")
    
    if best_model_src.exists():
        shutil.copy2(best_model_src, best_model_dst)
        print(f"📁 Model copied to: {best_model_dst}")
    
    return results


def evaluate_model():
    """Evaluate the trained model"""
    from ultralytics import YOLO
    
    model_path = Path("models/yolo_tomato_disease.pt")
    
    if not model_path.exists():
        print("❌ Model not found! Train the model first.")
        return
    
    model = YOLO(str(model_path))
    
    # Validate on validation set
    val_path = YOLO_DATASET_DIR / "val"
    if val_path.exists():
        results = model.val(data=str(YOLO_DATASET_DIR))
        print(f"\n📊 Validation Results:")
        print(f"   Top-1 Accuracy: {results.top1:.2%}")
        print(f"   Top-5 Accuracy: {results.top5:.2%}")
        return results
    else:
        print("⚠️  Validation dataset not found. Run training first.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Tomato Disease Training")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare dataset")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model")
    args = parser.parse_args()
    
    if args.prepare_only:
        prepare_yolo_dataset()
    elif args.evaluate:
        evaluate_model()
    else:
        train_yolo()

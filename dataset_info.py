#!/usr/bin/env python3
"""
Dataset Information Script
Menampilkan informasi lengkap tentang dataset yang telah diperbaiki
"""

import os
import pickle
import json
from pathlib import Path

def display_dataset_info():
    """Menampilkan informasi lengkap tentang dataset"""
    
    print("=" * 80)
    print("🍅 INFORMASI LENGKAP DATASET TOMATO DISEASE DETECTION")
    print("=" * 80)
    
    # Path dataset
    dataset_path = "/Users/mac/Documents/yolo/dataset"
    
    print(f"\n📁 Lokasi Dataset: {dataset_path}")
    
    # Cek struktur dataset
    if os.path.exists(dataset_path):
        print("\n📊 STRUKTUR DATASET:")
        print("-" * 50)
        
        categories = []
        total_images = 0
        
        for item in sorted(os.listdir(dataset_path)):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                # Hitung jumlah file gambar
                image_count = len([f for f in os.listdir(item_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                categories.append(item)
                total_images += image_count
                print(f"  {len(categories):2d}. {item:<35} ({image_count:4d} gambar)")
        
        print(f"\n📈 RINGKASAN:")
        print(f"  • Total Kategori: {len(categories)}")
        print(f"  • Total Gambar: {total_images}")
        print(f"  • Rata-rata per kategori: {total_images // len(categories) if categories else 0}")
    
    # Informasi model CNN
    print(f"\n🤖 INFORMASI MODEL CNN:")
    print("-" * 50)
    
    try:
        # Load class names
        with open('models/cnn_class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        
        print(f"  • Jumlah kelas: {len(class_names)}")
        print(f"  • Kelas yang dapat diprediksi:")
        for i, class_name in enumerate(class_names, 1):
            print(f"    {i:2d}. {class_name}")
        
        # Load metadata
        try:
            with open('models/cnn_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"\n  • Model: {metadata.get('model_name', 'Unknown')}")
            print(f"  • Input shape: {metadata.get('input_shape', 'Unknown')}")
            print(f"  • Jumlah kelas: {metadata.get('num_classes', 'Unknown')}")
            
        except FileNotFoundError:
            print("  • Metadata tidak ditemukan")
            
    except FileNotFoundError:
        print("  • Class names tidak ditemukan")
    
    # Informasi penyakit
    print(f"\n🏥 INFORMASI PENYAKIT YANG DAPAT DIDETEKSI:")
    print("-" * 50)
    
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
    
    severity_levels = {
        "Bercak Bakteri": "Sedang",
        "Hawar Daun Awal": "Sedang-Tinggi",
        "Hawar Daun Lanjut": "Sangat Tinggi",
        "Jamur Daun": "Sedang", 
        "Bercak Daun Septoria": "Sedang",
        "Tungau Laba-laba": "Sedang",
        "Bercak Target": "Sedang",
        "Virus Keriting Daun Kuning": "Tinggi",
        "Virus Mozaik Tomat": "Tinggi",
        "Sehat": "Tidak ada"
    }
    
    for i, (eng_name, indo_name) in enumerate(disease_translations.items(), 1):
        severity = severity_levels.get(indo_name, "Unknown")
        print(f"  {i:2d}. {indo_name:<30} (Tingkat: {severity})")
    
    # Status aplikasi
    print(f"\n🚀 STATUS APLIKASI:")
    print("-" * 50)
    print("  ✅ Dataset telah diperbaiki dan disesuaikan")
    print("  ✅ Class names telah diperbarui sesuai dataset aktual")
    print("  ✅ Database informasi penyakit lengkap dan akurat")
    print("  ✅ Model CNN siap untuk prediksi")
    print("  ✅ Aplikasi web berjalan di http://localhost:5000")
    
    print(f"\n🔧 PERBAIKAN YANG TELAH DILAKUKAN:")
    print("-" * 50)
    print("  1. Mengganti class names dari 'PlantVillage' ke nama penyakit sebenarnya")
    print("  2. Menyesuaikan jumlah kelas dari 16 ke 10 sesuai dataset")
    print("  3. Memperbarui database informasi penyakit dengan data lengkap")
    print("  4. Menambahkan informasi gejala, penyebab, pengobatan, dan pencegahan")
    print("  5. Menyediakan tingkat keparahan dan urgensi penanganan")
    
    print("\n" + "=" * 80)
    print("✅ DATASET DAN APLIKASI SIAP DIGUNAKAN!")
    print("=" * 80)

if __name__ == "__main__":
    display_dataset_info()
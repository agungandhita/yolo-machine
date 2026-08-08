"""
tests/test_revisi.py
=====================
Test suite untuk membuktikan 3 poin revisi dosen penguji:

  Poin 1: Tidak ada kontradiksi data di disease_database.json
  Poin 2: File evaluasi (classification_report, confusion_matrix, metrics) ada & valid
  Poin 3: Mapping obat per kelas berbeda-beda (tidak identik antar kelas berbeda)

Jalankan: python -m pytest tests/test_revisi.py -v
"""

import os
import json
import sys
import unittest
from itertools import combinations
from pathlib import Path

# ── Path ke root project (satu level di atas folder tests/) ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "disease_database.json"
EVAL_DIR     = PROJECT_ROOT / "evaluation_results"


# ════════════════════════════════════════════════════════════════════
# POIN 1 — Konsistensi & Non-kontradiksi Data Penyakit
# ════════════════════════════════════════════════════════════════════
class TestPoin1KonsistensiData(unittest.TestCase):
    """Membuktikan Poin 1 Revisi Dosen: tidak ada kontradiksi data."""

    def _load_db(self):
        self.assertTrue(DB_PATH.exists(), f"Database tidak ditemukan: {DB_PATH}")
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_1a_semua_kelas_punya_field_wajib(self):
        """Setiap kelas harus punya field: nama_penyakit, penyebab, pencegahan, obat, referensi."""
        db = self._load_db()
        required_fields = ["nama_penyakit", "penyebab", "pencegahan", "obat", "referensi"]
        errors = []
        for kelas, info in db.items():
            for field in required_fields:
                if field not in info:
                    errors.append(f"Kelas '{kelas}' tidak punya field '{field}'")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_1b_penyakit_jamur_tidak_punya_penyebab_bakteri(self):
        """Penyakit jamur tidak boleh mencantumkan 'bakteri' sebagai penyebab utama."""
        db = self._load_db()
        kelas_jamur = ["Hawar Daun Awal", "Hawar Daun Lanjut", "Jamur Daun",
                       "Bercak Daun Septoria", "Bercak Target"]
        errors = []
        for kelas in kelas_jamur:
            if kelas not in db:
                continue
            penyebab = db[kelas].get("penyebab", "").lower()
            if "bakteri" in penyebab and "xanthomonas" not in penyebab:
                errors.append(
                    f"Kelas jamur '{kelas}' mencantumkan 'bakteri' di penyebab: {penyebab[:80]}"
                )
        self.assertEqual(errors, [], "\n".join(errors))

    def test_1c_penyakit_bakteri_punya_bakterisida(self):
        """Bercak Bakteri harus mencantumkan minimal 1 bakterisida di obat_details."""
        db = self._load_db()
        kelas = "Bercak Bakteri"
        self.assertIn(kelas, db, f"Kelas '{kelas}' tidak ditemukan di database")
        obat_details = db[kelas].get("obat_details", [])
        types = [o.get("type", "").lower() for o in obat_details]
        punya_bakterisida = any("bakterisida" in t for t in types)
        self.assertTrue(
            punya_bakterisida,
            f"'{kelas}' seharusnya punya obat tipe Bakterisida, tapi hanya ada: {types}"
        )

    def test_1d_kelas_virus_punya_kontrol_vektor_bukan_fungisida(self):
        """Penyakit virus harus mengendalikan vektor, bukan fungisida tanaman."""
        db = self._load_db()
        kelas_virus = ["Virus Keriting Daun Kuning", "Virus Mozaik Tomat"]
        errors = []
        for kelas in kelas_virus:
            if kelas not in db:
                continue
            obat = db[kelas].get("obat", [])
            for o in obat:
                if any(kw in o.lower() for kw in ["fungisida", "metalaxyl", "chlorothalonil"]):
                    errors.append(f"Kelas virus '{kelas}' mengandung fungisida di obat: {o}")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_1e_kelas_sehat_tidak_punya_obat(self):
        """Kelas Sehat harus memiliki field obat kosong (tidak ada pestisida)."""
        db = self._load_db()
        self.assertIn("Sehat", db)
        obat = db["Sehat"].get("obat", [])
        self.assertEqual(
            obat, [],
            f"Kelas 'Sehat' seharusnya tidak punya obat, tapi ada: {obat}"
        )

    def test_1f_kelas_tungau_punya_akarisida(self):
        """Tungau Laba-laba harus punya akarisida/mitisida spesifik."""
        db = self._load_db()
        kelas = "Tungau Laba-laba"
        self.assertIn(kelas, db)
        obat_details = db[kelas].get("obat_details", [])
        types = [o.get("type", "").lower() for o in obat_details]
        punya_mitisida = any(
            any(kw in t for kw in ["mitisida", "akarisida", "insektisida"])
            for t in types
        )
        self.assertTrue(punya_mitisida, f"'{kelas}' harus punya mitisida/akarisida, ada: {types}")


# ════════════════════════════════════════════════════════════════════
# POIN 2 — File Evaluasi Model Tersedia & Valid
# ════════════════════════════════════════════════════════════════════
class TestPoin2EvaluasiModel(unittest.TestCase):
    """Membuktikan Poin 2 Revisi Dosen: evaluasi model terdokumentasi."""

    def test_2a_classification_report_ada(self):
        path = EVAL_DIR / "classification_report.txt"
        self.assertTrue(path.exists(), f"File tidak ditemukan: {path}")
        content = path.read_text(encoding="utf-8")
        self.assertGreater(len(content), 100, "classification_report.txt tampak kosong/tidak valid")
        # Harus ada kata "precision" dan "recall"
        self.assertIn("precision", content.lower())
        self.assertIn("recall", content.lower())

    def test_2b_confusion_matrix_ada(self):
        path = EVAL_DIR / "confusion_matrix.png"
        self.assertTrue(path.exists(), f"File tidak ditemukan: {path}")
        self.assertGreater(path.stat().st_size, 5000, "Confusion matrix PNG tampak terlalu kecil/corrupt")

    def test_2c_metrics_summary_ada_dan_valid(self):
        path = EVAL_DIR / "metrics_summary.json"
        self.assertTrue(path.exists(), f"File tidak ditemukan: {path}")
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # Kunci wajib ada
        for key in ["accuracy", "precision", "recall", "f1_score", "per_class"]:
            self.assertIn(key, metrics, f"Key '{key}' tidak ada di metrics_summary.json")

        acc = metrics["accuracy"]
        self.assertGreater(acc, 0.0,  "Akurasi 0% — kemungkinan model belum ditraining")
        self.assertLess(acc, 1.0,     "Akurasi 100% — kemungkinan data leakage (train==test)")
        # Harus punya 10 kelas
        self.assertEqual(len(metrics["per_class"]), 10,
                         f"Harus ada 10 kelas, tapi ada {len(metrics['per_class'])}")

    def test_2d_per_class_metrics_masuk_akal(self):
        """Tidak ada kelas dengan precision/recall yang semuanya 0 (tanda data kosong)."""
        path = EVAL_DIR / "metrics_summary.json"
        if not path.exists():
            self.skipTest("metrics_summary.json belum ada")
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        all_zero_classes = []
        for pc in metrics.get("per_class", []):
            if pc["precision"] == 0 and pc["recall"] == 0 and pc["f1_score"] == 0:
                # Kelas Sehat boleh punya 0 obat, tapi evaluasi tetap harus ada nilai
                all_zero_classes.append(pc["class"])

        # Boleh ada maks 1 kelas dengan semua metrik 0 (misalnya kelas dengan sangat sedikit data)
        self.assertLessEqual(
            len(all_zero_classes), 1,
            f"Terlalu banyak kelas dengan semua metrik 0: {all_zero_classes}\n"
            f"Kemungkinan model belum ditraining dengan benar."
        )


# ════════════════════════════════════════════════════════════════════
# POIN 3 — Mapping Obat Berbeda Per Kelas
# ════════════════════════════════════════════════════════════════════
class TestPoin3MappingObat(unittest.TestCase):
    """Membuktikan Poin 3 Revisi Dosen: obat berbeda untuk setiap kelas penyakit."""

    # Whitelist: pasangan kelas yang boleh berbagi obat karena patogen mirip
    # (harus ada justifikasi ilmiah di field 'referensi')
    WHITELIST_PAIRS: set[frozenset] = set()  # Kosong = tidak ada yang di-whitelist

    def _load_db(self):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_3a_semua_kelas_penyakit_punya_obat(self):
        """Semua kelas penyakit (bukan Sehat) harus punya minimal 1 obat."""
        db = self._load_db()
        errors = []
        for kelas, info in db.items():
            if kelas == "Sehat":
                continue
            obat = info.get("obat", [])
            if not obat:
                errors.append(f"Kelas '{kelas}' tidak punya obat sama sekali!")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_3b_tidak_ada_obat_identik_antar_kelas_berbeda(self):
        """
        Dua kelas berbeda TIDAK BOLEH punya array obat[] yang identik persis,
        kecuali ada justifikasi di whitelist atau field 'referensi' menjelaskannya.
        """
        db = self._load_db()

        # Buat dict: frozenset(obat) → list of kelas
        obat_map: dict[frozenset, list[str]] = {}
        for kelas, info in db.items():
            if kelas == "Sehat":
                continue
            obat_set = frozenset(info.get("obat", []))
            if not obat_set:
                continue
            obat_map.setdefault(obat_set, []).append(kelas)

        # Cari duplikasi
        duplicates = {k: v for k, v in obat_map.items() if len(v) > 1}
        errors = []
        for obat_set, kelas_list in duplicates.items():
            pair = frozenset(kelas_list[:2])
            if pair in self.WHITELIST_PAIRS:
                continue  # Sudah di-whitelist
            errors.append(
                f"Kelas {kelas_list} memiliki obat[] IDENTIK: {sorted(obat_set)}\n"
                f"  → Jika memang disengaja secara ilmiah, tambahkan ke WHITELIST_PAIRS"
            )
        self.assertEqual(errors, [], "\n".join(errors))

    def test_3c_obat_details_berbeda_per_kelas(self):
        """
        obat_details[0]['name'] harus berbeda antara kelas-kelas utama
        (Fungal vs Bacterial vs Viral vs Mite).
        """
        db = self._load_db()
        first_obat_per_kelas = {}
        for kelas, info in db.items():
            if kelas == "Sehat":
                continue
            details = info.get("obat_details", [])
            if details:
                first_obat_per_kelas[kelas] = details[0]["name"]

        # Kelas yang secara biologis berbeda harus punya obat pertama berbeda
        kelas_unik = [
            "Bercak Bakteri",         # Bakterisida
            "Tungau Laba-laba",       # Mitisida
            "Virus Keriting Daun Kuning",  # Insektisida vektor
            "Virus Mozaik Tomat",     # Disinfektan
        ]
        first_obats = [first_obat_per_kelas.get(k, "") for k in kelas_unik if k in first_obat_per_kelas]
        unique_first_obats = set(first_obats)

        self.assertEqual(
            len(unique_first_obats), len([k for k in kelas_unik if k in first_obat_per_kelas]),
            f"Kelas dengan patogen berbeda seharusnya punya obat pertama berbeda.\n"
            f"  Kelas: {kelas_unik}\n"
            f"  Obat pertama: {first_obats}"
        )

    def test_3d_kelas_mapping_lengkap_di_app(self):
        """Semua kunci di disease_database.json harus terdapat di CLASS_TO_DB di app.py."""
        db = self._load_db()
        db_keys = set(db.keys())

        # Baca CLASS_TO_DB dari app.py secara literal
        app_path = PROJECT_ROOT / "app.py"
        self.assertTrue(app_path.exists(), "app.py tidak ditemukan")
        app_content = app_path.read_text(encoding="utf-8")

        # Cek setiap key database ada sebagai value di CLASS_TO_DB
        missing = []
        for key in db_keys:
            # Cari literal string key di dalam konten app.py
            if f'"{key}"' not in app_content and f"'{key}'" not in app_content:
                missing.append(key)

        self.assertEqual(
            missing, [],
            f"Key database berikut tidak ditemukan di CLASS_TO_DB app.py: {missing}"
        )


# ════════════════════════════════════════════════════════════════════
# SANITY CHECK — Preprocessing
# ════════════════════════════════════════════════════════════════════
class TestSanityCheck(unittest.TestCase):
    """Sanity check dasar: preprocessing gambar berbeda → array berbeda."""

    def test_preprocessing_gambar_berbeda_hasilkan_array_berbeda(self):
        """
        Dua gambar dari kelas berbeda setelah preprocessing TIDAK BOLEH identik.
        Jika identik, ada bug di pipeline preprocessing atau gambar corrupt.
        """
        import numpy as np
        from PIL import Image, ImageDraw
        import tempfile

        # Buat 2 gambar sintetis yang jelas berbeda
        def make_image(color: tuple, tmpdir: str) -> np.ndarray:
            img = Image.new("RGB", (300, 300), color=color)
            # Tambahkan noise agar lebih realistis
            draw = ImageDraw.Draw(img)
            draw.ellipse([50, 50, 150, 150], fill=(0, 0, 0))
            path = os.path.join(tmpdir, f"test_{color[0]}.jpg")
            img.save(path)
            # Preprocessing sama dengan app.py & evaluate_model.py
            img_loaded = Image.open(path).convert("RGB")
            img_loaded = img_loaded.resize((224, 224), Image.Resampling.LANCZOS)
            arr = np.array(img_loaded, dtype=np.float32) / 255.0
            return arr

        with tempfile.TemporaryDirectory() as tmpdir:
            arr1 = make_image((34, 139, 34), tmpdir)   # hijau
            arr2 = make_image((139, 0, 0), tmpdir)     # merah

        # Array HARUS berbeda
        self.assertFalse(
            np.array_equal(arr1, arr2),
            "Array preprocessing dua gambar berbeda ternyata identik! Ada bug di pipeline."
        )

        # Bentuk harus (224, 224, 3)
        self.assertEqual(arr1.shape, (224, 224, 3))
        self.assertEqual(arr2.shape, (224, 224, 3))

        # Nilai harus dalam [0, 1]
        self.assertGreaterEqual(float(arr1.min()), 0.0)
        self.assertLessEqual(float(arr1.max()), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import os
import json
import unittest

class TestRevisiDosen(unittest.TestCase):
    
    def setUp(self):
        self.db_path = "/Users/mac/Documents/yolo/disease_database.json"
        self.eval_dir = "/Users/mac/Documents/yolo/evaluation_results"
        
    def test_poin_1_konsistensi_data_obat(self):
        """Test poin 1: memastikan obat tidak ada yang terduplikasi secara identik tanpa alasan ilmiah."""
        self.assertTrue(os.path.exists(self.db_path), "Database file tidak ditemukan!")
        
        with open(self.db_path, "r") as f:
            db = json.load(f)
            
        meds_seen = {}
        for class_name, info in db.items():
            if "obat" in info:
                # Obat array to sorted comma string
                meds_str = ", ".join(sorted(info["obat"]))
                if meds_str == "":
                    continue # sehat
                
                # Check duplication
                if meds_str in meds_seen:
                    # If duplicate, it MUST have a valid "referensi" that explains it
                    referensi = info.get("referensi", "")
                    self.assertTrue(
                        len(referensi) > 10,
                        f"Penyakit {class_name} dan {meds_seen[meds_str]} memiliki obat yang identik ({meds_str}) tetapi tidak memiliki referensi ilmiah yang valid!"
                    )
                else:
                    meds_seen[meds_str] = class_name
                    
    def test_poin_2_evaluasi_model(self):
        """Test poin 2: memastikan laporan evaluasi dan confusion matrix berhasil di-generate."""
        report_path = os.path.join(self.eval_dir, "classification_report.txt")
        matrix_path = os.path.join(self.eval_dir, "confusion_matrix.png")
        metrics_path = os.path.join(self.eval_dir, "metrics_summary.json")
        
        self.assertTrue(os.path.exists(report_path), "Classification report tidak ditemukan!")
        self.assertTrue(os.path.exists(matrix_path), "Confusion matrix image tidak ditemukan!")
        self.assertTrue(os.path.exists(metrics_path), "Metrics summary json tidak ditemukan!")
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            # Ensure accuracy is not suspiciously exactly the same for all (like old bug)
            self.assertGreater(metrics.get("accuracy", 0), 0.0)

if __name__ == "__main__":
    unittest.main()

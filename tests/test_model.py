import json
import unittest
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]

class SavedModelTests(unittest.TestCase):
    def test_dataset_has_expected_features(self):
        data = pd.read_csv(ROOT / "dataset" / "flood.csv", nrows=5)
        features = json.loads((ROOT / "reports" / "feature_names.json").read_text())
        self.assertEqual(len(features), 20)
        self.assertTrue(set(features).issubset(data.columns))
        self.assertIn("FloodProbability", data.columns)

    def test_saved_model_accepts_one_row(self):
        features = json.loads((ROOT / "reports" / "feature_names.json").read_text())
        model = XGBClassifier()
        model.load_model(str(ROOT / "models" / "xgboost_flood_classifier.json"))
        sample = pd.DataFrame([[5] * len(features)], columns=features)
        prediction = model.predict_proba(sample)
        self.assertEqual(prediction.shape, (1, 2))
        self.assertAlmostEqual(float(prediction.sum()), 1.0, places=5)

if __name__ == "__main__":
    unittest.main()

"""
Unit tests for Isolation Forest anomaly detector, artifact serialization, and detection behavior.
Member 3: ML Analysis, Integration, QA and Deployment.
"""

import os
import shutil
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

# Ensure packages directory is in sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
PACKAGES_DIR = os.path.join(PROJECT_ROOT, "packages")
if PACKAGES_DIR not in sys.path:
    sys.path.insert(0, PACKAGES_DIR)

from health_analysis.baseline import BaselineManager
from health_analysis.features import FeatureExtractor, StreamingFeatureExtractor
from health_analysis.models import (
    AssessmentOutput,
    IsolationForestDetector,
    ModelConfig,
)
from health_analysis.schema import (
    ALL_FEATURE_NAMES_V1,
    FEATURE_SCHEMA_VERSION,
    PHYSIOLOGICAL_FEATURE_NAMES_V1,
    DataQualityState,
    FeatureExtractorConfig,
    FeatureRecord,
)


class TestIsolationForestModel(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = ModelConfig(
            model_version="1.0.0",
            n_estimators=50,
            max_samples=64,
            contamination=0.05,
            random_state=42,
            decision_threshold=0.50,
            min_training_samples=20,
        )

        # Generate synthetic normal training data (70 samples)
        np.random.seed(42)
        n_samples = 70
        data = {}
        for feat in self.config.feature_names:
            if "hr" in feat:
                data[feat] = np.random.normal(loc=80.0, scale=5.0, size=n_samples)
            elif "spo2" in feat:
                data[feat] = np.random.normal(loc=98.0, scale=1.0, size=n_samples)
            elif "temp" in feat:
                data[feat] = np.random.normal(loc=98.6, scale=0.4, size=n_samples)
            else:
                data[feat] = np.random.normal(loc=0.0, scale=0.5, size=n_samples)

        self.df_train = pd.DataFrame(data)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_fit_and_score_direction(self):
        """
        Verify that more anomalous inputs receive higher anomaly scores in [0.0, 1.0].
        """
        detector = IsolationForestDetector(config=self.config)
        detector.fit(self.df_train)

        # Normal sample: near training means
        normal_vec = [float(self.df_train[f].mean()) for f in self.config.feature_names]

        # Extreme anomalous sample: HR=160 bpm, SpO2=80%, Temp=104F
        anom_vec = []
        for feat in self.config.feature_names:
            if "hr" in feat:
                anom_vec.append(160.0)
            elif "spo2" in feat:
                anom_vec.append(80.0)
            elif "temp" in feat:
                anom_vec.append(104.0)
            elif "cross" in feat:
                anom_vec.append(15.0)
            else:
                anom_vec.append(10.0)

        X = np.array([normal_vec, anom_vec], dtype=float)
        scores, raw_dec = detector.score_samples(X)

        self.assertEqual(len(scores), 2)
        # Normal observation score should be low (< 0.40)
        self.assertLess(scores[0], 0.40)
        # Anomalous observation score should be high (> 0.50)
        self.assertGreater(scores[1], 0.50)
        # Score direction: anom > normal
        self.assertGreater(scores[1], scores[0])

    def test_reproducibility_with_fixed_seed(self):
        """
        Two model instances initialized with the same random seed must produce identical scores.
        """
        d1 = IsolationForestDetector(config=self.config).fit(self.df_train)
        d2 = IsolationForestDetector(config=self.config).fit(self.df_train)

        test_vec = self.df_train.iloc[:5].to_numpy()
        scores1, _ = d1.score_samples(test_vec)
        scores2, _ = d2.score_samples(test_vec)

        np.testing.assert_array_almost_equal(scores1, scores2, decimal=6)

    def test_artifact_save_and_load_roundtrip(self):
        """
        Verify model serialization and deserialization with metadata validation.
        """
        detector = IsolationForestDetector(config=self.config)
        detector.fit(self.df_train)

        detector.save(self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "model.joblib")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "metadata.json")))

        loaded = IsolationForestDetector.load(self.tmp_dir)
        self.assertTrue(loaded.is_fitted)
        self.assertEqual(loaded.config.model_version, self.config.model_version)
        self.assertEqual(loaded.config.feature_names, self.config.feature_names)

        # Scored outputs must match exactly
        test_vec = self.df_train.iloc[:3].to_numpy()
        scores_orig, _ = detector.score_samples(test_vec)
        scores_loaded, _ = loaded.score_samples(test_vec)

        np.testing.assert_array_almost_equal(scores_orig, scores_loaded, decimal=6)

    def test_feature_order_strictness(self):
        """
        Input DataFrames with shuffled columns must be realigned to the canonical feature order.
        """
        detector = IsolationForestDetector(config=self.config).fit(self.df_train)

        shuffled_cols = list(self.config.feature_names)
        np.random.shuffle(shuffled_cols)
        df_shuffled = self.df_train[shuffled_cols].iloc[:2].copy()

        scores_canonical, _ = detector.score_samples(self.df_train.iloc[:2])
        scores_shuffled, _ = detector.score_samples(df_shuffled)

        np.testing.assert_array_almost_equal(scores_canonical, scores_shuffled, decimal=6)

    def test_invalid_and_insufficient_inputs(self):
        """
        Model fit must reject non-finite inputs and matrices below min_training_samples.
        """
        detector = IsolationForestDetector(config=self.config)

        # Insufficient rows (e.g. 5 samples < min_training_samples=20)
        df_tiny = self.df_train.iloc[:5].copy()
        with self.assertRaises(ValueError):
            detector.fit(df_tiny)

        # NaN values in training input
        df_nan = self.df_train.copy()
        df_nan.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            detector.fit(df_nan)

        # Missing required feature column
        df_missing_col = self.df_train.drop(columns=[self.config.feature_names[0]])
        with self.assertRaises(ValueError):
            detector.fit(df_missing_col)

    def test_assess_record_insufficient_history_handling(self):
        """
        Records marked with INSUFFICIENT_HISTORY must receive score 0.0 and appropriate status.
        """
        detector = IsolationForestDetector(config=self.config).fit(self.df_train)

        rec = FeatureRecord(
            patient_id="P101",
            timestamp="2023-10-01 08:00:00",
            quality_state=DataQualityState.INSUFFICIENT_HISTORY,
            is_eligible_for_scoring=True,
        )

        assessment = detector.assess_record(rec)
        self.assertEqual(assessment.anomaly_score, 0.0)
        self.assertFalse(assessment.is_anomaly)
        self.assertEqual(assessment.result_status, "insufficient_history")

    def test_explainability_attribution_features(self):
        """
        High anomaly observations must produce top contributing features.
        """
        detector = IsolationForestDetector(config=self.config).fit(self.df_train)

        # Build reading starting from training means, then perturbing HR and SpO2
        base_features = {f: float(self.df_train[f].mean()) for f in self.config.feature_names}
        base_features["hr_latest"] = 160.0  # Acute tachycardia
        base_features["spo2_latest"] = 80.0  # Acute hypoxia

        rec = FeatureRecord(
            patient_id="P101",
            timestamp="2023-10-01 08:00:00",
            quality_state=DataQualityState.VALID,
            is_eligible_for_scoring=True,
            features=base_features,
        )

        assessment = detector.assess_record(rec)
        self.assertGreater(len(assessment.top_contributing_features), 0)
        top_feat_names = [f["feature"] for f in assessment.top_contributing_features]
        # At least one of the perturbed features must appear in top reasons
        self.assertTrue(any("hr" in fn or "spo2" in fn for fn in top_feat_names))


if __name__ == "__main__":
    unittest.main()

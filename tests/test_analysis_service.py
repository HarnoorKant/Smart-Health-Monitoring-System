"""
Comprehensive unit and integration tests for HealthAnalysisService and AssessmentResult contract.
Member 3: ML Analysis, Integration, QA and Deployment.
"""

import json
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

from health_analysis import (
    AssessmentResult,
    HealthAnalysisService,
    evaluate_safety_rules,
)
from health_analysis.baseline import BaselineManager
from health_analysis.models import IsolationForestDetector, ModelConfig
from health_analysis.schema import FEATURE_SCHEMA_VERSION, FeatureExtractorConfig


class TestAnalysisServiceIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_dir = os.path.join(PROJECT_ROOT, "models", "iforest_v1")
        raw_csv = os.path.join(PROJECT_ROOT, "data", "raw", "kaggle_health_dataset.csv")

        # Fit baseline on historical training rows
        df_hist = pd.read_csv(raw_csv)
        cls.baseline_mgr = BaselineManager().fit(df_hist.iloc[:1048])

    def setUp(self):
        # Create fresh service instance with baseline manager
        self.service = HealthAnalysisService(
            artifact_dir=self.model_dir,
            baseline_manager=self.baseline_mgr,
        )

    def _prime_patient(self, patient_id="P101", count=5):
        """Helper to prime streaming buffer with normal readings."""
        for i in range(count):
            self.service.assess_telemetry(
                patient_id=patient_id,
                timestamp=f"2023-10-12 08:0{i}:00",
                heart_rate=72.0,
                spo2=98.0,
                temperature_f=98.6,
                source_event_id=f"evt_prime_{patient_id}_{i}",
            )

    def test_normal_assessment_construction(self):
        """Verify normal telemetry produces result='normal_like' with low anomaly score."""
        self._prime_patient("P101", 5)

        res = self.service.assess_telemetry(
            patient_id="P101",
            timestamp="2023-10-12 08:30:00",
            heart_rate=74.0,
            spo2=98.0,
            temperature_f=98.6,
            source_event_id="evt_normal_1",
        )

        self.assertEqual(res.result, "normal_like")
        self.assertLess(res.score, 0.40)
        self.assertEqual(res.patient_id, "P101")
        self.assertIn("All vital signs", res.explanation[0])
        self.assertIsInstance(res.feature_summary, dict)

    def test_anomalous_assessment_construction(self):
        """Verify combined anomaly produces result='anomalous' with score >= 0.50."""
        self._prime_patient("P101", 5)

        res = self.service.assess_telemetry(
            patient_id="P101",
            timestamp="2023-10-12 08:30:00",
            heart_rate=128.0,
            spo2=91.0,
            temperature_f=102.4,
            source_event_id="evt_anom_1",
        )

        self.assertEqual(res.result, "anomalous")
        self.assertGreaterEqual(res.score, 0.50)
        self.assertGreater(len(res.explanation), 0)

    def test_insufficient_history_returns_insufficient_data(self):
        """A new unprimed patient with < 5 samples must return result='insufficient_data'."""
        fresh_service = HealthAnalysisService(
            artifact_dir=self.model_dir,
            baseline_manager=self.baseline_mgr,
        )

        res = fresh_service.assess_telemetry(
            patient_id="NEW_PATIENT_999",
            timestamp="2023-10-12 08:00:00",
            heart_rate=72.0,
            spo2=98.0,
            temperature_f=98.6,
        )

        self.assertEqual(res.result, "insufficient_data")
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("Insufficient historical observations" in exp for exp in res.explanation))

    def test_invalid_timestamps_and_non_finite_measurements(self):
        """Corrupted inputs must return result='insufficient_data' and never 'normal_like'."""
        self._prime_patient("P101", 5)

        # Invalid timestamp string
        res_bad_ts = self.service.assess_telemetry(
            patient_id="P101",
            timestamp="invalid_not_a_date",
            heart_rate=72.0,
            spo2=98.0,
            temperature_f=98.6,
        )
        self.assertEqual(res_bad_ts.result, "insufficient_data")
        self.assertEqual(res_bad_ts.score, 0.0)

        # Non-finite NaN in heart rate
        res_nan = self.service.assess_telemetry(
            patient_id="P101",
            timestamp="2023-10-12 08:30:00",
            heart_rate=np.nan,
            spo2=98.0,
            temperature_f=98.6,
        )
        self.assertEqual(res_nan.result, "insufficient_data")
        self.assertEqual(res_nan.score, 0.0)

        # Missing None in SpO2
        res_none = self.service.assess_telemetry(
            patient_id="P101",
            timestamp="2023-10-12 08:30:00",
            heart_rate=75.0,
            spo2=None,
            temperature_f=98.6,
        )
        self.assertEqual(res_none.result, "insufficient_data")
        self.assertEqual(res_none.score, 0.0)

    def test_incompatible_model_schema_version_rejected(self):
        """Loading a model metadata file with mismatched schema version must raise ValueError."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Copy model files to temp dir
            shutil.copy(os.path.join(self.model_dir, "model.joblib"), os.path.join(temp_dir, "model.joblib"))
            with open(os.path.join(self.model_dir, "metadata.json"), "r") as f:
                meta = json.load(f)

            # Mutate schema version
            meta["feature_schema_version"] = "999.0.0"
            with open(os.path.join(temp_dir, "metadata.json"), "w") as f:
                json.dump(meta, f)

            with self.assertRaises(ValueError):
                IsolationForestDetector.load(temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_model_artifact_missing_or_corrupted_graceful_handling(self):
        """If model artifact path is invalid, service must log warning and return insufficient_data."""
        broken_service = HealthAnalysisService(artifact_dir="/non/existent/model/path")
        self.assertIsNone(broken_service.detector)

        res = broken_service.assess_telemetry(
            patient_id="P101",
            timestamp="2023-10-12 08:00:00",
            heart_rate=72.0,
            spo2=98.0,
            temperature_f=98.6,
        )
        self.assertEqual(res.result, "insufficient_data")
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("not loaded or unavailable" in exp for exp in res.explanation))

    def test_json_serialization_roundtrip(self):
        """AssessmentResult.to_dict() must be 100% valid JSON without serialization errors."""
        self._prime_patient("P101", 5)

        res = self.service.assess_telemetry(
            patient_id="P101",
            timestamp="2023-10-12 08:30:00",
            heart_rate=128.0,
            spo2=91.0,
            temperature_f=102.4,
            source_event_id="evt_test_json",
        )

        res_dict = res.to_dict()
        serialized_json = json.dumps(res_dict)
        self.assertIsInstance(serialized_json, str)

        deserialized = json.loads(serialized_json)
        self.assertEqual(deserialized["patient_id"], "P101")
        self.assertEqual(deserialized["result"], "anomalous")
        self.assertEqual(deserialized["assessment_id"], "asmt_evt_test_json")
        self.assertIsInstance(deserialized["feature_summary"], dict)

    def test_deterministic_repeated_input(self):
        """Scoring identical inputs must produce identical AssessmentResult records."""
        self._prime_patient("P101", 5)

        # Call twice with identical parameters
        r1 = self.service.assess_telemetry(
            patient_id="P101",
            timestamp="2023-10-12 08:30:00",
            heart_rate=128.0,
            spo2=91.0,
            temperature_f=102.4,
        )

        # Create identical second service state
        svc2 = HealthAnalysisService(artifact_dir=self.model_dir, baseline_manager=self.baseline_mgr)
        for i in range(5):
            svc2.assess_telemetry("P101", f"2023-10-12 08:0{i}:00", 72.0, 98.0, 98.6)

        r2 = svc2.assess_telemetry(
            patient_id="P101",
            timestamp="2023-10-12 08:30:00",
            heart_rate=128.0,
            spo2=91.0,
            temperature_f=102.4,
        )

        self.assertEqual(r1.score, r2.score)
        self.assertEqual(r1.result, r2.result)
        self.assertEqual(r1.feature_summary, r2.feature_summary)

    def test_independent_rule_evaluation_when_ml_fails(self):
        """Clinical safety rules must trigger even if ML is completely broken."""
        broken_service = HealthAnalysisService(artifact_dir="/broken/path")
        self.assertIsNone(broken_service.detector)

        # Severe hypoxemia reading
        hr, spo2, temp = 110.0, 85.0, 98.6

        # Step A: Rule evaluation
        rules = evaluate_safety_rules(hr, spo2, temp)
        self.assertTrue(any(r.rule_code == "CRITICAL_HYPOXPOXEMIA" or r.severity == "CRITICAL" for r in rules))

        # Step B: ML evaluation returns insufficient_data
        ml_res = broken_service.assess_telemetry("P101", "2023-10-12 08:00:00", hr, spo2, temp)
        self.assertEqual(ml_res.result, "insufficient_data")

        # Key integration invariant: Rule engine protected patient despite ML failure
        self.assertTrue(any(r.severity == "CRITICAL" for r in rules))

    def test_duplicate_event_handling_idempotency(self):
        """Providing the same source_event_id produces identical assessment_id for deduplication."""
        self._prime_patient("P101", 5)

        event_id = "evt_idempotent_999"
        res1 = self.service.assess_telemetry(
            patient_id="P101",
            timestamp="2023-10-12 08:30:00",
            heart_rate=75.0,
            spo2=98.0,
            temperature_f=98.6,
            source_event_id=event_id,
        )

        self.assertEqual(res1.assessment_id, f"asmt_{event_id}")

    def test_live_simulator_input_compatibility(self):
        """
        Simulator payloads with respiratory_rate and blood_pressure must be accepted safely
        without polluting the 3-measurement historical model vector.
        """
        self._prime_patient("P101", 5)

        res = self.service.assess_telemetry(
            patient_id="P101",
            timestamp="2023-10-12 08:30:00",
            heart_rate=75.0,
            spo2=98.0,
            temperature_f=98.6,
            respiratory_rate=18.0,       # Live simulator field
            systolic_bp=120.0,          # Live simulator field
            diastolic_bp=80.0,           # Live simulator field
            is_simulated=True,
        )

        self.assertEqual(res.result, "normal_like")
        self.assertTrue(res.is_simulated)
        # Verify unmodeled fields are recorded in summary for traceability without model contamination
        self.assertEqual(res.feature_summary.get("respiratory_rate_unmodeled"), 18.0)
        self.assertEqual(res.feature_summary.get("blood_pressure_unmodeled"), "120/80")


if __name__ == "__main__":
    unittest.main()

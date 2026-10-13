"""
Unit tests for health dataset validation logic.
Member 3: ML Analysis, Integration, QA and Deployment.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure packages/ directory is in sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
PACKAGES_DIR = os.path.join(PROJECT_ROOT, "packages")
if PACKAGES_DIR not in sys.path:
    sys.path.insert(0, PACKAGES_DIR)

from health_analysis.validation import (
    EXPECTED_COLUMNS,
    FORBIDDEN_HISTORICAL_COLUMNS,
    audit_dataframe,
    audit_dataset_file,
    compute_sha256,
)


class TestDatasetValidationLogic(unittest.TestCase):
    def setUp(self):
        # A valid baseline fixture matching the exact Kaggle schema
        self.valid_data = pd.DataFrame({
            "patient_id": ["P101", "P101", "P102", "P102"],
            "heart_rate": [72, 75, 80, 82],
            "spo2": [98.5, 97.9, 96.0, 96.5],
            "temperature_f": [98.6, 98.7, 99.1, 98.9],
            "timestamp": [
                "2023-10-01 08:00:00",
                "2023-10-01 08:15:00",
                "2023-10-01 08:00:00",
                "2023-10-01 08:30:00",
            ],
        })

    def test_valid_dataframe_passes_with_no_error_flags(self):
        result = audit_dataframe(self.valid_data)
        self.assertEqual(result.row_count, 4)
        self.assertEqual(result.column_count, 5)
        self.assertEqual(result.full_duplicate_rows, 0)
        self.assertEqual(result.duplicate_patient_timestamp_pairs, 0)
        self.assertEqual(result.invalid_timestamp_count, 0)
        self.assertFalse(result.has_diagnostic_labels)
        self.assertFalse(result.has_anomaly_labels)
        self.assertFalse(result.has_respiratory_rate)
        self.assertFalse(result.has_blood_pressure)
        self.assertEqual(result.data_quality_flags, [])

    def test_missing_expected_column_flagged(self):
        invalid_df = self.valid_data.drop(columns=["spo2"])
        result = audit_dataframe(invalid_df)
        self.assertTrue(any("MISSING_EXPECTED_COLUMN: spo2" in f for f in result.data_quality_flags))

    def test_forbidden_columns_flagged(self):
        # Adding blood pressure or anomaly label must be flagged
        forbidden_df = self.valid_data.copy()
        forbidden_df["blood_pressure"] = "120/80"
        forbidden_df["is_anomaly"] = 0
        result = audit_dataframe(forbidden_df)
        self.assertTrue(any("UNEXPECTED_COLUMN_PRESENT: blood_pressure" in f for f in result.data_quality_flags))
        self.assertTrue(any("UNEXPECTED_COLUMN_PRESENT: is_anomaly" in f for f in result.data_quality_flags))
        self.assertTrue(result.has_blood_pressure)
        self.assertTrue(result.has_anomaly_labels)

    def test_duplicate_row_detection(self):
        dupe_df = pd.concat([self.valid_data, self.valid_data.iloc[[0]]], ignore_index=True)
        result = audit_dataframe(dupe_df)
        self.assertEqual(result.full_duplicate_rows, 1)
        self.assertEqual(result.duplicate_patient_timestamp_pairs, 1)
        self.assertTrue(any("FULL_DUPLICATE_ROWS" in f for f in result.data_quality_flags))

    def test_duplicate_patient_timestamp_with_different_values(self):
        # Same patient and timestamp, but conflicting values
        conflicting_row = pd.DataFrame([{
            "patient_id": "P101",
            "heart_rate": 110,
            "spo2": 92.0,
            "temperature_f": 101.2,
            "timestamp": "2023-10-01 08:00:00",
        }])
        key_dupe_df = pd.concat([self.valid_data, conflicting_row], ignore_index=True)
        result = audit_dataframe(key_dupe_df)
        self.assertEqual(result.full_duplicate_rows, 0)
        self.assertEqual(result.duplicate_patient_timestamp_pairs, 1)
        self.assertTrue(any("DUPLICATE_PATIENT_TIMESTAMP_PAIRS" in f for f in result.data_quality_flags))

    def test_non_numeric_and_nan_detection(self):
        bad_num_df = self.valid_data.copy()
        bad_num_df["heart_rate"] = bad_num_df["heart_rate"].astype(object)
        bad_num_df.loc[0, "heart_rate"] = "corrupted_str"
        bad_num_df.loc[1, "spo2"] = np.nan
        result = audit_dataframe(bad_num_df)
        self.assertTrue(any("NON_NUMERIC_VALUES_IN_HEART_RATE" in f for f in result.data_quality_flags))
        self.assertTrue(any("MISSING_VALUES_IN_SPO2" in f for f in result.data_quality_flags))

    def test_non_chronological_sequence_flagged(self):
        out_of_order = pd.DataFrame({
            "patient_id": ["P101", "P101"],
            "heart_rate": [70, 75],
            "spo2": [98.0, 97.0],
            "temperature_f": [98.6, 98.6],
            "timestamp": ["2023-10-01 10:00:00", "2023-10-01 09:00:00"],  # Decreasing!
        })
        result = audit_dataframe(out_of_order)
        self.assertFalse(result.temporal_profiles["P101"].is_chronological)
        self.assertTrue(any("NON_CHRONOLOGICAL_DATA_FOR_PATIENT_P101" in f for f in result.data_quality_flags))

    def test_actual_kaggle_raw_dataset_file(self):
        raw_csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "raw", "kaggle_health_dataset.csv"
        )
        if not os.path.exists(raw_csv_path):
            self.skipTest(f"Raw CSV not found at {raw_csv_path}")

        result = audit_dataset_file(raw_csv_path)

        # Confirm reference expectations
        self.assertEqual(result.row_count, 1500)
        self.assertEqual(result.columns, EXPECTED_COLUMNS)
        self.assertEqual(result.full_duplicate_rows, 0)
        self.assertEqual(result.duplicate_patient_timestamp_pairs, 0)
        self.assertEqual(result.invalid_timestamp_count, 0)
        self.assertEqual(result.unique_patients, ["P101", "P102", "P103", "P104", "P105"])
        self.assertEqual(result.patient_observation_counts, {
            "P101": 293,
            "P102": 302,
            "P103": 296,
            "P104": 325,
            "P105": 284,
        })
        self.assertEqual(result.sha256_hash, "50f391190cd58f87636c0f6ae45731f97791563210275f14e5208c75be96ec82")
        self.assertFalse(result.has_diagnostic_labels)
        self.assertFalse(result.has_anomaly_labels)
        self.assertFalse(result.has_respiratory_rate)
        self.assertFalse(result.has_blood_pressure)
        self.assertEqual(result.data_quality_flags, [])


if __name__ == "__main__":
    unittest.main()

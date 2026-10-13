"""
Unit tests for health telemetry feature engineering, baseline calculation, and leakage prevention.
Member 3: ML Analysis, Integration, QA and Deployment.
"""

import math
import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure packages/ directory is in sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
PACKAGES_DIR = os.path.join(PROJECT_ROOT, "packages")
if PACKAGES_DIR not in sys.path:
    sys.path.insert(0, PACKAGES_DIR)

from health_analysis.baseline import (
    BaselineManager,
    IncrementalBaselineTracker,
    calculate_metric_baseline,
)
from health_analysis.features import (
    FeatureExtractor,
    StreamingFeatureExtractor,
    compute_window_statistics,
)
from health_analysis.schema import (
    ALL_FEATURE_NAMES_V1,
    DATA_QUALITY_FEATURE_NAMES_V1,
    PHYSIOLOGICAL_FEATURE_NAMES_V1,
    DataQualityState,
    FeatureExtractorConfig,
)
from health_analysis.units import (
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    normalize_temperature,
)


class TestFeatureEngineeringPipeline(unittest.TestCase):
    def setUp(self):
        self.config = FeatureExtractorConfig(
            window_duration_minutes=120.0,
            min_baseline_samples=5,
            max_gap_minutes=180.0,
            temperature_unit="F",
        )

    def test_temperature_unit_conversions(self):
        # 32 F == 0 C
        self.assertAlmostEqual(fahrenheit_to_celsius(32.0), 0.0, places=5)
        self.assertAlmostEqual(celsius_to_fahrenheit(0.0), 32.0, places=5)

        # 212 F == 100 C
        self.assertAlmostEqual(fahrenheit_to_celsius(212.0), 100.0, places=5)
        self.assertAlmostEqual(celsius_to_fahrenheit(100.0), 212.0, places=5)

        # 98.6 F == 37.0 C
        self.assertAlmostEqual(fahrenheit_to_celsius(98.6), 37.0, places=2)
        self.assertAlmostEqual(celsius_to_fahrenheit(37.0), 98.6, places=2)

        # Array & NaN handling
        arr_f = np.array([32.0, 212.0, np.nan])
        arr_c = fahrenheit_to_celsius(arr_f)
        self.assertAlmostEqual(arr_c[0], 0.0, places=5)
        self.assertAlmostEqual(arr_c[1], 100.0, places=5)
        self.assertTrue(np.isnan(arr_c[2]))

    def test_hand_calculated_window_statistics(self):
        """
        Verify mathematical calculations against known manual calculations.
        Reading 1: t = 0.0h, HR = 80
        Reading 2: t = 0.5h (30 min later), HR = 90
        """
        # Window of [80, 90], prev_val=80, elapsed=0.5h
        stats = compute_window_statistics(
            values=[80.0, 90.0],
            prev_val=80.0,
            elapsed_hours_from_prev=0.5,
            metric_baseline=None,
        )

        self.assertEqual(stats["latest"], 90.0)
        self.assertEqual(stats["rolling_mean"], 85.0)
        self.assertEqual(stats["rolling_median"], 85.0)
        # sample std: sqrt(((80-85)^2 + (90-85)^2)/(2-1)) = sqrt(50) = 7.0710678...
        self.assertAlmostEqual(stats["rolling_std"], math.sqrt(50), places=5)
        # MAD: median(|80-85|=5, |90-85|=5) = 5.0
        self.assertEqual(stats["rolling_mad"], 5.0)
        self.assertEqual(stats["window_min"], 80.0)
        self.assertEqual(stats["window_max"], 90.0)
        # abs change: 90 - 80 = 10
        self.assertEqual(stats["abs_change"], 10.0)
        # rel change: 10 / 80 = 0.125
        self.assertEqual(stats["rel_change"], 0.125)
        # trend slope: 10 / 0.5h = 20.0 bpm/hr
        self.assertEqual(stats["trend_slope"], 20.0)

    def test_hand_calculated_end_to_end_observation(self):
        """
        Verify end-to-end extraction against manual hand calculation.
        """
        df = pd.DataFrame({
            "patient_id": ["P101", "P101"],
            "timestamp": ["2023-10-01 08:00:00", "2023-10-01 08:30:00"],
            "heart_rate": [80, 90],
            "spo2": [98.0, 96.0],
            "temperature_f": [98.6, 99.2],
        })

        extractor = FeatureExtractor(config=self.config)
        records = extractor.extract_from_dataframe(df)

        self.assertEqual(len(records), 2)
        rec2 = records[1]

        # Verify HR features
        self.assertEqual(rec2.features["hr_latest"], 90.0)
        self.assertEqual(rec2.features["hr_rolling_mean"], 85.0)
        self.assertEqual(rec2.features["hr_trend_slope"], 20.0)

        # Verify SpO2 features: change = 96 - 98 = -2; slope = -2 / 0.5h = -4.0 %/hr
        self.assertEqual(rec2.features["spo2_latest"], 96.0)
        self.assertEqual(rec2.features["spo2_abs_change"], -2.0)
        self.assertEqual(rec2.features["spo2_trend_slope"], -4.0)

        # Verify cross metric hypoxia tachycardia ratio: 90 / 96 = 0.9375
        self.assertAlmostEqual(rec2.features["cross_hypoxia_tachycardia_ratio"], 90.0 / 96.0, places=4)
        self.assertEqual(rec2.features["quality_sample_count"], 2.0)
        self.assertEqual(rec2.features["quality_elapsed_minutes"], 30.0)

    def test_chronological_ordering_is_enforced(self):
        """
        Input rows supplied out of order must be sorted chronologically before windowing.
        """
        df_unordered = pd.DataFrame({
            "patient_id": ["P101", "P101"],
            "timestamp": ["2023-10-01 09:00:00", "2023-10-01 08:00:00"],  # Reversed!
            "heart_rate": [95, 75],
            "spo2": [95.0, 99.0],
            "temperature_f": [99.0, 98.4],
        })

        extractor = FeatureExtractor(config=self.config)
        records = extractor.extract_from_dataframe(df_unordered)

        # Output records must be chronologically ordered
        self.assertEqual(records[0].timestamp, "2023-10-01 08:00:00")
        self.assertEqual(records[0].features["hr_latest"], 75.0)
        self.assertEqual(records[1].timestamp, "2023-10-01 09:00:00")
        self.assertEqual(records[1].features["hr_latest"], 95.0)
        self.assertEqual(records[1].features["hr_trend_slope"], 20.0)  # (95-75)/1.0h

    def test_multiple_patients_never_share_baselines_or_windows(self):
        """
        Ensure strict patient isolation: P101 and P102 must never share state.
        """
        df = pd.DataFrame({
            "patient_id": ["P101", "P102", "P101", "P102"],
            "timestamp": [
                "2023-10-01 08:00:00",
                "2023-10-01 08:00:00",
                "2023-10-01 08:30:00",
                "2023-10-01 08:30:00",
            ],
            "heart_rate": [120, 60, 125, 62],
            "spo2": [90.0, 99.0, 91.0, 99.0],
            "temperature_f": [101.0, 98.6, 101.2, 98.6],
        })

        extractor = FeatureExtractor(config=self.config)
        records = extractor.extract_from_dataframe(df)

        p101_recs = [r for r in records if r.patient_id == "P101"]
        p102_recs = [r for r in records if r.patient_id == "P102"]

        self.assertEqual(len(p101_recs), 2)
        self.assertEqual(len(p102_recs), 2)

        # P101 rolling mean is ~122.5, P102 rolling mean is ~61.0
        self.assertEqual(p101_recs[1].features["hr_rolling_mean"], 122.5)
        self.assertEqual(p102_recs[1].features["hr_rolling_mean"], 61.0)
        self.assertEqual(p101_recs[1].features["hr_abs_change"], 5.0)
        self.assertEqual(p102_recs[1].features["hr_abs_change"], 2.0)

    def test_constant_measurements_and_zero_variability_baselines(self):
        """
        Safe handling of zero-variability baselines without zero-division errors.
        """
        # Baseline with identical values (std = 0, MAD = 0)
        identical_vals = [70.0, 70.0, 70.0, 70.0, 70.0]
        base = calculate_metric_baseline(identical_vals)
        self.assertEqual(base.mad, 0.0)
        self.assertEqual(base.std, 0.0)

        # Deviation when value is also 70.0 -> normalized dev should be 0.0
        raw_dev, norm_dev = base.compute_deviation(70.0)
        self.assertEqual(raw_dev, 0.0)
        self.assertEqual(norm_dev, 0.0)

        # Deviation when value changes to 85.0 -> safe non-infinite output
        raw_dev2, norm_dev2 = base.compute_deviation(85.0)
        self.assertEqual(raw_dev2, 15.0)
        self.assertEqual(norm_dev2, 1.0)  # Safe direction indicator

    def test_missing_and_non_finite_readings(self):
        """
        Missing values must not be replaced with zero or assumed normal.
        """
        df_missing = pd.DataFrame({
            "patient_id": ["P101", "P101", "P101"],
            "timestamp": ["2023-10-01 08:00:00", "2023-10-01 08:30:00", "2023-10-01 09:00:00"],
            "heart_rate": [80, np.nan, 85],
            "spo2": [98.0, np.nan, 97.0],
            "temperature_f": [98.6, np.nan, 98.7],
        })

        extractor = FeatureExtractor(config=self.config)
        records = extractor.extract_from_dataframe(df_missing)

        # Reading 2 is fully missing
        self.assertEqual(records[1].quality_state, DataQualityState.EXCESSIVE_MISSINGNESS)
        self.assertFalse(records[1].is_eligible_for_scoring)
        self.assertEqual(records[1].features["quality_missing_count"], 3.0)

        # Reading 3 recovers: rolling window contains valid readings 1 and 3 (mean = (80+85)/2 = 82.5)
        self.assertEqual(records[2].features["hr_rolling_mean"], 82.5)
        self.assertNotEqual(records[2].features["hr_latest"], 0.0)

    def test_irregular_sampling_intervals_and_gap_exceeded(self):
        """
        Test that trend slope uses elapsed time, and prolonged gaps (>max_gap) are flagged.
        """
        df_gap = pd.DataFrame({
            "patient_id": ["P101", "P101"],
            "timestamp": ["2023-10-01 08:00:00", "2023-10-01 13:00:00"],  # 5-hour gap (300 min)
            "heart_rate": [75, 95],
            "spo2": [98.0, 95.0],
            "temperature_f": [98.6, 99.0],
        })

        extractor = FeatureExtractor(config=self.config)
        records = extractor.extract_from_dataframe(df_gap)

        rec2 = records[1]
        self.assertEqual(rec2.features["quality_gap_exceeded"], 1.0)
        self.assertEqual(rec2.features["quality_elapsed_minutes"], 300.0)
        # Because gap exceeded 180 min, slope is decoupled across the large void
        self.assertEqual(rec2.features["hr_trend_slope"], 0.0)

    def test_insufficient_history_behavior(self):
        """
        Before 5 baseline samples are reached, baseline dev is 0.0 and flagged.
        """
        df_short = pd.DataFrame({
            "patient_id": ["P101", "P101"],
            "timestamp": ["2023-10-01 08:00:00", "2023-10-01 08:30:00"],
            "heart_rate": [75, 80],
            "spo2": [98.0, 97.0],
            "temperature_f": [98.6, 98.7],
        })

        # No baseline fitted
        extractor = FeatureExtractor(config=self.config, baseline_manager=None)
        records = extractor.extract_from_dataframe(df_short)

        self.assertEqual(records[0].features["quality_has_baseline"], 0.0)
        self.assertEqual(records[0].features["hr_baseline_dev"], 0.0)
        self.assertEqual(records[0].quality_state, DataQualityState.INSUFFICIENT_HISTORY)

    def test_prevention_of_future_data_leakage(self):
        """
        Verify fit/transform leakage prevention:
        Baseline fit on training split (days 1-3) must remain frozen during test split (days 4-5).
        """
        train_df = pd.DataFrame({
            "patient_id": ["P101"] * 5,
            "timestamp": [
                f"2023-10-01 0{i}:00:00" for i in range(1, 6)
            ],
            "heart_rate": [70, 72, 70, 74, 70],  # Median = 70.0
            "spo2": [98.0, 98.0, 98.0, 98.0, 98.0],
            "temperature_f": [98.6, 98.6, 98.6, 98.6, 98.6],
        })

        # Future evaluation data where heart rate spikes to 120
        eval_df = pd.DataFrame({
            "patient_id": ["P101"],
            "timestamp": ["2023-10-02 08:00:00"],
            "heart_rate": [120],
            "spo2": [94.0],
            "temperature_f": [100.5],
        })

        bm = BaselineManager(min_baseline_samples=5)
        bm.fit(train_df)

        # Baseline median must be frozen at 70.0
        self.assertEqual(bm.get_patient_baseline("P101").hr.median, 70.0)

        extractor = FeatureExtractor(config=self.config, baseline_manager=bm)
        eval_records = extractor.extract_from_dataframe(eval_df)

        # Evaluation deviation must be 120 - 70 = 50.0
        self.assertEqual(eval_records[0].features["hr_baseline_dev"], 50.0)

    def test_streaming_and_batch_equivalence(self):
        """
        StreamingFeatureExtractor (simulating real-time live events) produces
        identical features to batch FeatureExtractor.
        """
        df = pd.DataFrame({
            "patient_id": ["P101"] * 6,
            "timestamp": [
                "2023-10-01 08:00:00",
                "2023-10-01 08:15:00",
                "2023-10-01 08:30:00",
                "2023-10-01 08:45:00",
                "2023-10-01 09:00:00",
                "2023-10-01 09:30:00",
            ],
            "heart_rate": [72, 75, 78, 80, 85, 90],
            "spo2": [98.0, 98.0, 97.5, 97.0, 96.0, 95.0],
            "temperature_f": [98.6, 98.6, 98.7, 98.8, 99.1, 99.5],
        })

        bm = BaselineManager(min_baseline_samples=5).fit(df.iloc[:5])

        # Batch extraction
        batch_extractor = FeatureExtractor(config=self.config, baseline_manager=bm)
        batch_records = batch_extractor.extract_from_dataframe(df)

        # Streaming extraction
        streaming_extractor = StreamingFeatureExtractor(config=self.config, baseline_manager=bm)
        stream_records = []
        for _, row in df.iterrows():
            rec = streaming_extractor.process_reading(
                patient_id=row["patient_id"],
                timestamp=row["timestamp"],
                heart_rate=row["heart_rate"],
                spo2=row["spo2"],
                temperature_f=row["temperature_f"],
            )
            stream_records.append(rec)

        self.assertEqual(len(batch_records), len(stream_records))
        for b_rec, s_rec in zip(batch_records, stream_records):
            for feat_name in PHYSIOLOGICAL_FEATURE_NAMES_V1:
                self.assertAlmostEqual(
                    b_rec.features[feat_name],
                    s_rec.features[feat_name],
                    places=5,
                    msg=f"Mismatch for feature {feat_name} at {b_rec.timestamp}"
                )

    def test_determinism_repeated_execution(self):
        """
        Processing the same input twice produces bit-for-bit identical vectors.
        """
        df = pd.DataFrame({
            "patient_id": ["P101", "P101"],
            "timestamp": ["2023-10-01 08:00:00", "2023-10-01 08:20:00"],
            "heart_rate": [72, 84],
            "spo2": [98.0, 96.0],
            "temperature_f": [98.6, 99.0],
        })

        extractor = FeatureExtractor(config=self.config)
        recs_1 = extractor.extract_from_dataframe(df)
        recs_2 = extractor.extract_from_dataframe(df)

        vec_1 = recs_1[1].to_feature_vector()
        vec_2 = recs_2[1].to_feature_vector()

        self.assertEqual(vec_1, vec_2)
        self.assertEqual(len(vec_1), len(ALL_FEATURE_NAMES_V1))


if __name__ == "__main__":
    unittest.main()

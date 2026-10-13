"""
Window-based feature extraction for health telemetry.
Member 3: ML Analysis, Integration, QA and Deployment.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from health_analysis.baseline import BaselineManager, MetricBaseline, PatientBaseline
from health_analysis.schema import (
    ALL_FEATURE_NAMES_V1,
    DATA_QUALITY_FEATURE_NAMES_V1,
    FEATURE_SCHEMA_VERSION,
    PHYSIOLOGICAL_FEATURE_NAMES_V1,
    DataQualityState,
    FeatureExtractorConfig,
    FeatureRecord,
)
from health_analysis.units import normalize_temperature


@dataclass
class Observation:
    """Internal timestamped observation for a single patient."""
    timestamp: pd.Timestamp
    heart_rate: Optional[float]
    spo2: Optional[float]
    temperature: Optional[float]
    raw_timestamp_str: str


def compute_window_statistics(
    values: List[float],
    prev_val: Optional[float],
    elapsed_hours_from_prev: Optional[float],
    metric_baseline: Optional[MetricBaseline],
    epsilon: float = 1e-6,
    mad_normal_factor: float = 1.4826,
) -> Dict[str, float]:
    """
    Compute statistical and temporal features for a single metric over a time window.
    """
    clean = [v for v in values if v is not None and not np.isnan(v) and not np.isinf(v)]
    if not clean:
        return {
            "latest": 0.0,
            "rolling_mean": 0.0,
            "rolling_median": 0.0,
            "rolling_std": 0.0,
            "rolling_mad": 0.0,
            "window_min": 0.0,
            "window_max": 0.0,
            "abs_change": 0.0,
            "rel_change": 0.0,
            "trend_slope": 0.0,
            "baseline_dev": 0.0,
            "norm_baseline_dev": 0.0,
        }

    latest = float(clean[-1])
    n = len(clean)
    mean_val = float(np.mean(clean))
    median_val = float(np.median(clean))
    std_val = float(np.std(clean, ddof=1)) if n > 1 else 0.0
    abs_devs = np.abs(np.array(clean) - median_val)
    mad_val = float(np.median(abs_devs))
    min_val = float(np.min(clean))
    max_val = float(np.max(clean))

    # Change from previous reading
    if prev_val is not None and not np.isnan(prev_val) and not np.isinf(prev_val):
        abs_change = float(latest - prev_val)
        denom = abs(prev_val) if abs(prev_val) > epsilon else epsilon
        rel_change = float(abs_change / denom)

        # Trend slope using elapsed time (rate of change per hour)
        if elapsed_hours_from_prev is not None and elapsed_hours_from_prev > (epsilon / 3600.0):
            trend_slope = float(abs_change / elapsed_hours_from_prev)
        else:
            trend_slope = 0.0
    else:
        abs_change = 0.0
        rel_change = 0.0
        trend_slope = 0.0

    # Baseline deviation
    if metric_baseline is not None:
        base_dev, norm_base_dev = metric_baseline.compute_deviation(
            latest, epsilon=epsilon, mad_normal_factor=mad_normal_factor
        )
    else:
        base_dev = 0.0
        norm_base_dev = 0.0

    return {
        "latest": latest,
        "rolling_mean": mean_val,
        "rolling_median": median_val,
        "rolling_std": std_val,
        "rolling_mad": mad_val,
        "window_min": min_val,
        "window_max": max_val,
        "abs_change": abs_change,
        "rel_change": rel_change,
        "trend_slope": trend_slope,
        "baseline_dev": base_dev,
        "norm_baseline_dev": norm_base_dev,
    }


class FeatureExtractor:
    """
    Extracts time-aware rolling window features for patient telemetry.
    Supports deterministic batch processing of historical data and
    leakage-free evaluation with frozen baseline managers.
    """

    def __init__(
        self,
        config: Optional[FeatureExtractorConfig] = None,
        baseline_manager: Optional[BaselineManager] = None,
    ):
        self.config = config or FeatureExtractorConfig()
        self.baseline_manager = baseline_manager

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
    ) -> List[FeatureRecord]:
        """
        Extract feature records from a DataFrame.
        Ensures strict chronological sorting per patient_id before processing.
        """
        required = {"patient_id", "timestamp", "heart_rate", "spo2", "temperature_f"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Input DataFrame missing required columns: {missing}")

        records: List[FeatureRecord] = []

        # Process each patient strictly in isolation
        for pid, group in df.groupby("patient_id"):
            pid_str = str(pid)
            # Sort chronologically by timestamp
            p_df = group.copy()
            p_df["_parsed_ts"] = pd.to_datetime(p_df["timestamp"], errors="coerce")
            p_df = p_df.sort_values(by="_parsed_ts", ascending=True).reset_index(drop=True)

            patient_records = self._process_patient_series(pid_str, p_df)
            records.extend(patient_records)

        return records

    def _process_patient_series(
        self,
        patient_id: str,
        df_patient: pd.DataFrame,
    ) -> List[FeatureRecord]:
        """Process a single patient's chronological observations."""
        records: List[FeatureRecord] = []
        patient_baseline = (
            self.baseline_manager.get_patient_baseline(patient_id)
            if self.baseline_manager is not None
            else None
        )

        history: List[Observation] = []

        for idx, row in df_patient.iterrows():
            ts_parsed = row["_parsed_ts"]
            raw_ts = str(row["timestamp"])

            # Validate timestamp
            if pd.isnull(ts_parsed):
                record = FeatureRecord(
                    patient_id=patient_id,
                    timestamp=raw_ts,
                    quality_state=DataQualityState.CORRUPTED_MEASUREMENT,
                    is_eligible_for_scoring=False,
                    quality_flags=["INVALID_TIMESTAMP"],
                )
                records.append(record)
                continue

            # Extract and normalize measurements
            hr = row["heart_rate"]
            spo2 = row["spo2"]
            temp_raw = row["temperature_f"]

            hr_clean = float(hr) if pd.notnull(hr) and not np.isinf(hr) else None
            spo2_clean = float(spo2) if pd.notnull(spo2) and not np.isinf(spo2) else None
            temp_clean = None
            if pd.notnull(temp_raw) and not np.isinf(temp_raw):
                temp_clean = normalize_temperature(
                    float(temp_raw), source_unit="F", target_unit=self.config.temperature_unit
                )

            current_obs = Observation(
                timestamp=ts_parsed,
                heart_rate=hr_clean,
                spo2=spo2_clean,
                temperature=temp_clean,
                raw_timestamp_str=raw_ts,
            )

            # Build window: observations within [ts - window_duration, ts]
            window_cutoff = ts_parsed - pd.Timedelta(minutes=self.config.window_duration_minutes)
            window_obs = [obs for obs in history if obs.timestamp >= window_cutoff] + [current_obs]

            record = self._build_feature_record(
                patient_id=patient_id,
                current_obs=current_obs,
                window_obs=window_obs,
                history=history,
                patient_baseline=patient_baseline,
            )
            records.append(record)
            history.append(current_obs)

        return records

    def _build_feature_record(
        self,
        patient_id: str,
        current_obs: Observation,
        window_obs: List[Observation],
        history: List[Observation],
        patient_baseline: Optional[PatientBaseline],
    ) -> FeatureRecord:
        """Construct the evaluated feature record for the current observation and window."""
        flags: List[str] = []
        quality_state = DataQualityState.VALID
        is_eligible = True

        # Missingness checks on current observation
        missing_count = sum(1 for m in [current_obs.heart_rate, current_obs.spo2, current_obs.temperature] if m is None)
        if missing_count == 3:
            quality_state = DataQualityState.EXCESSIVE_MISSINGNESS
            is_eligible = False
            flags.append("ALL_METRICS_MISSING_IN_CURRENT_OBSERVATION")
        elif missing_count > 0:
            flags.append(f"PARTIAL_MISSINGNESS_COUNT_{missing_count}")

        # Elapsed time from previous observation
        elapsed_minutes = 0.0
        elapsed_hours = 0.0
        prev_obs: Optional[Observation] = history[-1] if history else None
        gap_exceeded = False

        if prev_obs is not None:
            delta_sec = (current_obs.timestamp - prev_obs.timestamp).total_seconds()
            if delta_sec < 0:
                flags.append("TIME_TRAVEL_OUT_OF_ORDER")
                quality_state = DataQualityState.CORRUPTED_MEASUREMENT
                is_eligible = False
            else:
                elapsed_minutes = delta_sec / 60.0
                elapsed_hours = delta_sec / 3600.0
                if elapsed_minutes > self.config.max_gap_minutes:
                    gap_exceeded = True
                    flags.append(f"GAP_EXCEEDED_{elapsed_minutes:.1f}M")
                    # Do not link slope across large multi-hour disconnect
                    prev_obs = None
                    elapsed_hours = None

        # Check baseline eligibility
        has_baseline = patient_baseline is not None
        if not has_baseline:
            flags.append("INSUFFICIENT_HISTORY_FOR_BASELINE")
            if quality_state == DataQualityState.VALID:
                quality_state = DataQualityState.INSUFFICIENT_HISTORY

        # Gather values in window
        hr_window = [o.heart_rate for o in window_obs if o.heart_rate is not None]
        spo2_window = [o.spo2 for o in window_obs if o.spo2 is not None]
        temp_window = [o.temperature for o in window_obs if o.temperature is not None]

        hr_prev = prev_obs.heart_rate if prev_obs else None
        spo2_prev = prev_obs.spo2 if prev_obs else None
        temp_prev = prev_obs.temperature if prev_obs else None

        hr_base = patient_baseline.hr if patient_baseline else None
        spo2_base = patient_baseline.spo2 if patient_baseline else None
        temp_base = patient_baseline.temp if patient_baseline else None

        # Compute metric features
        hr_stats = compute_window_statistics(
            hr_window, hr_prev, elapsed_hours, hr_base, self.config.epsilon, self.config.mad_normal_factor
        )
        spo2_stats = compute_window_statistics(
            spo2_window, spo2_prev, elapsed_hours, spo2_base, self.config.epsilon, self.config.mad_normal_factor
        )
        temp_stats = compute_window_statistics(
            temp_window, temp_prev, elapsed_hours, temp_base, self.config.epsilon, self.config.mad_normal_factor
        )

        # Cross-metric features
        hr_lat = hr_stats["latest"]
        spo2_lat = spo2_stats["latest"]
        temp_lat = temp_stats["latest"]

        if hr_lat > 0 and spo2_lat > self.config.epsilon:
            cross_htr = float(hr_lat / spo2_lat)
        else:
            cross_htr = 0.0

        # Vital strain proxy: relative deviation across vitals
        # Reference centering: HR ~ 80 bpm, Temp ~ 98.6 F (or 37.0 C), SpO2 ~ 97 %
        temp_center = 98.6 if self.config.temperature_unit == "F" else 37.0
        temp_scale = 1.0 if self.config.temperature_unit == "F" else 0.55
        if hr_lat > 0 and spo2_lat > 0 and temp_lat > 0:
            cross_strain = float(
                ((hr_lat - 80.0) / 15.0)
                + ((temp_lat - temp_center) / temp_scale)
                - ((spo2_lat - 97.0) / 2.5)
            )
        else:
            cross_strain = 0.0

        # Assemble full feature map
        feat_map: Dict[str, float] = {
            # Heart Rate
            "hr_latest": hr_stats["latest"],
            "hr_rolling_mean": hr_stats["rolling_mean"],
            "hr_rolling_median": hr_stats["rolling_median"],
            "hr_rolling_std": hr_stats["rolling_std"],
            "hr_rolling_mad": hr_stats["rolling_mad"],
            "hr_window_min": hr_stats["window_min"],
            "hr_window_max": hr_stats["window_max"],
            "hr_abs_change": hr_stats["abs_change"],
            "hr_rel_change": hr_stats["rel_change"],
            "hr_trend_slope": hr_stats["trend_slope"],
            "hr_baseline_dev": hr_stats["baseline_dev"],
            "hr_norm_baseline_dev": hr_stats["norm_baseline_dev"],
            # SpO2
            "spo2_latest": spo2_stats["latest"],
            "spo2_rolling_mean": spo2_stats["rolling_mean"],
            "spo2_rolling_median": spo2_stats["rolling_median"],
            "spo2_rolling_std": spo2_stats["rolling_std"],
            "spo2_rolling_mad": spo2_stats["rolling_mad"],
            "spo2_window_min": spo2_stats["window_min"],
            "spo2_window_max": spo2_stats["window_max"],
            "spo2_abs_change": spo2_stats["abs_change"],
            "spo2_rel_change": spo2_stats["rel_change"],
            "spo2_trend_slope": spo2_stats["trend_slope"],
            "spo2_baseline_dev": spo2_stats["baseline_dev"],
            "spo2_norm_baseline_dev": spo2_stats["norm_baseline_dev"],
            # Temperature
            "temp_latest": temp_stats["latest"],
            "temp_rolling_mean": temp_stats["rolling_mean"],
            "temp_rolling_median": temp_stats["rolling_median"],
            "temp_rolling_std": temp_stats["rolling_std"],
            "temp_rolling_mad": temp_stats["rolling_mad"],
            "temp_window_min": temp_stats["window_min"],
            "temp_window_max": temp_stats["window_max"],
            "temp_abs_change": temp_stats["abs_change"],
            "temp_rel_change": temp_stats["rel_change"],
            "temp_trend_slope": temp_stats["trend_slope"],
            "temp_baseline_dev": temp_stats["baseline_dev"],
            "temp_norm_baseline_dev": temp_stats["norm_baseline_dev"],
            # Cross-metric
            "cross_hypoxia_tachycardia_ratio": cross_htr,
            "cross_vital_strain_index": cross_strain,
            # Data Quality
            "quality_sample_count": float(len(window_obs)),
            "quality_missing_count": float(missing_count),
            "quality_elapsed_minutes": float(elapsed_minutes),
            "quality_gap_exceeded": 1.0 if gap_exceeded else 0.0,
            "quality_has_baseline": 1.0 if has_baseline else 0.0,
        }

        return FeatureRecord(
            patient_id=patient_id,
            timestamp=current_obs.raw_timestamp_str,
            schema_version=FEATURE_SCHEMA_VERSION,
            quality_state=quality_state,
            is_eligible_for_scoring=is_eligible,
            features=feat_map,
            raw_measurements={
                "heart_rate": current_obs.heart_rate,
                "spo2": current_obs.spo2,
                "temperature": current_obs.temperature,
            },
            quality_flags=flags,
        )


class StreamingFeatureExtractor:
    """
    Stateful streaming feature extractor for live telemetry (e.g. from simulator or WebSocket).
    Maintains an in-memory sliding buffer per patient to extract features in real-time
    strictly using observations up to the current assessment time (zero lookahead).
    """

    def __init__(
        self,
        config: Optional[FeatureExtractorConfig] = None,
        baseline_manager: Optional[BaselineManager] = None,
        max_buffer_size: int = 100,
    ):
        self.config = config or FeatureExtractorConfig()
        self.baseline_manager = baseline_manager
        self.max_buffer_size = max_buffer_size
        self.buffers: Dict[str, deque] = {}
        self._extractor = FeatureExtractor(config=self.config, baseline_manager=self.baseline_manager)

    def process_reading(
        self,
        patient_id: str,
        timestamp: Union[str, pd.Timestamp],
        heart_rate: Optional[float],
        spo2: Optional[float],
        temperature_f: Optional[float],
    ) -> FeatureRecord:
        """
        Process a single incoming telemetry reading in real-time.
        """
        pid = str(patient_id)
        if pid not in self.buffers:
            self.buffers[pid] = deque(maxlen=self.max_buffer_size)

        ts_parsed = pd.to_datetime(timestamp, errors="coerce")
        raw_ts_str = str(timestamp)

        if pd.isnull(ts_parsed):
            return FeatureRecord(
                patient_id=pid,
                timestamp=raw_ts_str,
                quality_state=DataQualityState.CORRUPTED_MEASUREMENT,
                is_eligible_for_scoring=False,
                quality_flags=["INVALID_TIMESTAMP"],
            )

        hr_clean = float(heart_rate) if heart_rate is not None and not np.isnan(heart_rate) and not np.isinf(heart_rate) else None
        spo2_clean = float(spo2) if spo2 is not None and not np.isnan(spo2) and not np.isinf(spo2) else None
        temp_clean = None
        if temperature_f is not None and not np.isnan(temperature_f) and not np.isinf(temperature_f):
            temp_clean = normalize_temperature(
                float(temperature_f), source_unit="F", target_unit=self.config.temperature_unit
            )

        current_obs = Observation(
            timestamp=ts_parsed,
            heart_rate=hr_clean,
            spo2=spo2_clean,
            temperature=temp_clean,
            raw_timestamp_str=raw_ts_str,
        )

        history = list(self.buffers[pid])
        window_cutoff = ts_parsed - pd.Timedelta(minutes=self.config.window_duration_minutes)
        window_obs = [obs for obs in history if obs.timestamp >= window_cutoff] + [current_obs]

        patient_baseline = (
            self.baseline_manager.get_patient_baseline(pid)
            if self.baseline_manager is not None
            else None
        )

        record = self._extractor._build_feature_record(
            patient_id=pid,
            current_obs=current_obs,
            window_obs=window_obs,
            history=history,
            patient_baseline=patient_baseline,
        )

        # Update in-memory sliding buffer
        self.buffers[pid].append(current_obs)

        return record

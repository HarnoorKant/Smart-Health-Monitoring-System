"""
Patient baseline calculation with strict leakage prevention and fit/transform isolation.
Member 3: ML Analysis, Integration, QA and Deployment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from health_analysis.units import normalize_temperature


@dataclass
class MetricBaseline:
    """Baseline statistics for a single physiological metric."""
    sample_count: int
    median: float
    mad: float
    mean: float
    std: float
    min: float
    max: float

    def compute_deviation(
        self,
        value: Optional[float],
        epsilon: float = 1e-6,
        mad_normal_factor: float = 1.4826,
    ) -> Tuple[float, float]:
        """
        Compute (raw_deviation, normalized_robust_deviation) from baseline.
        Handles zero variability and near-zero MAD safely.
        """
        if value is None or np.isnan(value) or np.isinf(value):
            return 0.0, 0.0

        raw_dev = float(value - self.median)

        # Robust standard deviation proxy: 1.4826 * MAD
        robust_scale = self.mad * mad_normal_factor
        if robust_scale > epsilon:
            norm_dev = raw_dev / robust_scale
        elif self.std > epsilon:
            # Fallback to standard deviation if MAD is near zero
            norm_dev = raw_dev / self.std
        else:
            # Zero-variability baseline (e.g. all baseline readings identical)
            norm_dev = 0.0 if abs(raw_dev) <= epsilon else (1.0 if raw_dev > 0 else -1.0)

        return raw_dev, norm_dev


@dataclass
class PatientBaseline:
    """Established baseline statistics for a specific patient."""
    patient_id: str
    sample_count: int
    hr: MetricBaseline
    spo2: MetricBaseline
    temp: MetricBaseline
    created_at_observation_count: int = 0


def calculate_metric_baseline(values: Union[List[float], np.ndarray]) -> MetricBaseline:
    """
    Calculate summary baseline statistics for a clean numeric array.
    """
    clean = np.asarray([v for v in values if v is not None and not np.isnan(v) and not np.isinf(v)], dtype=float)
    count = len(clean)
    if count == 0:
        return MetricBaseline(sample_count=0, median=0.0, mad=0.0, mean=0.0, std=0.0, min=0.0, max=0.0)

    med = float(np.median(clean))
    abs_deviations = np.abs(clean - med)
    mad = float(np.median(abs_deviations))
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1)) if count > 1 else 0.0
    min_val = float(np.min(clean))
    max_val = float(np.max(clean))

    return MetricBaseline(
        sample_count=count,
        median=med,
        mad=mad,
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
    )


class BaselineManager:
    """
    Manages fit/transform patient baseline calculations to prevent data leakage.
    During training, baselines are 'fit' on historical data and frozen.
    During evaluation or streaming inference, frozen baselines are queried.
    """

    def __init__(
        self,
        min_baseline_samples: int = 5,
        temperature_unit: str = "F",
        epsilon: float = 1e-6,
        mad_normal_factor: float = 1.4826,
    ):
        self.min_baseline_samples = min_baseline_samples
        self.temperature_unit = temperature_unit
        self.epsilon = epsilon
        self.mad_normal_factor = mad_normal_factor
        self.baselines: Dict[str, PatientBaseline] = {}
        self.is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame) -> "BaselineManager":
        """
        Fit patient baselines strictly using training data.
        Does NOT look at evaluation data.
        """
        required_cols = {"patient_id", "heart_rate", "spo2", "temperature_f"}
        missing = required_cols - set(df_train.columns)
        if missing:
            raise ValueError(f"Training DataFrame missing required columns: {missing}")

        self.baselines.clear()

        # Group by patient_id
        for pid, group in df_train.groupby("patient_id"):
            pid_str = str(pid)
            if len(group) < self.min_baseline_samples:
                continue

            hr_vals = group["heart_rate"].dropna().tolist()
            spo2_vals = group["spo2"].dropna().tolist()

            # Normalize temperature if required
            temp_raw = group["temperature_f"].dropna().tolist()
            temp_vals = [
                normalize_temperature(t, source_unit="F", target_unit=self.temperature_unit)
                for t in temp_raw
            ]

            hr_base = calculate_metric_baseline(hr_vals)
            spo2_base = calculate_metric_baseline(spo2_vals)
            temp_base = calculate_metric_baseline(temp_vals)

            if min(hr_base.sample_count, spo2_base.sample_count, temp_base.sample_count) >= self.min_baseline_samples:
                self.baselines[pid_str] = PatientBaseline(
                    patient_id=pid_str,
                    sample_count=len(group),
                    hr=hr_base,
                    spo2=spo2_base,
                    temp=temp_base,
                    created_at_observation_count=len(group),
                )

        self.is_fitted = True
        return self

    def get_patient_baseline(self, patient_id: str) -> Optional[PatientBaseline]:
        """Return the frozen baseline for a patient if established."""
        return self.baselines.get(str(patient_id))

    def has_baseline(self, patient_id: str) -> bool:
        """Check if a patient has an established baseline."""
        return str(patient_id) in self.baselines


class IncrementalBaselineTracker:
    """
    Sequential/streaming baseline tracker for live simulator observations.
    Calculates patient baselines on-the-fly strictly from observations
    seen up to the current assessment time, preventing lookahead leakage.
    """

    def __init__(
        self,
        min_baseline_samples: int = 5,
        temperature_unit: str = "F",
        epsilon: float = 1e-6,
        mad_normal_factor: float = 1.4826,
    ):
        self.min_baseline_samples = min_baseline_samples
        self.temperature_unit = temperature_unit
        self.epsilon = epsilon
        self.mad_normal_factor = mad_normal_factor
        self.history: Dict[str, Dict[str, List[float]]] = {}

    def observe(self, patient_id: str, heart_rate: float, spo2: float, temperature: float) -> None:
        """Register a valid historical observation for a patient."""
        pid = str(patient_id)
        if pid not in self.history:
            self.history[pid] = {"hr": [], "spo2": [], "temp": []}

        if heart_rate is not None and not np.isnan(heart_rate) and not np.isinf(heart_rate):
            self.history[pid]["hr"].append(float(heart_rate))
        if spo2 is not None and not np.isnan(spo2) and not np.isinf(spo2):
            self.history[pid]["spo2"].append(float(spo2))
        if temperature is not None and not np.isnan(temperature) and not np.isinf(temperature):
            temp_norm = normalize_temperature(temperature, source_unit="F", target_unit=self.temperature_unit)
            self.history[pid]["temp"].append(float(temp_norm))

    def get_baseline(self, patient_id: str) -> Optional[PatientBaseline]:
        """
        Return the patient's baseline computed strictly from history observed so far.
        Returns None if history < min_baseline_samples.
        """
        pid = str(patient_id)
        if pid not in self.history:
            return None

        h = self.history[pid]
        if min(len(h["hr"]), len(h["spo2"]), len(h["temp"])) < self.min_baseline_samples:
            return None

        return PatientBaseline(
            patient_id=pid,
            sample_count=len(h["hr"]),
            hr=calculate_metric_baseline(h["hr"]),
            spo2=calculate_metric_baseline(h["spo2"]),
            temp=calculate_metric_baseline(h["temp"]),
            created_at_observation_count=len(h["hr"]),
        )

"""
Stable feature schema, configuration, and quality state definitions.
Member 3: ML Analysis, Integration, QA and Deployment.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


FEATURE_SCHEMA_VERSION: str = "1.0.0"

# Ordered list of core physiological features (38 features)
PHYSIOLOGICAL_FEATURE_NAMES_V1: List[str] = [
    # Heart Rate (12 features)
    "hr_latest",
    "hr_rolling_mean",
    "hr_rolling_median",
    "hr_rolling_std",
    "hr_rolling_mad",
    "hr_window_min",
    "hr_window_max",
    "hr_abs_change",
    "hr_rel_change",
    "hr_trend_slope",
    "hr_baseline_dev",
    "hr_norm_baseline_dev",
    # SpO2 (12 features)
    "spo2_latest",
    "spo2_rolling_mean",
    "spo2_rolling_median",
    "spo2_rolling_std",
    "spo2_rolling_mad",
    "spo2_window_min",
    "spo2_window_max",
    "spo2_abs_change",
    "spo2_rel_change",
    "spo2_trend_slope",
    "spo2_baseline_dev",
    "spo2_norm_baseline_dev",
    # Temperature (12 features)
    "temp_latest",
    "temp_rolling_mean",
    "temp_rolling_median",
    "temp_rolling_std",
    "temp_rolling_mad",
    "temp_window_min",
    "temp_window_max",
    "temp_abs_change",
    "temp_rel_change",
    "temp_trend_slope",
    "temp_baseline_dev",
    "temp_norm_baseline_dev",
    # Cross-Metric Interactions (2 features)
    "cross_hypoxia_tachycardia_ratio",
    "cross_vital_strain_index",
]

# Ordered list of data quality & context features (5 features)
DATA_QUALITY_FEATURE_NAMES_V1: List[str] = [
    "quality_sample_count",
    "quality_missing_count",
    "quality_elapsed_minutes",
    "quality_gap_exceeded",
    "quality_has_baseline",
]

# Complete ordered feature vector (43 features)
ALL_FEATURE_NAMES_V1: List[str] = PHYSIOLOGICAL_FEATURE_NAMES_V1 + DATA_QUALITY_FEATURE_NAMES_V1


class DataQualityState(str, Enum):
    """Data quality and eligibility status for a given window."""
    VALID = "VALID"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    GAP_EXCEEDED = "GAP_EXCEEDED"
    EXCESSIVE_MISSINGNESS = "EXCESSIVE_MISSINGNESS"
    CORRUPTED_MEASUREMENT = "CORRUPTED_MEASUREMENT"


@dataclass
class FeatureExtractorConfig:
    """
    Configuration parameters for feature extraction and window construction.
    """
    # Time window duration in minutes (default 120m / 2 hours)
    window_duration_minutes: float = 120.0
    # Minimum observations required in window to compute rolling statistics
    min_window_samples: int = 1
    # Minimum historical observations required to establish patient baseline
    min_baseline_samples: int = 5
    # Maximum allowable gap (in minutes) between consecutive readings before flagging GAP_EXCEEDED
    max_gap_minutes: float = 180.0  # 3 hours (allows normal intermittent monitoring, flags prolonged dropouts)
    # Unit for temperature feature values ('F' or 'C')
    temperature_unit: str = "F"
    # Epsilon to prevent division by zero in relative change and normalization
    epsilon: float = 1e-6
    # Normal consistency factor for MAD to approximate standard deviation
    mad_normal_factor: float = 1.4826


@dataclass
class FeatureRecord:
    """
    Evaluated feature record for a single observation window.
    """
    patient_id: str
    timestamp: str
    schema_version: str = FEATURE_SCHEMA_VERSION
    quality_state: DataQualityState = DataQualityState.VALID
    is_eligible_for_scoring: bool = True
    # Named dictionary of ordered features
    features: Dict[str, float] = field(default_factory=dict)
    # Raw input values for traceability
    raw_measurements: Dict[str, Optional[float]] = field(default_factory=dict)
    # Quality diagnostic notes
    quality_flags: List[str] = field(default_factory=list)

    def to_feature_vector(self, feature_names: Optional[List[str]] = None) -> List[float]:
        """Return the ordered float vector according to the requested feature list."""
        names = feature_names if feature_names is not None else ALL_FEATURE_NAMES_V1
        return [float(self.features.get(name, 0.0)) for name in names]

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a flat dictionary suitable for DataFrame or serialization."""
        res = {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "quality_state": self.quality_state.value,
            "is_eligible_for_scoring": self.is_eligible_for_scoring,
            "quality_flags": ";".join(self.quality_flags),
        }
        res.update(self.features)
        return res

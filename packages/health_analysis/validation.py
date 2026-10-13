"""
Validation and data auditing utilities for the Smart Health Monitoring System.
Member 3: ML Analysis, Integration, QA and Deployment.
"""

from dataclasses import dataclass, field
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

EXPECTED_COLUMNS: List[str] = [
    "patient_id",
    "heart_rate",
    "spo2",
    "temperature_f",
    "timestamp",
]

NUMERIC_COLUMNS: List[str] = [
    "heart_rate",
    "spo2",
    "temperature_f",
]

FORBIDDEN_HISTORICAL_COLUMNS: List[str] = [
    "respiratory_rate",
    "respiratory_rate_bpm",
    "systolic_bp",
    "diastolic_bp",
    "blood_pressure",
    "label",
    "is_anomaly",
    "anomaly",
    "diagnosis",
]


@dataclass
class MetricStatistics:
    min: float
    max: float
    mean: float
    median: float
    std: float
    non_numeric_count: int
    inf_count: int
    null_count: int


@dataclass
class PatientTemporalProfile:
    patient_id: str
    observation_count: int
    first_timestamp: str
    last_timestamp: str
    is_chronological: bool
    interval_min_minutes: Optional[float]
    interval_max_minutes: Optional[float]
    interval_median_minutes: Optional[float]
    interval_mean_minutes: Optional[float]
    gaps_greater_than_2h: int


@dataclass
class DatasetProfileResult:
    filepath: str
    filename: str
    file_size_bytes: int
    sha256_hash: str
    row_count: int
    column_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values_per_column: Dict[str, int]
    full_duplicate_rows: int
    duplicate_patient_timestamp_pairs: int
    invalid_timestamp_count: int
    unique_patients: List[str]
    patient_observation_counts: Dict[str, int]
    metric_statistics: Dict[str, MetricStatistics]
    temporal_profiles: Dict[str, PatientTemporalProfile]
    is_overall_chronological: bool
    has_diagnostic_labels: bool
    has_anomaly_labels: bool
    has_respiratory_rate: bool
    has_blood_pressure: bool
    data_quality_flags: List[str] = field(default_factory=list)


def compute_sha256(filepath: str) -> str:
    """Compute the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_dataset_file(filepath: str) -> DatasetProfileResult:
    """
    Perform a complete audit of the given CSV file according to project ML data standards.
    Does NOT modify the file, alter rows, or drop data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    file_size = os.path.getsize(filepath)
    file_hash = compute_sha256(filepath)
    filename = os.path.basename(filepath)

    df = pd.read_csv(filepath)

    return audit_dataframe(df, filepath=filepath, filename=filename, file_size=file_size, file_hash=file_hash)


def audit_dataframe(
    df: pd.DataFrame,
    filepath: str = "<in-memory>",
    filename: str = "<in-memory>",
    file_size: int = 0,
    file_hash: str = "",
) -> DatasetProfileResult:
    """
    Audit an in-memory DataFrame against schema and data quality rules.
    """
    columns = list(df.columns)
    row_count = len(df)
    col_count = len(columns)
    dtypes = {c: str(df[c].dtype) for c in columns}

    # Missing values
    missing_vals = {c: int(df[c].isnull().sum()) for c in columns}

    # Duplicates
    full_dupes = int(df.duplicated().sum())

    # Key duplicates
    if "patient_id" in df.columns and "timestamp" in df.columns:
        key_dupes = int(df.duplicated(subset=["patient_id", "timestamp"]).sum())
    else:
        key_dupes = 0

    # Presence of unauthorized / absent fields
    lower_cols = [c.lower() for c in columns]
    has_diag = any("diag" in c for c in lower_cols)
    has_anom = any("anom" in c or "label" in c for c in lower_cols)
    has_resp = any("resp" in c for c in lower_cols)
    has_bp = any("bp" in c or "blood_pressure" in c or "systolic" in c for c in lower_cols)

    flags: List[str] = []

    # Check for forbidden columns
    for fc in FORBIDDEN_HISTORICAL_COLUMNS:
        if fc in lower_cols:
            flags.append(f"UNEXPECTED_COLUMN_PRESENT: {fc}")

    # Check for expected columns
    for ec in EXPECTED_COLUMNS:
        if ec not in columns:
            flags.append(f"MISSING_EXPECTED_COLUMN: {ec}")

    # Metric stats
    metric_stats: Dict[str, MetricStatistics] = {}
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            non_numeric = int(series.isnull().sum() - df[col].isnull().sum())
            infs = int(np.isinf(series).sum())
            null_count = int(df[col].isnull().sum())

            valid_values = series.dropna()
            if len(valid_values) > 0:
                stats = MetricStatistics(
                    min=float(valid_values.min()),
                    max=float(valid_values.max()),
                    mean=float(valid_values.mean()),
                    median=float(valid_values.median()),
                    std=float(valid_values.std()),
                    non_numeric_count=non_numeric,
                    inf_count=infs,
                    null_count=null_count,
                )
            else:
                stats = MetricStatistics(
                    min=0.0, max=0.0, mean=0.0, median=0.0, std=0.0,
                    non_numeric_count=non_numeric, inf_count=infs, null_count=null_count
                )
            metric_stats[col] = stats

            if non_numeric > 0:
                flags.append(f"NON_NUMERIC_VALUES_IN_{col.upper()}: {non_numeric}")
            if infs > 0:
                flags.append(f"INFINITE_VALUES_IN_{col.upper()}: {infs}")
            if null_count > 0:
                flags.append(f"MISSING_VALUES_IN_{col.upper()}: {null_count}")

    # Timestamp checks
    invalid_ts = 0
    is_overall_chronological = True
    parsed_timestamps = None
    if "timestamp" in df.columns:
        parsed_timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        invalid_ts = int(parsed_timestamps.isnull().sum())
        if invalid_ts > 0:
            flags.append(f"INVALID_TIMESTAMPS: {invalid_ts}")
        else:
            is_overall_chronological = bool(parsed_timestamps.is_monotonic_increasing)

    # Patients and temporal profiles
    unique_patients: List[str] = []
    patient_counts: Dict[str, int] = {}
    temporal_profiles: Dict[str, PatientTemporalProfile] = {}

    if "patient_id" in df.columns and "timestamp" in df.columns and parsed_timestamps is not None:
        unique_patients = sorted(df["patient_id"].astype(str).unique().tolist())
        patient_counts = {p: int((df["patient_id"] == p).sum()) for p in unique_patients}

        for pid in unique_patients:
            mask = df["patient_id"] == pid
            p_series = df.loc[mask, "timestamp"]
            p_parsed = parsed_timestamps.loc[mask]

            is_chrono = bool(p_parsed.is_monotonic_increasing)
            obs_count = len(p_series)

            diffs = p_parsed.diff()
            diff_mins = diffs.dt.total_seconds() / 60.0
            valid_diffs = diff_mins.dropna()

            gaps_2h = int((valid_diffs > 120.0).sum()) if len(valid_diffs) > 0 else 0

            profile = PatientTemporalProfile(
                patient_id=pid,
                observation_count=obs_count,
                first_timestamp=str(p_series.iloc[0]) if obs_count > 0 else "",
                last_timestamp=str(p_series.iloc[-1]) if obs_count > 0 else "",
                is_chronological=is_chrono,
                interval_min_minutes=float(valid_diffs.min()) if len(valid_diffs) > 0 else None,
                interval_max_minutes=float(valid_diffs.max()) if len(valid_diffs) > 0 else None,
                interval_median_minutes=float(valid_diffs.median()) if len(valid_diffs) > 0 else None,
                interval_mean_minutes=float(valid_diffs.mean()) if len(valid_diffs) > 0 else None,
                gaps_greater_than_2h=gaps_2h,
            )
            temporal_profiles[pid] = profile

            if not is_chrono:
                flags.append(f"NON_CHRONOLOGICAL_DATA_FOR_PATIENT_{pid}")

    if full_dupes > 0:
        flags.append(f"FULL_DUPLICATE_ROWS: {full_dupes}")
    if key_dupes > 0:
        flags.append(f"DUPLICATE_PATIENT_TIMESTAMP_PAIRS: {key_dupes}")

    return DatasetProfileResult(
        filepath=filepath,
        filename=filename,
        file_size_bytes=file_size,
        sha256_hash=file_hash,
        row_count=row_count,
        column_count=col_count,
        columns=columns,
        dtypes=dtypes,
        missing_values_per_column=missing_vals,
        full_duplicate_rows=full_dupes,
        duplicate_patient_timestamp_pairs=key_dupes,
        invalid_timestamp_count=invalid_ts,
        unique_patients=unique_patients,
        patient_observation_counts=patient_counts,
        metric_statistics=metric_stats,
        temporal_profiles=temporal_profiles,
        is_overall_chronological=is_overall_chronological,
        has_diagnostic_labels=has_diag,
        has_anomaly_labels=has_anom,
        has_respiratory_rate=has_resp,
        has_blood_pressure=has_bp,
        data_quality_flags=flags,
    )

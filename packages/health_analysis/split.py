"""
Dataset partitioning strategies with strict temporal isolation and leakage prevention.
Member 3: ML Analysis, Integration, QA and Deployment.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class SplitMetadata:
    """Metadata detailing the exact partition boundaries."""
    strategy: str
    train_rows: int
    eval_rows: int
    train_ratio: float
    train_patient_ids: List[str]
    eval_patient_ids: List[str]
    train_time_range: Dict[str, Tuple[str, str]]
    eval_time_range: Dict[str, Tuple[str, str]]


def chronological_patient_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
) -> Tuple[pd.DataFrame, pd.DataFrame, SplitMetadata]:
    """
    Partition telemetry data chronologically within each patient cohort.
    For each patient, the first train_ratio (default 70%) of observations
    forms the training set; the remaining 30% forms the evaluation set.

    Guarantees:
    - Zero temporal lookahead leakage.
    - Each patient is present in both train and eval partitions.
    - Sorting is strictly monotonic increasing by timestamp.
    """
    if not (0.1 <= train_ratio <= 0.9):
        raise ValueError(f"train_ratio must be between 0.1 and 0.9, got {train_ratio}")

    required = {"patient_id", "timestamp", "heart_rate", "spo2", "temperature_f"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    train_dfs: List[pd.DataFrame] = []
    eval_dfs: List[pd.DataFrame] = []

    train_time_ranges: Dict[str, Tuple[str, str]] = {}
    eval_time_ranges: Dict[str, Tuple[str, str]] = {}

    unique_patients = sorted(df["patient_id"].astype(str).unique().tolist())

    for pid in unique_patients:
        p_df = df[df["patient_id"].astype(str) == pid].copy()
        p_df["_parsed_ts"] = pd.to_datetime(p_df["timestamp"], errors="coerce")
        p_df = p_df.sort_values(by="_parsed_ts", ascending=True).reset_index(drop=True)

        n_rows = len(p_df)
        n_train = int(n_rows * train_ratio)

        # Ensure at least 5 training samples for baseline calculation
        if n_train < 5 and n_rows >= 5:
            n_train = 5

        p_train = p_df.iloc[:n_train].copy().drop(columns=["_parsed_ts"])
        p_eval = p_df.iloc[n_train:].copy().drop(columns=["_parsed_ts"])

        train_dfs.append(p_train)
        eval_dfs.append(p_eval)

        train_time_ranges[pid] = (
            str(p_train["timestamp"].iloc[0]),
            str(p_train["timestamp"].iloc[-1]),
        )
        if len(p_eval) > 0:
            eval_time_ranges[pid] = (
                str(p_eval["timestamp"].iloc[0]),
                str(p_eval["timestamp"].iloc[-1]),
            )
        else:
            eval_time_ranges[pid] = ("", "")

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_eval = pd.concat(eval_dfs, ignore_index=True)

    metadata = SplitMetadata(
        strategy="chronological_within_patient",
        train_rows=len(df_train),
        eval_rows=len(df_eval),
        train_ratio=train_ratio,
        train_patient_ids=unique_patients,
        eval_patient_ids=unique_patients,
        train_time_range=train_time_ranges,
        eval_time_range=eval_time_ranges,
    )

    return df_train, df_eval, metadata


def leave_patient_out_split(
    df: pd.DataFrame,
    test_patient_id: str = "P105",
) -> Tuple[pd.DataFrame, pd.DataFrame, SplitMetadata]:
    """
    Partition data by patient ID for cross-subject generalization testing.
    All rows of test_patient_id form the evaluation set; all other patients
    form the training set.
    """
    pid_test = str(test_patient_id)
    df_train = df[df["patient_id"].astype(str) != pid_test].copy()
    df_eval = df[df["patient_id"].astype(str) == pid_test].copy()

    if len(df_eval) == 0:
        raise ValueError(f"Patient {pid_test} not found in dataset")

    metadata = SplitMetadata(
        strategy="leave_patient_out",
        train_rows=len(df_train),
        eval_rows=len(df_eval),
        train_ratio=float(len(df_train) / len(df)),
        train_patient_ids=sorted(df_train["patient_id"].astype(str).unique().tolist()),
        eval_patient_ids=[pid_test],
        train_time_range={
            p: (str(df_train[df_train["patient_id"] == p]["timestamp"].min()),
                str(df_train[df_train["patient_id"] == p]["timestamp"].max()))
            for p in df_train["patient_id"].unique()
        },
        eval_time_range={
            pid_test: (str(df_eval["timestamp"].min()), str(df_eval["timestamp"].max()))
        },
    )

    return df_train, df_eval, metadata

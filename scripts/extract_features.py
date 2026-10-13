#!/usr/bin/env python3
"""
Feature Extraction Script for Smart Health Monitoring System.
Member 3: ML Analysis, Integration, QA and Deployment.

Usage:
    python scripts/extract_features.py [input_csv] [--output-csv path_to_output_csv]
"""

import argparse
import os
import sys
import time
import pandas as pd

# Ensure packages directory is discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "packages"))

from health_analysis.baseline import BaselineManager
from health_analysis.features import FeatureExtractor
from health_analysis.schema import ALL_FEATURE_NAMES_V1, FeatureExtractorConfig


def main():
    parser = argparse.ArgumentParser(description="Extract features from health telemetry CSV.")
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=os.path.join(project_root, "data", "raw", "kaggle_health_dataset.csv"),
        help="Path to input CSV dataset",
    )
    parser.add_argument(
        "--output-csv",
        help="Path to save extracted features as CSV",
    )
    parser.add_argument(
        "--window-minutes",
        type=float,
        default=120.0,
        help="Window duration in minutes (default 120.0)",
    )
    parser.add_argument(
        "--temp-unit",
        default="F",
        choices=["F", "C"],
        help="Target temperature unit (default 'F')",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading telemetry from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows across {df['patient_id'].nunique()} patient cohorts.")

    config = FeatureExtractorConfig(
        window_duration_minutes=args.window_minutes,
        temperature_unit=args.temp_unit,
    )

    t0 = time.time()
    # Fit baselines on historical data
    bm = BaselineManager(
        min_baseline_samples=config.min_baseline_samples,
        temperature_unit=config.temperature_unit,
    ).fit(df)

    extractor = FeatureExtractor(config=config, baseline_manager=bm)
    records = extractor.extract_from_dataframe(df)
    elapsed = time.time() - t0

    print(f"Extracted {len(records)} feature records in {elapsed * 1000:.1f}ms "
          f"({elapsed * 1000 / len(records):.2f}ms/record).")

    # Summarize quality states
    q_counts = pd.Series([r.quality_state.value for r in records]).value_counts().to_dict()
    print("Quality State Distribution:", q_counts)

    if args.output_csv:
        rows = [r.to_dict() for r in records]
        out_df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        out_df.to_csv(args.output_csv, index=False)
        print(f"Saved feature dataset with {len(out_df.columns)} columns to: {args.output_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Dataset Audit Script for Smart Health Monitoring System.
Member 3: ML Analysis, Integration, QA and Deployment.

Usage:
    python scripts/audit_dataset.py [path_to_csv] [--json-out path_to_output_json]
"""

import argparse
from dataclasses import asdict
import json
import os
import sys

# Ensure packages are discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "packages"))

from health_analysis.validation import audit_dataset_file


def format_report(result) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("SMART HEALTH MONITORING SYSTEM — DATASET AUDIT REPORT")
    lines.append("Member 3: ML Analysis, Integration, QA and Deployment")
    lines.append("=" * 78)
    lines.append(f"File Path:         {result.filepath}")
    lines.append(f"Filename:          {result.filename}")
    lines.append(f"File Size:         {result.file_size_bytes} bytes ({result.file_size_bytes / 1024:.2f} KB)")
    lines.append(f"SHA-256 Checksum:  {result.sha256_hash}")
    lines.append(f"Total Rows:        {result.row_count}")
    lines.append(f"Total Columns:     {result.column_count}")
    lines.append(f"Columns:           {', '.join(result.columns)}")
    lines.append("-" * 78)

    lines.append("COLUMN DATA TYPES & NULL COUNTS:")
    for col in result.columns:
        dt = result.dtypes.get(col, "unknown")
        nulls = result.missing_values_per_column.get(col, 0)
        lines.append(f"  - {col:18s}: dtype={dt:10s} null_count={nulls}")
    lines.append("-" * 78)

    lines.append("INTEGRITY & DUPLICATION CHECKS:")
    lines.append(f"  - Full Duplicate Rows:                 {result.full_duplicate_rows}")
    lines.append(f"  - Duplicate (patient_id, timestamp):   {result.duplicate_patient_timestamp_pairs}")
    lines.append(f"  - Malformed / Invalid Timestamps:      {result.invalid_timestamp_count}")
    lines.append(f"  - Overall Chronologically Ordered:     {result.is_overall_chronological}")
    lines.append("-" * 78)

    lines.append("SCOPE & PROVENANCE BOUNDARY CHECKS:")
    lines.append(f"  - Diagnostic Labels Present:           {result.has_diagnostic_labels}")
    lines.append(f"  - Anomaly Labels Present:              {result.has_anomaly_labels}")
    lines.append(f"  - Respiratory Rate Present:            {result.has_respiratory_rate}")
    lines.append(f"  - Blood Pressure Present:              {result.has_blood_pressure}")
    lines.append("-" * 78)

    lines.append("MEASUREMENT SUMMARY STATISTICS:")
    for metric, stats in result.metric_statistics.items():
        lines.append(f"  [{metric.upper()}]")
        lines.append(f"    Min:        {stats.min:.2f}")
        lines.append(f"    Max:        {stats.max:.2f}")
        lines.append(f"    Mean:       {stats.mean:.2f}")
        lines.append(f"    Median:     {stats.median:.2f}")
        lines.append(f"    Std Dev:    {stats.std:.2f}")
        lines.append(f"    Non-Numeric:{stats.non_numeric_count}")
        lines.append(f"    Infinities: {stats.inf_count}")
    lines.append("-" * 78)

    lines.append("PATIENT COHORT & TEMPORAL PROFILES:")
    lines.append(f"  Unique Patients ({len(result.unique_patients)}): {', '.join(result.unique_patients)}")
    for pid, profile in result.temporal_profiles.items():
        lines.append(f"  Patient {pid}:")
        lines.append(f"    Observations:      {profile.observation_count}")
        lines.append(f"    First Timestamp:   {profile.first_timestamp}")
        lines.append(f"    Last Timestamp:    {profile.last_timestamp}")
        lines.append(f"    Chronological:     {profile.is_chronological}")
        if profile.interval_median_minutes is not None:
            lines.append(f"    Interval (min):    min={profile.interval_min_minutes:.1f}m, "
                         f"median={profile.interval_median_minutes:.1f}m, "
                         f"mean={profile.interval_mean_minutes:.1f}m, "
                         f"max={profile.interval_max_minutes:.1f}m")
            lines.append(f"    Gaps > 2 Hours:    {profile.gaps_greater_than_2h}")
    lines.append("-" * 78)

    lines.append("DATA QUALITY FLAGS / FINDINGS:")
    if result.data_quality_flags:
        for flag in result.data_quality_flags:
            lines.append(f"  [FLAG] {flag}")
    else:
        lines.append("  [INFO] No integrity errors found. Dataset satisfies schema preconditions.")
    lines.append("=" * 78)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit and profile health dataset CSV.")
    parser.add_argument(
        "filepath",
        nargs="?",
        default=os.path.join(project_root, "data", "raw", "kaggle_health_dataset.csv"),
        help="Path to CSV file to audit",
    )
    parser.add_argument("--json-out", help="Optional path to output JSON results")

    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File not found at {args.filepath}", file=sys.stderr)
        sys.exit(1)

    result = audit_dataset_file(args.filepath)
    report_text = format_report(result)
    print(report_text)

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nWrote JSON audit summary to: {args.json_out}")


if __name__ == "__main__":
    main()

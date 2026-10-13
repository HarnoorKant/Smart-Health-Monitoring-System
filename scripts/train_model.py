#!/usr/bin/env python3
"""
Model Training & Evaluation Script for Smart Health Monitoring System.
Member 3: ML Analysis, Integration, QA and Deployment.

Usage:
    python scripts/train_model.py [--input-csv path_to_csv] [--artifact-dir path_to_save_dir]
"""

import argparse
from dataclasses import asdict
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Ensure packages directory is discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "packages"))

from health_analysis.baseline import BaselineManager
from health_analysis.features import FeatureExtractor
from health_analysis.models import IsolationForestDetector, ModelConfig
from health_analysis.schema import (
    PHYSIOLOGICAL_FEATURE_NAMES_V1,
    FeatureExtractorConfig,
)
from health_analysis.split import chronological_patient_split
from health_analysis.validation import audit_dataset_file


def run_training_pipeline(
    input_csv: str,
    artifact_dir: str,
    train_ratio: float = 0.70,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Any]:
    print("=" * 78)
    print("SMART HEALTH MONITORING SYSTEM — MODEL TRAINING & EVALUATION")
    print("Member 3: ML Analysis, Integration, QA and Deployment")
    print("=" * 78)

    # 1. Audit and Load Dataset
    print(f"\n[Step 1] Auditing and Loading Telemetry: {input_csv}")
    audit_res = audit_dataset_file(input_csv)
    print(f"  Rows: {audit_res.row_count}, Patients: {', '.join(audit_res.unique_patients)}")
    print(f"  SHA-256 Checksum: {audit_res.sha256_hash}")
    df_raw = pd.read_csv(input_csv)

    # 2. Chronological Split within Patient Cohorts
    print(f"\n[Step 2] Executing Chronological Split (Train: {train_ratio*100:.0f}%, Eval: {(1-train_ratio)*100:.0f}%)")
    df_train, df_eval, split_meta = chronological_patient_split(df_raw, train_ratio=train_ratio)
    print(f"  Training Rows:   {len(df_train)} across {len(split_meta.train_patient_ids)} patients")
    print(f"  Evaluation Rows: {len(df_eval)} across {len(split_meta.eval_patient_ids)} patients")
    for pid in split_meta.train_patient_ids:
        t_tr = split_meta.train_time_range[pid]
        t_ev = split_meta.eval_time_range[pid]
        print(f"  Patient {pid}: Train [{t_tr[0]} -> {t_tr[1]}] | Eval [{t_ev[0]} -> {t_ev[1]}]")

    # 3. Fit Baselines on Training History Only (Leakage Prevention)
    print("\n[Step 3] Fitting Patient Baselines on Training Partition (Frozen)")
    fe_config = FeatureExtractorConfig(
        window_duration_minutes=120.0,
        min_baseline_samples=5,
        temperature_unit="F",
    )
    bm = BaselineManager(
        min_baseline_samples=fe_config.min_baseline_samples,
        temperature_unit=fe_config.temperature_unit,
    ).fit(df_train)

    for pid in split_meta.train_patient_ids:
        b = bm.get_patient_baseline(pid)
        print(f"  Patient {pid} Baseline: HR Median={b.hr.median:.1f} (MAD={b.hr.mad:.1f}), "
              f"SpO2 Median={b.spo2.median:.1f} (MAD={b.spo2.mad:.1f}), "
              f"Temp Median={b.temp.median:.1f} (MAD={b.temp.mad:.1f})")

    # 4. Extract Feature Records (Zero Lookahead)
    print("\n[Step 4] Extracting Feature Windows using Frozen Baselines")
    extractor = FeatureExtractor(config=fe_config, baseline_manager=bm)
    train_records = extractor.extract_from_dataframe(df_train)
    eval_records = extractor.extract_from_dataframe(df_eval)

    # Convert to feature matrices
    train_features = [r.to_feature_vector(PHYSIOLOGICAL_FEATURE_NAMES_V1) for r in train_records if r.is_eligible_for_scoring]
    eval_features = [r.to_feature_vector(PHYSIOLOGICAL_FEATURE_NAMES_V1) for r in eval_records if r.is_eligible_for_scoring]

    X_train = np.array(train_features, dtype=float)
    X_eval = np.array(eval_features, dtype=float)

    print(f"  Eligible Training Feature Windows:   {len(X_train)} (shape: {X_train.shape})")
    print(f"  Eligible Evaluation Feature Windows: {len(X_eval)} (shape: {X_eval.shape})")

    # 5. Configure & Train Isolation Forest
    print("\n[Step 5] Training Unsupervised Isolation Forest Anomaly Detector")
    model_config = ModelConfig(
        model_version="1.0.0",
        n_estimators=100,
        max_samples=256,
        contamination=contamination,
        max_features=1.0,
        random_state=random_state,
        decision_threshold=0.50,
        score_scaling_temperature=0.15,
        feature_names=list(PHYSIOLOGICAL_FEATURE_NAMES_V1),
    )
    print(f"  Hyperparameters: n_estimators={model_config.n_estimators}, "
          f"max_samples={model_config.max_samples}, "
          f"contamination={model_config.contamination}, "
          f"random_state={model_config.random_state}")

    detector = IsolationForestDetector(config=model_config)
    detector.fit(
        X_train,
        feature_names=model_config.feature_names,
        metadata={
            "dataset_sha256": audit_res.sha256_hash,
            "train_ratio": train_ratio,
            "train_rows": len(df_train),
            "eval_rows": len(df_eval),
        },
    )
    print("  Model fit complete.")

    # 6. Evaluate on Chronological Holdout (Historical Unlabelled Data)
    print("\n[Step 6] Evaluating on Chronological Holdout (Unsupervised Historical Evaluation)")
    eval_scores, eval_decisions = detector.score_samples(X_eval)

    flagged_mask = eval_scores >= model_config.decision_threshold
    flagged_count = int(np.sum(flagged_mask))
    flagged_pct = float(flagged_count / len(eval_scores) * 100.0)

    score_stats = {
        "min": float(np.min(eval_scores)),
        "max": float(np.max(eval_scores)),
        "mean": float(np.mean(eval_scores)),
        "median": float(np.median(eval_scores)),
        "p25": float(np.percentile(eval_scores, 25)),
        "p75": float(np.percentile(eval_scores, 75)),
        "p95": float(np.percentile(eval_scores, 95)),
    }

    print(f"  Evaluation Score Distribution:")
    print(f"    Min:    {score_stats['min']:.4f}")
    print(f"    Mean:   {score_stats['mean']:.4f}")
    print(f"    Median: {score_stats['median']:.4f}")
    print(f"    Max:    {score_stats['max']:.4f}")
    print(f"    p95:    {score_stats['p95']:.4f}")
    print(f"  Flagged Observations (score >= {model_config.decision_threshold}): {flagged_count} / {len(eval_scores)} ({flagged_pct:.2f}%)")

    # Patient-by-patient breakdown on evaluation set
    eval_eligible_records = [r for r in eval_records if r.is_eligible_for_scoring]
    patient_breakdown = {}
    for i, rec in enumerate(eval_eligible_records):
        pid = rec.patient_id
        if pid not in patient_breakdown:
            patient_breakdown[pid] = {"count": 0, "flagged": 0, "scores": []}
        patient_breakdown[pid]["count"] += 1
        patient_breakdown[pid]["scores"].append(float(eval_scores[i]))
        if eval_scores[i] >= model_config.decision_threshold:
            patient_breakdown[pid]["flagged"] += 1

    print("\n  Per-Patient Evaluation Breakdown:")
    for pid, pdata in sorted(patient_breakdown.items()):
        p_mean = np.mean(pdata["scores"])
        p_max = np.max(pdata["scores"])
        p_pct = (pdata["flagged"] / pdata["count"]) * 100.0
        print(f"    Patient {pid}: {pdata['count']} obs | Flagged: {pdata['flagged']} ({p_pct:.1f}%) | Mean Score: {p_mean:.4f} | Max Score: {p_max:.4f}")

    # Top anomalous observations in historical evaluation
    top_indices = np.argsort(eval_scores)[::-1][:3]
    top_historical_examples = []
    print("\n  Top 3 High-Scoring Historical Observations (Unsupervised Candidates):")
    for rank, idx in enumerate(top_indices, 1):
        rec = eval_eligible_records[idx]
        score = eval_scores[idx]
        assessment = detector.assess_record(rec)
        print(f"    #{rank}: Patient {rec.patient_id} at {rec.timestamp} — Anomaly Score: {score:.4f} ({assessment.result_status})")
        print(f"        Raw Vitals: HR={rec.raw_measurements['heart_rate']} bpm, SpO2={rec.raw_measurements['spo2']}%, Temp={rec.raw_measurements['temperature']} F")
        print(f"        Top Contributing Features: {', '.join([f'{f['feature']}={f['value']}' for f in assessment.top_contributing_features])}")
        top_historical_examples.append({
            "rank": rank,
            "patient_id": rec.patient_id,
            "timestamp": rec.timestamp,
            "score": float(score),
            "status": assessment.result_status,
            "raw_measurements": rec.raw_measurements,
            "top_features": assessment.top_contributing_features,
        })

    # 7. Controlled Synthetic Anomaly Injection Benchmark
    print("\n[Step 7] Controlled Synthetic Anomaly Scenarios (Ground-Truth Injected Verification)")
    scenarios = [
        {
            "name": "Normative Physiological Control",
            "patient_id": "P101",
            "hr": 75.0, "spo2": 98.0, "temp": 98.6,
            "expected_anom": False,
        },
        {
            "name": "Acute Tachycardia Spike (HR > 130 bpm)",
            "patient_id": "P101",
            "hr": 135.0, "spo2": 97.0, "temp": 98.6,
            "expected_anom": True,
        },
        {
            "name": "Acute Hypoxemic Desaturation (SpO2 < 87%)",
            "patient_id": "P101",
            "hr": 88.0, "spo2": 85.0, "temp": 98.6,
            "expected_anom": True,
        },
        {
            "name": "Combined Hyperthermic & Tachycardic Stress",
            "patient_id": "P101",
            "hr": 128.0, "spo2": 91.0, "temp": 102.5,
            "expected_anom": True,
        },
    ]

    synthetic_results = []
    from health_analysis.features import StreamingFeatureExtractor
    for sc in scenarios:
        s_extractor = StreamingFeatureExtractor(config=fe_config, baseline_manager=bm)
        # Prime with 5 normal observations for patient
        for i in range(5):
            s_extractor.process_reading(sc["patient_id"], f"2023-10-12 08:0{i}:00", 72.0, 98.0, 98.6)

        rec = s_extractor.process_reading(sc["patient_id"], "2023-10-12 08:30:00", sc["hr"], sc["spo2"], sc["temp"])
        assess = detector.assess_record(rec)
        matches = assess.is_anomaly == sc["expected_anom"]
        status_tag = "FLAGGED" if assess.is_anomaly else "NORMAL_LIKE"
        print(f"  [{status_tag}] {sc['name']}:")
        print(f"      Score: {assess.anomaly_score:.4f} | Flagged (>= {model_config.decision_threshold}): {assess.is_anomaly} | Status: {assess.result_status} | Latency: {assess.latency_ms:.2f}ms")
        print(f"      Top Reasons: {', '.join([f'{f['feature']}={f['value']}' for f in assess.top_contributing_features])}")
        synthetic_results.append({
            "scenario": sc["name"],
            "expected_anomaly": sc["expected_anom"],
            "score": assess.anomaly_score,
            "is_anomaly": assess.is_anomaly,
            "result_status": assess.result_status,
            "matched_expectation": matches,
            "latency_ms": assess.latency_ms,
            "top_features": assess.top_contributing_features,
        })

    # 8. Save Versioned Artifacts
    print(f"\n[Step 8] Saving Versioned Artifact to: {artifact_dir}")
    detector.save(artifact_dir)
    print("  Artifact model.joblib and metadata.json saved successfully.")

    # 9. Compile Full Evaluation Summary
    summary = {
        "model_version": model_config.model_version,
        "evaluation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_checksum": audit_res.sha256_hash,
        "train_rows": len(df_train),
        "eval_rows": len(df_eval),
        "train_ratio": train_ratio,
        "eligible_train_windows": len(X_train),
        "eligible_eval_windows": len(X_eval),
        "hyperparameters": asdict(model_config),
        "eval_score_distribution": score_stats,
        "eval_flagged_count": flagged_count,
        "eval_flagged_percent": flagged_pct,
        "patient_breakdown": {
            pid: {
                "count": pdata["count"],
                "flagged": pdata["flagged"],
                "flagged_percent": round((pdata["flagged"] / pdata["count"]) * 100.0, 2),
                "mean_score": round(float(np.mean(pdata["scores"])), 4),
            } for pid, pdata in patient_breakdown.items()
        },
        "top_historical_examples": top_historical_examples,
        "synthetic_benchmark_results": synthetic_results,
    }

    eval_json_path = os.path.join(project_root, "docs", "ml", "evaluation_results.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Exported full evaluation summary to: {eval_json_path}")
    print("=" * 78)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Isolation Forest anomaly detector.")
    parser.add_argument(
        "--input-csv",
        default=os.path.join(project_root, "data", "raw", "kaggle_health_dataset.csv"),
        help="Path to raw telemetry CSV",
    )
    parser.add_argument(
        "--artifact-dir",
        default=os.path.join(project_root, "models", "iforest_v1"),
        help="Path to directory for saving model artifacts",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Chronological train split ratio (default 0.70)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Isolation Forest contamination factor (default 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42)",
    )

    args = parser.parse_args()
    run_training_pipeline(
        input_csv=args.input_csv,
        artifact_dir=args.artifact_dir,
        train_ratio=args.train_ratio,
        contamination=args.contamination,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()

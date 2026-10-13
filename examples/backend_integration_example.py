#!/usr/bin/env python3
"""
Backend Integration Example for Smart Health Monitoring System.
Member 3 -> Member 1 Interface Demonstration.

Demonstrates how Member 1's FastAPI backend calls the health_analysis package
after validating and persisting an incoming telemetry reading.

Flow:
    1. Ingest & Validate raw HTTP payload (FastAPI / Pydantic).
    2. Persist canonical reading into MySQL InnoDB (authoritative).
    3. Invoke deterministic clinical safety rules (independent of ML).
    4. Invoke HealthAnalysisService ML assessment.
    5. Evaluate alert policy:
       - Trigger immediate alert if clinical rule breached.
       - Trigger anomaly alert if ML assessment score >= 0.50.
    6. Persist assessment into MySQL `model_assessments` table.
    7. Asynchronously mirror reading & assessment to MongoDB Atlas via outbox.
"""

import json
import os
import sys
from datetime import datetime, timezone

# Ensure packages directory is discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "packages"))

from health_analysis import (
    HealthAnalysisService,
    evaluate_safety_rules,
    get_analysis_service,
)
from health_analysis.baseline import BaselineManager
import pandas as pd


def simulate_backend_event_ingestion():
    print("=" * 78)
    print("MEMBER 1 BACKEND INGESTION & MEMBER 3 ML INTEGRATION FLOW")
    print("=" * 78)

    # 1. Initialize analysis service with trained model and historical baselines
    model_dir = os.path.join(project_root, "models", "iforest_v1")
    raw_csv = os.path.join(project_root, "data", "raw", "kaggle_health_dataset.csv")

    df_hist = pd.read_csv(raw_csv)
    bm = BaselineManager().fit(df_hist.iloc[:1048])
    service = HealthAnalysisService(artifact_dir=model_dir, baseline_manager=bm)

    # Prime patient P101 with 5 normal observations to satisfy min baseline history
    print("\n[Step 1] Priming Patient P101 history in streaming buffer...")
    for i in range(5):
        service.assess_telemetry(
            patient_id="P101",
            timestamp=f"2023-10-12 08:0{i}:00",
            heart_rate=72.0,
            spo2=98.0,
            temperature_f=98.6,
            source_event_id=f"evt_prime_{i}",
        )
    print("  Patient P101 primed with 5 readings.")

    # 2. Simulate incoming live event (e.g. from wearable simulator)
    # Notice: simulator provides optional respiratory_rate and blood_pressure!
    simulated_payload = {
        "source_event_id": "evt_sim_98234",
        "patient_id": "P101",
        "timestamp": "2023-10-12 08:30:00",
        "is_simulated": True,
        "measurements": {
            "heart_rate": 128.0,
            "spo2": 91.0,
            "temperature_f": 102.4,
            "respiratory_rate": 24.0,       # Live simulator extension
            "systolic_bp": 140.0,           # Live simulator extension
            "diastolic_bp": 92.0,           # Live simulator extension
        }
    }
    print("\n[Step 2] Received Live Telemetry Event:")
    print(json.dumps(simulated_payload, indent=2))

    # 3. Backend verifies and commits event to MySQL InnoDB
    print("\n[Step 3] Backend Action: Ingestion event persisted to MySQL (Authoritative).")

    # 4. Step A: Deterministic Rule Checks (Guaranteed to execute even if ML fails)
    m = simulated_payload["measurements"]
    rule_findings = evaluate_safety_rules(
        heart_rate=m["heart_rate"],
        spo2=m["spo2"],
        temperature_f=m["temperature_f"],
    )
    print(f"\n[Step 4] Deterministic Rule Evaluation: {len(rule_findings)} rule findings:")
    for rf in rule_findings:
        print(f"  [{rf.severity}] {rf.rule_code}: {rf.message}")

    # 5. Step B: Machine Learning Assessment
    assessment = service.assess_telemetry(
        patient_id=simulated_payload["patient_id"],
        timestamp=simulated_payload["timestamp"],
        heart_rate=m["heart_rate"],
        spo2=m["spo2"],
        temperature_f=m["temperature_f"],
        respiratory_rate=m.get("respiratory_rate"),
        systolic_bp=m.get("systolic_bp"),
        diastolic_bp=m.get("diastolic_bp"),
        source_event_id=simulated_payload["source_event_id"],
        is_simulated=simulated_payload["is_simulated"],
    )

    print("\n[Step 5] Machine Learning Assessment Result (JSON Contract):")
    print(json.dumps(assessment.to_dict(), indent=2))

    # 6. Backend Alert & Outbox Policy
    print("\n[Step 6] Backend Alert Policy Decision:")
    if assessment.result == "anomalous" or any(rf.severity == "CRITICAL" for rf in rule_findings):
        alert_reason = []
        if rule_findings:
            alert_reason.append("; ".join([rf.message for rf in rule_findings]))
        if assessment.result == "anomalous":
            alert_reason.append(f"ML Anomaly Score {assessment.score:.2f} ({'; '.join(assessment.explanation[:2])})")

        print("  --> ACTION: CREATE PERSISTENT ALERT IN MYSQL")
        print(f"      Severity: {'CRITICAL' if any(rf.severity == 'CRITICAL' for rf in rule_findings) else 'WARNING'}")
        print(f"      Reasons: {' | '.join(alert_reason)}")
        print("  --> ACTION: DISPATCH REALTIME WEBSOCKET NOTIFICATION TO DOCTOR DASHBOARD")
    else:
        print("  --> ACTION: NO ALERT TRIGGERED (Normal telemetry)")

    print("\n[Step 7] Backend Outbox Replication to MongoDB Atlas:")
    print("  --> Enqueue event & assessment with source_event_id='evt_sim_98234' for idempotent retry mirror.")
    print("=" * 78)


if __name__ == "__main__":
    simulate_backend_event_ingestion()

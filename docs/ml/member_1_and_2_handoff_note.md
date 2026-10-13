# Member 1 & Member 2 Integration Handoff Note

**From:** Member 3 (ML Analysis, Integration, QA and Deployment)  
**To:** Member 1 (Backend, Simulator & MySQL) & Member 2 (React & Flutter Frontends)  
**Package:** `packages/health_analysis`  
**Model Version:** `1.0.0`  
**Date:** 13 October 2026  

---

## 1. Handoff to Member 1 (Backend & Database Architecture)

### How to Import and Call the ML Package
The ML package is completely self-contained in `packages/health_analysis/` and requires **zero** FastAPI or database dependencies.

```python
# In your FastAPI route handler or ingestion worker:
from health_analysis import assess_health_event, evaluate_safety_rules

# After validating and persisting the raw telemetry reading to MySQL InnoDB:
assessment = assess_health_event(
    patient_id=event.patient_id,
    timestamp=event.timestamp,
    heart_rate=event.heart_rate,
    spo2=event.spo2,
    temperature_f=event.temperature_f,
    respiratory_rate=event.respiratory_rate, # Optional: simulator fields accepted safely
    systolic_bp=event.systolic_bp,           # Optional: simulator fields accepted safely
    diastolic_bp=event.diastolic_bp,         # Optional: simulator fields accepted safely
    source_event_id=event.source_event_id,   # Used for idempotent deduplication
    is_simulated=event.is_simulated,
)

# Convert to JSON-serializable dict:
assessment_payload = assessment.to_dict()
```

### Proposed Schema for MySQL `model_assessments` Table:
```sql
CREATE TABLE IF NOT EXISTS model_assessments (
    assessment_id VARCHAR(64) PRIMARY KEY,
    event_id VARCHAR(64) NOT NULL,
    patient_id VARCHAR(32) NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    score DECIMAL(5, 4) NOT NULL,
    result ENUM('normal_like', 'anomalous', 'insufficient_data') NOT NULL,
    features_json JSON NOT NULL,
    explanation_json JSON NOT NULL,
    window_start DATETIME NOT NULL,
    window_end DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_simulated BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (patient_id) REFERENCES patient_profiles(patient_id),
    INDEX idx_patient_time (patient_id, window_end)
) ENGINE=InnoDB;
```

### Key Integration Invariants for Member 1:
1. **Idempotency:** Passing `source_event_id` sets `assessment_id = f"asmt_{source_event_id}"`. When replaying pending outbox events or retrying failed network calls, this prevents duplicate assessment rows.
2. **Deterministic Rules Separation:** Always execute `evaluate_safety_rules()` alongside `assess_health_event()`. If the ML model ever returns `insufficient_data` (e.g. during a patient's first 4 readings), deterministic rules will still catch critical emergencies (SpO₂ < 90% or HR > 130 bpm).
3. **Simulator Fields:** The live simulator may emit respiratory rate and blood pressure. You can pass them directly to `assess_health_event()`; the service records them in `feature_summary` for auditability without corrupting the historical model vector.

---

## 2. Handoff to Member 2 (React Doctor Dashboard & Flutter Patient App)

### Data Contract Display Guidelines
FastAPI exposes the assessment dictionary via `/api/v1/patients/{id}/vitals/latest` and `/api/v1/patients/{id}/alerts`.

```json
{
  "assessment_id": "asmt_evt_sim_98234",
  "patient_id": "P101",
  "model_version": "1.0.0",
  "score": 0.5518,
  "result": "anomalous",
  "explanation": [
    "Multivariate vital strain index elevated to 9.4 (composite strain deviation).",
    "Temperature metric (temp_latest) measured 102.4 F vs historical baseline median of 98.6 F (+5.1x MAD)."
  ],
  "feature_summary": {
    "heart_rate_latest": 128.0,
    "spo2_latest": 91.0,
    "temperature_f_latest": 102.4,
    "heart_rate_rolling_mean": 81.3,
    "spo2_rolling_min": 91.0,
    "temp_rolling_max": 102.4,
    "vital_strain_index": 9.4
  },
  "is_simulated": true
}
```

### UI Implementation Rules:
1. **Score Rendering:** Render `score` as a normalized progress bar or gauge from `0.00` to `1.00`.
   - `< 0.40`: Green (`Normal`)
   - `0.40 – 0.49`: Yellow (`Elevated Variability`)
   - `≥ 0.50`: Red / Alert (`Anomalous Pattern`)
2. **Evidence Tooltips:** Display the `explanation` strings as bullet points on alert cards so clinicians understand *which vital signs triggered the flag*.
3. **Mandatory Disclaimer:** Display the visual pill `SIMULATED DATA — NOT FOR MEDICAL USE` in both the Doctor Dashboard header and the Flutter Patient App.
4. **Cold-Start / Insufficient History:** When `result == "insufficient_data"`, display an info badge: *"Calibrating Baseline (Reading X of 5)"* rather than showing a false green or red state.

---

## 3. Working Example Code
A complete end-to-end integration script demonstrating the full flow is available in:
[`examples/backend_integration_example.py`](file:///C:/Users/HARNOOR%20KANT/.gemini/antigravity/scratch/smart-health/examples/backend_integration_example.py).
Run it anytime with:
```bash
python examples/backend_integration_example.py
```

# Smart Health Monitoring System

A multi-tiered simulated health monitoring system designed for academic evaluation (Digital Assignment 3 / DA3).

---

## 1. System Architecture
- **Primary Relational Store (Authoritative):** MySQL 8 / InnoDB (Users, patient profiles, doctor assignments, normalized vital readings, canonical alerts, messages, audit logs).
- **Secondary Document Store (Analytics Mirror):** MongoDB Atlas (`smart_health_analytics.health_events`). Replicated idempotently via asynchronous outbox retry worker.
- **Backend API Gateway:** FastAPI (REST + WebSockets).
- **Frontend Clients:** React + TypeScript Doctor Dashboard & Flutter Patient App.
- **ML & Analytics Subsystem:** Member 3 Unsupervised Anomaly Detection (`packages/health_analysis`) & Streamlit Analytics Dashboard (`apps/analytics_streamlit`).

---

## 2. Member 3: ML Package (`packages/health_analysis`)
The ML component provides an importable, self-contained analysis package and integration contract for continuous anomaly detection on patient vitals:
- **Model:** Scikit-Learn `IsolationForest` (100 trees, 38 physiological features).
- **Core Measurements:** `heart_rate` (bpm), `spo2` (%), `temperature_f` (°F).
- **Live Simulator Compatibility:** Accepts optional `respiratory_rate` and `blood_pressure` without contaminating the historical model schema.
- **Leakage Prevention:** 70/30 chronological train/eval split per patient with frozen training baselines.
- **Deterministic Safeguards:** Paired with independent clinical safety rules (`evaluate_safety_rules`).

---

## 3. Quick Start & Reproducibility Commands

### Setup Environment
```bash
# Clone/navigate to project
cd "C:\Users\HARNOOR KANT\.gemini\antigravity\scratch\smart-health"

# Install dependencies (Python 3.12)
pip install -r requirements.txt
```

### 1. Audit Dataset
```bash
python scripts/audit_dataset.py data/raw/kaggle_health_dataset.csv --json-out docs/ml/dataset_profile.json
```

### 2. Extract Features
```bash
python scripts/extract_features.py data/raw/kaggle_health_dataset.csv --output-csv data/processed/features_kaggle_v1.csv
```

### 3. Train Model & Run Evaluation Benchmark
```bash
python scripts/train_model.py --input-csv data/raw/kaggle_health_dataset.csv --artifact-dir models/iforest_v1
```

### 4. Run Full Test Suite (38 Tests)
```bash
# Via pytest
python -m pytest tests/ -v

# Via standard Python unittest runner
python -m unittest discover -s tests -p "test_*.py" -v
```

### 5. Run Backend Integration Flow Example
```bash
python examples/backend_integration_example.py
```

---

## 4. Documentation Index (`docs/ml/`)
- [`model_card.md`](file:///C:/Users/HARNOOR%20KANT/.gemini/antigravity/scratch/smart-health/docs/ml/model_card.md): Official model card, intended use, prohibited claims.
- [`member_1_and_2_handoff_note.md`](file:///C:/Users/HARNOOR%20KANT/.gemini/antigravity/scratch/smart-health/docs/ml/member_1_and_2_handoff_note.md): Handoff guide for backend and frontend developers.
- [`model_evaluation_report.md`](file:///C:/Users/HARNOOR%20KANT/.gemini/antigravity/scratch/smart-health/docs/ml/model_evaluation_report.md): Holdout evaluation statistics and synthetic benchmark results.
- [`feature_engineering_pipeline_specification.md`](file:///C:/Users/HARNOOR%20KANT/.gemini/antigravity/scratch/smart-health/docs/ml/feature_engineering_pipeline_specification.md): Complete 43-feature schema and windowing policies.
- [`dataset_profiling_report.md`](file:///C:/Users/HARNOOR%20KANT/.gemini/antigravity/scratch/smart-health/docs/ml/dataset_profiling_report.md): Statistical profile of the 1,500-row historical CSV.
- [`data_dictionary.md`](file:///C:/Users/HARNOOR%20KANT/.gemini/antigravity/scratch/smart-health/docs/ml/data_dictionary.md): Units, constraints, and data-handling policies.
- [`test_report.md`](file:///C:/Users/HARNOOR%20KANT/.gemini/antigravity/scratch/smart-health/docs/ml/test_report.md): 38/38 unit and integration test register.
- [`implementation_checklist.md`](file:///C:/Users/HARNOOR%20KANT/.gemini/antigravity/scratch/smart-health/docs/ml/implementation_checklist.md): Roadmap of completed phases.

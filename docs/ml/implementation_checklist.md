# Member 3: ML, Analytics & Release Implementation Checklist

**System:** Smart Health Monitoring System  
**Role:** Member 3: ML Analysis, Integration, QA and Deployment  
**Status Date:** 13 October 2026  
**Current Phase:** Phase 1 Complete (Inspection, Data Profiling & ML Architecture)  

---

## 1. Repository State Record (Current Snapshot)

- **Root Location:** `C:\Users\HARNOOR KANT\.gemini\antigravity\scratch\smart-health`
- **Current Git / Directory Status:** Initialized with Master Plan directory structure.
- **Dataset File:** `data/raw/kaggle_health_dataset.csv` (1,500 rows, SHA-256: `50f391190cd58f87636c0f6ae45731f97791563210275f14e5208c75be96ec82`).
- **Test Suite Status:** 8/8 unit tests passing via `pytest` and `unittest` (`tests/test_data_validation.py`).
- **Installed Runtime:** Python 3.12.10 with `pandas 3.0.5`, `numpy 2.5.3`, `scipy 1.18.1`, `sklearn 1.9.1`, `pytest 9.1.1`.

---

## 2. Multi-Phase Implementation Checklist

### Phase 1: Inspection, Profiling & Architecture (CURRENT — COMPLETED)
- [x] Locate and inspect all documentation packs and reference files.
- [x] Locate historical Kaggle dataset (`D:\Downloads\kaggle_health_dataset.csv`).
- [x] Preserve original CSV unchanged and mirror to `data/raw/` with verified checksum.
- [x] Implement reproducible dataset audit script (`scripts/audit_dataset.py`).
- [x] Implement schema and data validation module (`packages/health_analysis/validation.py`).
- [x] Validate all data quality invariants (0 nulls, 0 duplicates, 5 patients, unzoned timestamps).
- [x] Formulate unit tests for validation logic (`tests/test_data_validation.py`) and verify 8/8 pass.
- [x] Document dataset profiling report (`docs/ml/dataset_profiling_report.md`).
- [x] Document data dictionary with unit/provenance rules (`docs/ml/data_dictionary.md`).
- [x] Define ML architecture & 8 feature families (`docs/ml/ml_architecture_and_feature_schema.md`).
- [x] Register project risks, assumptions & unresolved decisions (`docs/ml/risks_assumptions_decisions.md`).
- [x] Document proposed directory structure & dependencies (`docs/ml/proposed_directory_structure_and_dependencies.md`).

---

### Phase 2: Feature Engineering & Baseline Models (COMPLETED)
- [x] Implement `packages/health_analysis/units.py`:
  - [x] Canonical Fahrenheit/Celsius normalization and conversion functions.
- [x] Implement `packages/health_analysis/schema.py`:
  - [x] Ordered 43-feature schema (`FEATURE_SCHEMA_VERSION = 1.0.0`).
  - [x] Data quality states (`DataQualityState`) and configuration (`FeatureExtractorConfig`).
- [x] Implement `packages/health_analysis/baseline.py`:
  - [x] Fit/transform patient baseline manager (`BaselineManager`).
  - [x] Sequential streaming baseline tracker (`IncrementalBaselineTracker`).
  - [x] Robust scaling via Median Absolute Deviation (MAD).
  - [x] Zero-variability safeguard.
- [x] Implement `packages/health_analysis/features.py`:
  - [x] Time-aware sliding window (default 120 minutes).
  - [x] Rolling mean, median, standard deviation, and MAD.
  - [x] Elapsed-time trend slope accounting for irregular intervals.
  - [x] Absolute and relative change indicators.
  - [x] Cross-metric vital interaction features.
  - [x] In-memory streaming feature extractor (`StreamingFeatureExtractor`).
- [x] Implement feature extraction CLI tool (`scripts/extract_features.py`).
- [x] Implement comprehensive unit tests (`tests/test_features.py`) — 20/20 tests passing.
- [x] Document feature pipeline specification (`docs/ml/feature_engineering_pipeline_specification.md`).

---

### Phase 3: Isolation Forest Model Pipeline & Synthetic Benchmark (COMPLETED)
- [x] Implement `packages/health_analysis/split.py`:
  - [x] Chronological 70/30 split per patient with zero lookahead leakage.
  - [x] Leave-one-patient-out cross-validation generator.
- [x] Implement `packages/health_analysis/models.py`:
  - [x] Wrap `sklearn.ensemble.IsolationForest` with deterministic random seed (`42`).
  - [x] Calibrated $[0.0, 1.0]$ score normalization (higher = more anomalous).
  - [x] Feature contribution extractor for clinical explainability.
  - [x] Safe serialization and deserialization with metadata schema validation.
- [x] Implement `scripts/train_model.py`:
  - [x] End-to-end training, chronological evaluation, and synthetic benchmark runner.
  - [x] Versioned artifact persistence (`models/iforest_v1/model.joblib` and `metadata.json`).
- [x] Implement unit tests (`tests/test_model.py`) — 27/27 tests passing across the test suite.
- [x] Document model evaluation report (`docs/ml/model_evaluation_report.md`).
- [x] Export machine-readable evaluation results (`docs/ml/evaluation_results.json`).

---

### Phase 4: MongoDB Aggregations & DA3 SQL vs NoSQL Benchmark (NEXT IMMEDIATE STEP)
- [ ] Design and implement multi-stage MongoDB aggregation pipelines on `smart_health_analytics.health_events`:
  - [ ] Hourly/daily rollups of metric statistics per patient.
  - [ ] Anomaly count summaries across patients.
- [ ] Implement reproducible SQL vs NoSQL benchmark script (`scripts/benchmark_sql_nosql.py`):
  - [ ] Workload A: Latest $N$ readings for patient.
  - [ ] Workload B: Aggregate statistics over time window.
  - [ ] Workload C: Date/hour bucket counts.
  - [ ] Measure 20–30 repetitions, collect p50, p95, p99 latency and execution plans.

---

### Phase 5: Streamlit Analytics Dashboard
- [ ] Build `apps/analytics_streamlit/app.py`:
  - [ ] Patient telemetry timeline and rolling trend visualizer.
  - [ ] Anomaly score gauge and top contributing feature breakdown.
  - [ ] MongoDB Atlas aggregation charts.
  - [ ] SQL vs NoSQL benchmark latency comparison graphs.

---

### Phase 6: Dockerization, Backup/Restore Rehearsal & E2E QA
- [ ] Prepare Docker Compose baseline configuration for services.
- [ ] Write and test backup/restore script (`mysqldump` and `mongodump` / `mongorestore`).
- [ ] Compile full test report with captured evidence for the DA3 viva submission.

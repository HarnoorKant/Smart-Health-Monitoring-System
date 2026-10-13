# Dataset Profiling & Audit Report

**System:** Smart Health Monitoring System  
**Component:** Member 3 — ML Analysis, Integration, QA & Deployment  
**Audit Date:** 13 October 2026  
**Auditor:** Member 3 ML Subsystem (`scripts/audit_dataset.py`)  

---

## 1. Executive Summary

This report establishes the baseline empirical profile of the historical health dataset supplied for the Smart Health Monitoring System. The dataset was located in `D:\Downloads\kaggle_health_dataset.csv` and mirrored into `data/raw/kaggle_health_dataset.csv` with SHA-256 integrity verification.

All statistical values, sampling distributions, and integrity checks below reflect **actual measured calculations** produced by the reproducible auditing tool `scripts/audit_dataset.py`.

---

## 2. File Metadata & Cryptographic Integrity

| Attribute | Measured Value |
| :--- | :--- |
| **Primary File Path** | `data/raw/kaggle_health_dataset.csv` |
| **Source File Path** | `D:\Downloads\kaggle_health_dataset.csv` |
| **Alternative Found File** | `D:\Downloads\DOC-20261013-WA0056.csv` |
| **File Size (Primary)** | `58,751 bytes` (57.37 KB) |
| **SHA-256 Checksum** | `50f391190cd58f87636c0f6ae45731f97791563210275f14e5208c75be96ec82` |
| **Total Rows** | `1,500` |
| **Total Columns** | `5` |
| **Header Row** | `patient_id,heart_rate,spo2,temperature_f,timestamp` |

> [!NOTE]  
> The secondary file `DOC-20261013-WA0056.csv` (SHA-256: `ca15cf7776a0a670c834ba2693edacbf4eb785acd26d06d1617d3a56425f65a7`, 57,549 bytes) contains the exact same 1,500 numerical readings and records, but omits leading zero padding in hours (e.g. `8:19:00` vs `08:19:00`). `kaggle_health_dataset.csv` is maintained as the authoritative canonical reference.

---

## 3. Schema & Column Attributes

| Column Name | Inferred Dtype | Storage Dtype | Null Count | Null % | Value Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `patient_id` | Categorical String | `object` / `string` | `0` | `0.0%` | Categorical ID (`P101` – `P105`) |
| `heart_rate` | Integer | `int64` | `0` | `0.0%` | Discrete bpm (60 – 135) |
| `spo2` | Continuous Float | `float64` | `0` | `0.0%` | Percentage % (88.1 – 99.9) |
| `temperature_f` | Continuous Float | `float64` | `0` | `0.0%` | Degrees Fahrenheit (97.5 – 103.0) |
| `timestamp` | Datetime String | `object` | `0` | `0.0%` | Unzoned string (`YYYY-MM-DD HH:MM:SS`) |

---

## 4. Data Quality & Integrity Checks

- **Full Duplicate Rows:** `0` (no identical records).
- **Duplicate (patient_id, timestamp) Pairs:** `0` (each patient has at most one reading per timestamp).
- **Malformed / Non-Numeric Values:** `0` across `heart_rate`, `spo2`, and `temperature_f`.
- **Infinite / NaN Values:** `0`.
- **Invalid Timestamp Formats:** `0` (all 1,500 strings parse validly via ISO/standard datetime parser).
- **Global Chronological Ordering:** `True` (records in the CSV are strictly sorted by increasing timestamp).

---

## 5. Scope & Boundary Verification

The dataset was inspected for absent fields according to the Version 3.0 Dataset Alignment Addendum:

- **Diagnostic Labels Present:** `False` (no medical ground truth or clinical disease classification).
- **Anomaly Labels Present:** `False` (no binary or multi-class anomaly ground-truth column).
- **Respiratory Rate Present:** `False` (must **never** be fabricated for historical data).
- **Blood Pressure Present:** `False` (must **never** be fabricated for historical data).

---

## 6. Measurement Summary Statistics

| Metric | Unit | Min | Max | Mean | Median | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`heart_rate`** | beats/min (bpm) | `60.00` | `135.00` | `82.90` | `82.00` | `15.11` |
| **`spo2`** | percentage (%) | `88.10` | `99.90` | `96.49` | `96.60` | `2.28` |
| **`temperature_f`** | Fahrenheit (°F) | `97.50` | `103.00` | `98.69` | `98.60` | `0.97` |

---

## 7. Patient Cohort & Temporal Profiles

The dataset spans 5 unique synthetic or de-identified patient cohorts over approximately 10 days (2023-10-01 to 2023-10-11).

| Patient ID | Observations | First Timestamp | Last Timestamp | Chronological | Min Interval | Median Interval | Mean Interval | Max Interval | Gaps > 2h |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P101** | `293` | 2023-10-01 08:19:00 | 2023-10-11 11:28:00 | `True` | 5.0 min | 35.0 min | 50.0 min | 311.0 min (5.18h) | 21 |
| **P102** | `302` | 2023-10-01 09:06:00 | 2023-10-11 13:05:00 | `True` | 5.0 min | 35.0 min | 48.6 min | 261.0 min (4.35h) | 19 |
| **P103** | `296` | 2023-10-01 08:28:00 | 2023-10-11 13:20:00 | `True` | 5.0 min | 35.0 min | 49.8 min | 290.0 min (4.83h) | 22 |
| **P104** | `325` | 2023-10-01 09:27:00 | 2023-10-11 13:11:00 | `True` | 5.0 min | 30.5 min | 45.1 min | 253.0 min (4.22h) | 24 |
| **P105** | `284` | 2023-10-01 08:13:00 | 2023-10-11 13:35:00 | `True` | 5.0 min | 36.0 min | 52.0 min | 275.0 min (4.58h) | 23 |
| **Total / Overall** | `1,500` | 2023-10-01 08:13:00 | 2023-10-11 13:35:00 | `True` | — | — | — | — | 109 |

### Temporal Observations & Sampling Dynamics:
1. **Irregular Sampling Intervals**: Readings are **not periodic 1-second streams** (unlike the live simulator BLE model). The minimum interval between consecutive readings for any patient is **5.0 minutes**, the median interval is **~35 minutes**, and the mean interval is **~49 minutes**.
2. **Nighttime & Inactivity Gaps**: Each patient exhibits between 19 and 24 gaps exceeding 2 hours (up to ~5.2 hours maximum). These naturally correspond to sleep intervals or periods of non-wear.
3. **Implication for Window-Based Feature Engineering**: Rolling feature extraction must use **time-aware rolling windows** (e.g. `pandas.rolling(window='2h', on='timestamp')`) rather than fixed row-count windows (e.g. `window=10`). A 10-row window could span anywhere from 50 minutes to 15 hours.

---

## 8. Validation Script Reference

This report is programmatically reproducible via:
```bash
python scripts/audit_dataset.py data/raw/kaggle_health_dataset.csv --json-out docs/ml/dataset_profile.json
```
And its integrity invariants are tested via:
```bash
pytest tests/test_data_validation.py -v
```

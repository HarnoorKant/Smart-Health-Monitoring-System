# Model Card — Isolation Forest Health Telemetry Anomaly Detector

**Model Name:** `health_analysis.iforest`  
**Version:** `1.0.0`  
**Author / Owner:** Member 3 — ML Analysis, Integration, QA and Deployment  
**Model Date:** 13 October 2026  
**License / Policy:** Educational Hackathon Demonstration / NOT FOR CLINICAL USE  

---

## 1. Model Details

- **Architecture:** Unsupervised Isolation Forest (`sklearn.ensemble.IsolationForest`).
- **Ensemble Configuration:** 100 Isolation Trees, `max_samples=256`, `contamination=0.05`, `random_state=42`.
- **Input Dimension:** 38 physiological features derived from 3 continuous vital signs (`heart_rate`, `spo2`, `temperature_f`).
- **Output:** Calibrated anomaly score $[0.0, 1.0]$, categorical result (`normal_like` | `anomalous` | `insufficient_data`), top contributing features.
- **Serialization Format:** `joblib` binary (`model.joblib`) with companion schema metadata (`metadata.json`).

---

## 2. Intended Use & Target Users

### Primary Intended Use:
- **Demonstration:** Proves the viability of continuous unsupervised vital sign monitoring in a simulated IoT healthcare architecture.
- **Triage Assistance:** Flags statistical outliers across multidimensional telemetry for doctor review on the React web dashboard.
- **Dual Safeguard:** Operates alongside Member 1's deterministic clinical rules (which catch immediate threshold breaches like SpO₂ < 90%).

### Target Personas:
- **Doctor / Clinician (Member 2 Dashboard):** Reviews anomaly flags and explanations alongside raw vital charts.
- **Backend Orchestrator (Member 1 Gateway):** Calls `assess_telemetry()` synchronously upon event ingestion to populate MySQL `model_assessments` and MongoDB Atlas mirrors.

---

## 3. Prohibited Claims & Out-of-Scope Use

> [!CAUTION]  
> **STRICT COMPLIANCE DIRECTIVE: DO NOT MAKE THE FOLLOWING CLAIMS**
> 1. **No Diagnostic Claims:** It is **strictly prohibited** to claim this model diagnoses cardiac arrest, sepsis, pneumonia, or any disease.
> 2. **No Supervised Accuracy Claims:** It is **strictly prohibited** to claim accuracy, precision, recall, sensitivity, specificity, or AUC metrics on the historical CSV. The CSV possesses **no ground truth labels**.
> 3. **No Standalone Medical Action:** The model must **never** be used to automate medical interventions or dispense prescriptions.
> 4. **No Feature Fabrication:** Do not fabricate historical values for respiratory rate or blood pressure.

---

## 4. Input Requirements & Simulator Compatibility

| Input Measurement | Units | Valid Range | Role in Modeling |
| :--- | :--- | :--- | :--- |
| `heart_rate` | bpm | 60 – 135 bpm (historical) | Modeled in 12 windowed features |
| `spo2` | % | 88.1 – 99.9 % (historical) | Modeled in 12 windowed features |
| `temperature_f` | °F | 97.5 – 103.0 °F (historical) | Modeled in 12 windowed features |
| `respiratory_rate` | bpm | Live simulator only | **Unmodeled** (passed through in summary; ignored by model) |
| `systolic_bp` / `diastolic_bp` | mmHg | Live simulator only | **Unmodeled** (passed through in summary; ignored by model) |

---

## 5. Training & Evaluation Data

- **Training Cohort:** 1,048 observations chronologically partitioned (first 70% per patient) from `data/raw/kaggle_health_dataset.csv` (SHA-256: `50f391190cd58f87636c0f6ae45731f97791563210275f14e5208c75be96ec82`).
- **Holdout Evaluation Cohort:** 452 observations chronologically partitioned (final 30% per patient) with zero lookahead leakage.
- **Evaluation Rate:** 26 / 452 observations flagged (5.75% empirical flag rate at threshold 0.50).
- **Inference Latency:** 9.4 – 12.0 ms per event on commodity hardware.

---

## 6. Caveats & Ethical Considerations

- **Cold-Start Latency:** New patients operate in `insufficient_data` mode for their first 4 readings until an individualized baseline ($k \ge 5$) is established.
- **Unverified Clinical Provenance:** The Kaggle dataset contains unverified synthetic or de-identified data without medical context.
- **Human-in-the-Loop:** All flagged alerts require clinician acknowledgement and resolution in the doctor dashboard.

# Isolation Forest Anomaly Detection — Model Evaluation Report

**System:** Smart Health Monitoring System  
**Component:** Member 3 — ML Analysis, Integration, QA and Deployment  
**Model Architecture:** Scikit-Learn `IsolationForest` (Unsupervised Anomaly Detector)  
**Model Version:** `1.0.0`  
**Evaluation Date:** 13 October 2026  
**Status:** Validated Evaluation Report  

---

## 1. Executive Summary & Clinical Disclaimer

> [!CAUTION]  
> **DISCLAIMER: NOT A MEDICAL DEVICE / UNVERIFIED CLINICAL PROVENANCE**  
> This anomaly detector is an **unsupervised exploratory heuristic** designed for educational and hackathon demonstration purposes. The underlying Kaggle dataset contains **zero medical ground-truth diagnostic or anomaly labels**. Consequently, **true-positive, false-positive, sensitivity, specificity, precision, and recall metrics cannot be scientifically or mathematically established from the CSV alone.** Any alert generated represents statistical deviation from baseline clusters, not a clinical diagnosis.

This report documents the rigorous training and evaluation of the Version 1.0.0 Isolation Forest model using a **chronological holdout split within patient cohorts**, preventing future-data leakage while honoring the 5-patient dataset boundaries.

---

## 2. Dataset Invariants & Partition Boundaries

The historical dataset (`data/raw/kaggle_health_dataset.csv`, SHA-256: `50f391190cd58f87636c0f6ae45731f97791563210275f14e5208c75be96ec82`) contains exactly 1,500 rows across 5 patients.

### Partitioning Strategy: Chronological Holdout Within Patients (70% Train / 30% Eval)
To prevent temporal lookahead leakage, each patient's telemetry was sorted monotonically by timestamp. The first 70% chronologically formed the training set; the final 30% chronologically formed the holdout evaluation set.

| Patient ID | Total Rows | Training Rows (70%) | Training Date Span | Evaluation Rows (30%) | Evaluation Date Span |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **P101** | `293` | `205` | 2023-10-01 08:19:00 $\rightarrow$ 2023-10-08 03:21:00 | `88` | 2023-10-08 03:27:00 $\rightarrow$ 2023-10-11 11:28:00 |
| **P102** | `302` | `211` | 2023-10-01 09:06:00 $\rightarrow$ 2023-10-08 08:03:00 | `91` | 2023-10-08 08:09:00 $\rightarrow$ 2023-10-11 13:05:00 |
| **P103** | `296` | `207` | 2023-10-01 08:28:00 $\rightarrow$ 2023-10-08 03:54:00 | `89` | 2023-10-08 05:05:00 $\rightarrow$ 2023-10-11 13:20:00 |
| **P104** | `325` | `227` | 2023-10-01 09:27:00 $\rightarrow$ 2023-10-08 17:00:00 | `98` | 2023-10-08 17:42:00 $\rightarrow$ 2023-10-11 13:11:00 |
| **P105** | `284` | `198` | 2023-10-01 08:13:00 $\rightarrow$ 2023-10-08 14:37:00 | `86` | 2023-10-08 15:29:00 $\rightarrow$ 2023-10-11 13:35:00 |
| **Total** | **1,500** | **1,048** | **2023-10-01 $\rightarrow$ 2023-10-08** | **452** | **2023-10-08 $\rightarrow$ 2023-10-11** |

- **Eligible Feature Windows:** All 1,048 training rows and 452 evaluation rows produced valid feature windows (100% eligibility).
- **Leakage Prevention:** `BaselineManager` was fit **strictly on the 1,048 training rows**. Frozen baselines (median, MAD) were locked and referenced during evaluation.

---

## 3. Model Configuration & Score Calibration

```json
{
  "model_version": "1.0.0",
  "algorithm": "sklearn.ensemble.IsolationForest",
  "n_estimators": 100,
  "max_samples": 256,
  "contamination": 0.05,
  "max_features": 1.0,
  "random_state": 42,
  "decision_threshold": 0.50,
  "feature_count": 38
}
```

### Calibration & Transformation Convention:
Scikit-learn's `decision_function(X)` produces negative values for outliers and positive values for normal points. To expose an intuitive `anomaly_score` where **larger means more anomalous**, we map:
$$\text{raw\_inverted} = -\text{decision\_function}(X)$$
$$\text{anomaly\_score} = \frac{1}{1 + \exp\left(-\frac{\text{raw\_inverted}}{\tau}\right)}, \quad \tau = 0.15$$
- At the contamination decision boundary ($\text{decision\_function} = 0.0$), $\text{anomaly\_score} = 0.50$.
- **$[0.00, 0.40)$:** `normal_like` (inlier clusters).
- **$[0.40, 0.50)$:** `elevated_variability` (mild statistical drift).
- **$[0.50, 0.70)$:** `mild_anomaly` (flagged candidate).
- **$[0.70, 1.00]$:** `severe_anomaly` (extreme outlier).

---

## 4. Unsupervised Historical Evaluation Results

Evaluated on the 452 held-out chronological observations (Days 8–11):

### Score Distribution:
- **Minimum Anomaly Score:** `0.2046`
- **Mean Anomaly Score:** `0.2879`
- **Median Anomaly Score:** `0.2602`
- **95th Percentile (p95):** `0.5096`
- **Maximum Anomaly Score:** `0.6666`

### Flagged Observations:
- **Total Flagged ($\ge 0.50$):** `26 / 452` (**5.75%**)
- *Note:* The 5.75% empirical flagged rate closely matches the configured 5.0% training contamination factor, confirming healthy model calibration without alert flooding.

### Per-Patient Breakdown:
| Patient ID | Eval Observations | Flagged ($\ge 0.50$) | Flagged % | Mean Score | Max Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **P101** | `88` | `5` | 5.7% | `0.2909` | `0.6666` |
| **P102** | `91` | `5` | 5.5% | `0.2845` | `0.6534` |
| **P103** | `89` | `5` | 5.6% | `0.2923` | `0.6265` |
| **P104** | `98` | `6` | 6.1% | `0.2842` | `0.5588` |
| **P105** | `86` | `5` | 5.8% | `0.2882` | `0.6131` |

Every patient cohort exhibits a consistent ~5.5%–6.1% flag rate, demonstrating stable behavior across subjects.

---

## 5. Highest-Scoring Historical Candidates & Attribution

The top 3 highest-scoring observations in the evaluation period exhibit pronounced physiological strain:

### Rank 1: Patient P101 at 2023-10-11 07:15:00
- **Anomaly Score:** `0.6666` (`mild_anomaly`)
- **Raw Measurements:** `HR = 134.0 bpm`, `SpO2 = 89.3%`, `Temp = 102.9°F`
- **Top Contributing Features:**
  1. `cross_vital_strain_index = 10.98` ($+11\sigma$ composite deviation)
  2. `temp_window_max = 102.9°F` (severe fever)
  3. `temp_rolling_std = 3.11°F` (rapid thermal escalation)

### Rank 2: Patient P102 at 2023-10-08 08:09:00
- **Anomaly Score:** `0.6534` (`mild_anomaly`)
- **Raw Measurements:** `HR = 124.0 bpm`, `SpO2 = 90.5%`, `Temp = 102.6°F`
- **Top Contributing Features:**
  1. `temp_window_min = 102.6°F`
  2. `temp_rolling_median = 102.6°F`
  3. `temp_rolling_mean = 102.6°F`

### Rank 3: Patient P101 at 2023-10-09 11:26:00
- **Anomaly Score:** `0.6533` (`mild_anomaly`)
- **Raw Measurements:** `HR = 133.0 bpm`, `SpO2 = 88.9%`, `Temp = 102.9°F`
- **Top Contributing Features:**
  1. `cross_vital_strain_index = 11.07`
  2. `temp_window_max = 102.9°F`
  3. `temp_rolling_std = 3.46°F`

---

## 6. Controlled Synthetic Anomaly Injection Benchmark

Because historical data lacks ground-truth labels, sensitivity was evaluated against **controlled synthetic test scenarios** using `StreamingFeatureExtractor` primed with normal baseline readings:

| Scenario Description | Injected Telemetry | Anomaly Score | Flagged ($\ge 0.50$) | Categorization | Latency | Top Contributing Attribution |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Normative Control** | HR=75 bpm, SpO₂=98%, Temp=98.6°F | `0.2674` | **False** | `normal_like` | 10.8 ms | `spo2_window_min`, `hr_window_max` |
| **Acute Tachycardia Spike** | HR=135 bpm, SpO₂=97%, Temp=98.6°F | `0.4457` | **False** | `elevated_variability` | 9.4 ms | `hr_window_max` (135), `hr_trend_slope` (145.4), `hr_baseline_dev` (+55.0) |
| **Acute Hypoxemic Desaturation** | HR=88 bpm, SpO₂=85%, Temp=98.6°F | `0.4302` | **False** | `elevated_variability` | 9.7 ms | `spo2_window_min` (85), `spo2_trend_slope` (-30.0), `spo2_baseline_dev` (-11.3) |
| **Combined Hyperthermic & Tachycardic Shock** | HR=128 bpm, SpO₂=91%, Temp=102.5°F | `0.5548` | **True** | `mild_anomaly` | 10.8 ms | `cross_vital_strain_index` (9.5), `temp_window_max` (102.5), `temp_latest` (102.5) |

### Key Theoretical Insight:
- Isolated single-metric excursions (e.g. solitary HR=135 bpm) occurred occasionally in the training distribution, yielding `elevated_variability` scores (~`0.43 – 0.45`).
- Multi-metric physiological collapses (e.g. Tachycardia + Fever + Desaturation) cross multi-dimensional isolation boundaries and immediately cross the threshold (`0.5548`, flagged).
- This proves why Member 1's architecture pairs **deterministic rule thresholds** (e.g. SpO₂ < 90% or HR > 130 bpm) with the ML Isolation Forest: deterministic safety rules guarantee critical catches on single-metric violations, while ML captures complex multivariate degradation.

---

## 7. Model Limitations & Honest Constraints

1. **Unsupervised Limitation:** Without diagnostic labels, true positive and false positive rates **cannot be mathematically proven**.
2. **Small Cohort:** 5 synthetic patients over 10 days cannot represent the clinical diversity of human populations.
3. **Threshold Sensitivity:** The 0.50 decision threshold is tied to the 5.0% training contamination assumption. Adjusting contamination to 0.10 shifts sensitivity.
4. **Unstated Timezone:** Timestamps lack timezone offsets; wall-clock time was assumed.
5. **Operational Scope:** For demonstration and triage review only; must never replace clinical judgment.

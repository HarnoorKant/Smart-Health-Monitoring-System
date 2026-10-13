# Machine Learning Architecture & Feature Schema Document

**System:** Smart Health Monitoring System  
**Component:** Member 3: ML Analysis, Integration, QA and Deployment  
**Model Family:** Scikit-Learn `IsolationForest` (Unsupervised Anomaly Detection)  
**Version:** 3.0 (Kaggle Historical & Live Simulator Compatible)  
**Status:** Architecture Specification  

---

## 1. Architectural Overview & Design Philosophy

The ML subsystem operates as an **unsupervised anomaly and trend detection engine** embedded within `packages/health_analysis/`. Its primary role is to ingest incoming patient physiological telemetry, compute temporal and cross-metric features, and produce a versioned, explainable assessment score.

```
       Incoming Telemetry Reading
 (patient_id, timestamp, HR, SpO2, Temp)
                   │
                   ▼
       ┌────────────────────────┐
       │ Data Validation Layer  │  <-- Schema, Types, Range Checks
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Historical Feature     │  <-- Patient-specific sliding history,
       │ Engineering Engine     │      time-aware rolling stats, MAD, trend
       └───────────┬────────────┘
                   │
                   ▼
         Feature Vector (14D)
                   │
                   ▼
       ┌────────────────────────┐
       │ Isolation Forest Model │  <-- Ensemble of Isolation Trees (unsupervised)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Post-Processor &       │  <-- Calibrate score [0.0 - 1.0],
       │ Explainability Engine  │      extract top contributing features
       └───────────┬────────────┘
                   │
                   ▼
    Standard Assessment Output
 (score, is_anomalous, reasons, latency_ms)
```

### Core Invariants:
1. **Unsupervised Formulation:** The model makes **no diagnostic claims**. It scores statistical outliers relative to normative physiological clusters and patient baselines.
2. **Deterministic Fallback:** If insufficient historical data exists (< 5 readings for a patient), the system gracefully outputs an `insufficient_history` flag rather than failing or hallucinating features.
3. **No Cross-Talk / Zero Leakage:** Features are computed strictly using past and current observations. No future values are accessed.

---

## 2. Feature Schema & Feature Families

The feature extractor produces a vector across **8 distinct feature families**, utilizing solely the 3 verified historical measurements (`heart_rate`, `spo2`, `temperature_f`) and metadata:

### Family 1: Current Measurement Values (Instantaneous)
- `feat_hr_curr`: Raw current heart rate ($60.0 \le x \le 135.0$ bpm).
- `feat_spo2_curr`: Raw current oxygen saturation ($88.1 \le x \le 99.9$ %).
- `feat_temp_curr`: Raw current temperature ($97.5 \le x \le 103.0$ °F).

### Family 2: Rolling Central Tendency (Mean & Median)
- `feat_hr_rolling_mean_2h`: Mean heart rate over prior 2-hour window.
- `feat_spo2_rolling_median_2h`: Median SpO₂ over prior 2-hour window.
- `feat_temp_rolling_mean_2h`: Mean temperature over prior 2-hour window.

### Family 3: Rolling Dispersion & Robust Variability
- `feat_hr_rolling_std_2h`: Standard deviation of heart rate over 2-hour window.
- `feat_hr_rolling_mad_2h`: Median Absolute Deviation of heart rate:
  $$\text{MAD} = \text{median}(|x_i - \text{median}(x)|)$$
  *Robust against extreme sensor spikes where sample standard deviation would distort.*
- `feat_spo2_rolling_std_2h`: SpO₂ dispersion over 2-hour window.

### Family 4: Temporal Trend & Rate-of-Change
- `feat_hr_slope_per_hour`: Trend slope of heart rate:
  $$\frac{\Delta \text{HR}}{\Delta t_{\text{hours}}} = \frac{\text{HR}_t - \text{HR}_{t-1}}{(t - t_{-1}) / 3600}$$
- `feat_temp_slope_per_hour`: Rate of temperature climb ($\Delta \text{Temp} / \Delta t$).

### Family 5: Patient-Specific Baseline Deviation
*Calculated only when $\ge 5$ prior historical observations exist for the patient.*
- `feat_hr_dev_from_baseline`: $\text{HR}_t - \text{Median}_{\text{patient}}(\text{HR})$.
- `feat_temp_dev_from_baseline`: $\text{Temp}_t - \text{Median}_{\text{patient}}(\text{Temp})$.

### Family 6: Missingness & Data Quality Indicators
- `feat_has_missing_metrics`: Binary indicator ($1$ if any expected metric is null, else $0$).
- `feat_sensor_variance_zero`: Binary indicator ($1$ if sensor values remain flatlined over 3+ readings, indicating stuck hardware).

### Family 7: Freshness & Measurement Gap
- `feat_sampling_interval_min`: Elapsed time since the patient's previous reading in minutes ($t_i - t_{i-1}$). Captures whether the reading immediately follows a protracted gap (> 2 hours).

### Family 8: Cross-Metric Interactions
- `feat_hypoxia_tachycardia_ratio`:
  $$\text{HTR} = \frac{\text{HR}}{\text{SpO}_2}$$
  *Physiological rationale: Concurrently falling SpO₂ with compensatory elevated Heart Rate signifies cardiovascular strain, elevating the ratio.*
- `feat_vital_strain_index`: Composite z-score proxy:
  $$Z_{\text{HR}} + Z_{\text{Temp}} - Z_{\text{SpO2}}$$

---

## 3. Model Architecture & Hyperparameters

The anomaly scoring engine employs `sklearn.ensemble.IsolationForest`:

```python
IsolationForest(
    n_estimators=100,            # 100 isolation trees
    max_samples=256,             # Standard sub-sampling to prevent masking/swamping
    contamination=0.05,          # Assumed ~5% anomalous events in reference data
    max_features=1.0,            # Use all engineered features
    bootstrap=False,
    random_state=42,             # Deterministic seed for reproducible evaluation
    n_jobs=-1
)
```

### Anomaly Score Calibration:
Scikit-learn's native `decision_function(X)` yields values where lower numbers indicate anomalies. We apply monotonic sigmoid / linear mapping to normalize the output score to $[0.0, 1.0]$:
$$\text{Score}_{\text{norm}} = \frac{1}{1 + e^{10 \cdot (\text{decision\_function}(X))}}$$
- **$[0.0, 0.40)$:** Typical / Normative physiological behavior.
- **$[0.40, 0.70)$:** Mild deviation / Elevated variability (Warning candidate).
- **$[0.70, 1.00]$:** Significant anomaly / Severe deviation (Critical alert candidate).

---

## 4. Evaluation Strategy & Zero-Leakage Policy

Time-series health data exhibits high temporal autocorrelation. Standard random $k$-fold cross-validation is **strictly prohibited** because adjacent readings from the same patient would leak into both training and test folds.

### Evaluation Protocols:
1. **Chronological Temporal Split (Within-Patient Evaluation):**
   - For each patient, the first 70% of readings chronologically form the training set.
   - The subsequent 30% chronologically form the evaluation/test set.
   - Proves whether a baseline trained on past history detects emergent anomalies in future days.
2. **Leave-One-Patient-Out Cross-Validation (Inter-Patient Generalization):**
   - Train on 4 patients (`P101`, `P102`, `P103`, `P104`); test on `P105`.
   - Rotate across all 5 patients ($5$-fold GroupKFold).
   - Proves whether the anomaly detector recognizes universal physiological abnormalities in patients it has never observed before.
3. **Synthetic Injection Test Benchmark:**
   - Unsupervised models cannot compute Precision/Recall without labeled anomalies.
   - Evaluation will inject controlled synthetic anomalies (e.g. simulated hypoxemia: SpO₂ drops to 85%; tachycardia: HR jumps to 130 bpm) into the test set to empirically verify recall and detection latency.

---

## 5. Assessment Output Contract

The analysis package exposes a typed dataclass / dictionary matching the Member 1 relational schema (`model_assessments` table) and MongoDB document contract:

```python
@dataclass
class AssessmentResult:
    assessment_id: str             # UUID
    patient_id: str                # e.g. "P101"
    timestamp: str                 # ISO timestamp
    model_version: str             # e.g. "iforest-v1.0.0"
    anomaly_score: float           # Normalized [0.0, 1.0]
    is_anomaly: bool               # Binary flag based on calibrated threshold
    result_status: str             # "normal_like" | "mild_anomaly" | "severe_anomaly" | "insufficient_history"
    top_contributing_features: list[str]  # e.g. ["spo2_rolling_min_2h (88.1%)", "hr_curr (128 bpm)"]
    is_simulated: bool = True      # Compliance flag
    latency_ms: float = 0.0        # Measured execution time
```

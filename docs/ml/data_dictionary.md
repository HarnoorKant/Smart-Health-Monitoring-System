# Data Dictionary — Historical Telemetry & Model Schema

**System:** Smart Health Monitoring System  
**Component:** Member 3: ML Analysis, Integration, QA & Deployment  
**Version:** 3.0 (Kaggle Aligned)  
**Status:** Approved Specification  

---

## 1. Overview & Provenance Disclaimer

> [!CAUTION]  
> **DISCLAIMER: NOT A MEDICAL DEVICE / NOT CLINICALLY VALIDATED**  
> All data referenced herein originates from a historical Kaggle synthetic/de-identified research dataset (`kaggle_health_dataset.csv`). Clinical provenance is unverified. Any downstream machine learning models are for educational, exploratory demonstration purposes only and must not be utilized for medical triage, diagnosis, or treatment decisions.

The historical dataset consists of **1,500 observations** spanning 5 distinct patient identifiers across a 10-day monitoring period in October 2023.

---

## 2. Historical Dataset Fields (Raw Input Contract)

| Field Name | Raw Type | Target Type | Standard Units | Observed Range | Medical Plausibility Context | Modeling Role | Handling / Preprocessing Rules |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`patient_id`** | String | String (`category`) | Categorical ID | `P101` – `P105` | De-identified / synthetic patient identifier | **Grouping / Partitioning Key** | **NEVER** pass as a numerical feature. Used for patient-specific baselines, rolling histories, and group-aware cross-validation splits. |
| **`timestamp`** | String | Datetime (Unzoned) | ISO String (`YYYY-MM-DD HH:MM:SS`) | 2023-10-01 08:13:00 to 2023-10-11 13:35:00 | Wall-clock observation time | **Temporal Index / Window Sorter** | **NEVER** pass as a raw numeric feature. Used for chronological ordering, rolling temporal windows, interval calculation, and leakage-free time-series splits. Preserved as unzoned wall-clock time. |
| **`heart_rate`** | Integer | Float32 | beats per minute (bpm) | 60 – 135 bpm | Normal adult resting: 60–100 bpm; mild tachycardia: >100 bpm | **Continuous Feature** | Standardized via RobustScaler / z-score; input to rolling mean/std/MAD and rate-of-change indicators. |
| **`spo2`** | Float | Float32 | percentage (%) | 88.1% – 99.9% | Normal arterial oxygen saturation: ≥95%; hypoxemia threshold: <90% | **Continuous Feature** | Input to rolling metrics, desaturation counters, and cross-metric shock/stress indicators. |
| **`temperature_f`** | Float | Float32 | Degrees Fahrenheit (°F) | 97.5°F – 103.0°F | Normal baseline: ~98.6°F (37.0°C); fever threshold: ≥100.4°F (38.0°C) | **Continuous Feature** | **Unit Policy:** Maintained natively in Fahrenheit throughout feature extraction to prevent floating-point roundoff issues. An explicit conversion utility `(T_f - 32) * 5/9` is provided and tested for API display contracts. |

---

## 3. Disallowed / Unavailable Historical Fields

To prevent accidental data leakage or invalid synthetic backfilling, the following attributes are **strictly forbidden** from being fabricated or assumed in historical data:

| Field Name | Status in Historical CSV | System Policy |
| :--- | :--- | :--- |
| `respiratory_rate` | **ABSENT** | Available **only** in live simulator telemetry. Do not impute, interpolate, or fabricate in historical data. Must be set to `null` / `None` in historical API responses. |
| `systolic_bp` / `diastolic_bp` | **ABSENT** | Available **only** in live simulator telemetry. Do not invent baseline values for historical Kaggle records. |
| `is_anomaly` / `label` | **ABSENT** | The Kaggle dataset does not contain diagnostic or anomaly ground truth. All ML anomaly detection is **strictly unsupervised**. |
| `diagnosis` / `condition` | **ABSENT** | No clinical conditions are mapped. Ground-truth evaluation must use controlled synthetic test fixtures (e.g. injected anomalies). |

---

## 4. Temperature Representation Policy

1. **Internal ML Feature Pipeline:**
   - Feature engineering operates on native Fahrenheit values (`temperature_f`).
   - Mean: `98.69°F`, Median: `98.60°F`, Std: `0.97°F`.
2. **Canonical Transformation Function:**
   $$T_c = \frac{T_f - 32}{1.8}$$
3. **Display & API Contracts:**
   - When serving the doctor dashboard or MongoDB mirror, temperature can be converted to Celsius via the validated conversion module, with unit explicit in the field name (`temperature_c` vs `temperature_f`).

---

## 5. Timezone & Freshness Policy

1. **Source Timezone Undefined:**
   - The Kaggle dataset timestamps do not provide UTC offset or timezone identifiers (`2023-10-01 08:13:00`).
   - Silently assuming `UTC` or `Asia/Kolkata` or `America/New_York` is scientifically invalid.
   - Timestamps are maintained as **naive local wall-clock datetimes** for historical profiling and feature engineering.
2. **Freshness Invariants:**
   - Historical records represent past events (October 2023) and are labeled `IS_HISTORICAL=True` with `freshness="historical"`.
   - Live simulator records represent realtime events and receive current timestamps with `IS_SIMULATED=True` and real-time freshness thresholds (`fresh` < 30s, `stale` > 60s, `disconnected` > 300s).

---

## 6. Derived Feature Families (Feature Store Target)

| Feature Name | Type | Unit | Description | Formulation |
| :--- | :--- | :--- | :--- | :--- |
| `hr_curr` | Float | bpm | Current Heart Rate measurement | Raw observation |
| `spo2_curr` | Float | % | Current Oxygen Saturation measurement | Raw observation |
| `temp_curr` | Float | °F | Current Temperature measurement | Raw observation |
| `hr_rolling_mean_2h` | Float | bpm | Rolling 2-hour mean heart rate | Time-based rolling mean |
| `hr_rolling_std_2h` | Float | bpm | Rolling 2-hour standard deviation | Time-based rolling std |
| `hr_rolling_mad_2h` | Float | bpm | Rolling Median Absolute Deviation | $\text{median}(\|x_i - \text{median}(x)\|)$ |
| `spo2_rolling_min_2h` | Float | % | Rolling 2-hour minimum SpO₂ | Critical dip detector |
| `temp_rolling_max_2h` | Float | °F | Rolling 2-hour maximum Temperature | Peak fever detector |
| `hr_rate_of_change` | Float | bpm/hr | First derivative of heart rate over time | $\frac{\Delta \text{HR}}{\Delta t}$ (bpm/hour) |
| `hr_baseline_diff` | Float | bpm | Deviation from patient's personal median | $\text{HR} - \text{Median}_{\text{patient}}(\text{HR})$ |
| `temp_baseline_diff` | Float | °F | Deviation from patient's personal median | $\text{Temp} - \text{Median}_{\text{patient}}(\text{Temp})$ |
| `sampling_gap_minutes` | Float | min | Elapsed time since patient's prior record | $t_i - t_{i-1}$ in minutes |
| `hypoxia_tachycardia_idx` | Float | ratio | Cross-metric stress indicator | $\frac{\text{HR}}{\text{SpO}_2}$ (elevated during hypoxemic strain) |
| `shock_index_proxy` | Float | ratio | Physiological strain ratio proxy | $\frac{\text{HR}}{\text{Temp}_f}$ normalized |

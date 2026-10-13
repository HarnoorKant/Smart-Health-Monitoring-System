# Feature Engineering Pipeline Specification & Benchmark Report

**System:** Smart Health Monitoring System  
**Component:** Member 3 — ML Analysis, Integration, QA and Deployment  
**Package:** `packages/health_analysis`  
**Feature Schema Version:** `1.0.0`  
**Date:** 13 October 2026  
**Status:** Implemented & Validated (20/20 Unit Tests Passing)  

---

## 1. Overview & Architecture

The feature-engineering pipeline processes raw historical and streaming vital sign telemetry into standardized, leakage-free feature records. It implements an isolated architecture within `packages/health_analysis/`:

```
                 Raw Telemetry Event
       (patient_id, timestamp, HR, SpO2, Temp)
                         │
                         ▼
             ┌───────────────────────┐
             │   units.py            │ <-- Canonical Fahrenheit/Celsius normalization
             └───────────┬───────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │   baseline.py         │ <-- Frozen training baselines or streaming
             │   (BaselineManager)   │     history (min 5 samples, robust MAD)
             └───────────┬───────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │   features.py         │ <-- Time-aware sliding window (2-hour default)
             │   (FeatureExtractor)  │     Elapsed-time slopes, MAD, deviations
             └───────────┬───────────┘
                         │
                         ▼
                 FeatureRecord (43D)
         [38 Physiological + 5 Quality/Context]
```

### Separation of Concerns:
- **No Database / Gateway Leaks:** The package contains **zero** database queries, FastAPI routes, or WebSocket dependencies.
- **Strict Historical Boundaries:** Blood pressure and respiratory rate are **never** required and are excluded from the historical model vector.
- **Dual Processing Engines:**
  - `FeatureExtractor`: Batch processor with strict per-patient chronological sorting and time-based windowing.
  - `StreamingFeatureExtractor`: Stateful in-memory sliding buffer ($N=100$) for real-time evaluation of live simulator events with zero lookahead.

---

## 2. Explicit Window Policy & Interval Handling

Based on empirical observations from Part 1 (`docs/ml/dataset_profiling_report.md`):
- Minimum sampling interval: `5.0 minutes`.
- Median sampling interval: `30.5 – 36.0 minutes`.
- Nighttime / inactivity gaps: Up to `5.18 hours` (109 gaps exceeding 2 hours across the cohort).

### Configured Window Rules (`FeatureExtractorConfig`):
1. **Window Duration:** **120.0 minutes (2.0 hours)**.
   - For an evaluation event at timestamp $t$, the window aggregates all valid observations in $[t - 120\text{m}, t]$.
   - Captures acute cardiovascular and temperature shifts while discarding obsolete history.
2. **Elapsed-Time Trend Slope:**
   $$\text{Slope} = \frac{x_t - x_{t-1}}{\Delta t_{\text{hours}}} = \frac{x_t - x_{t-1}}{(t - t_{prev})_{\text{seconds}} / 3600.0}$$
   - Units: **Rate of change per hour** (e.g. bpm/hr, %/hr).
   - Correctly accounts for irregular sampling intervals rather than assuming a fixed frequency.
3. **Maximum Allowable Continuous Gap (`max_gap_minutes` = 180.0 minutes / 3.0 hours):**
   - If $\Delta t > 180\text{m}$, the reading is flagged with `GAP_EXCEEDED` and `quality_gap_exceeded = 1.0`.
   - **Crucial Rule:** Across gaps $> 180\text{m}$, trend slope and absolute change are **not interpolated or linked across the multi-hour void**. They reset to `0.0` until a subsequent reading arrives in the new continuous monitoring session.

---

## 3. Data Leakage Prevention & Baseline Management

Patient baselines must never incorporate future readings or cross-contaminate cohorts.

### Fit / Transform Protocol:
- **`BaselineManager.fit(df_train)`:**
  - Computes median, Median Absolute Deviation (MAD), mean, and standard deviation for each metric per patient strictly using the training partition.
  - Baselines are frozen and immutable during evaluation.
- **Zero Cross-Cohort Sharing:**
  - P101, P102, P103, P104, and P105 maintain independent baseline models. Patient A's baseline is never referenced for Patient B.
- **Insufficient History Graceful Degradation:**
  - Minimum baseline threshold: `k >= 5` valid observations.
  - If a patient has $< 5$ historical samples, `has_baseline = False`, `quality_has_baseline = 0.0`, raw and normalized baseline deviations are assigned `0.0`, and the record is flagged as `INSUFFICIENT_HISTORY`.
- **Zero-Variability Safe Robust Scaling:**
  - Normalized baseline deviation is computed via robust scaling:
    $$Z_{\text{robust}} = \frac{x - \text{Median}}{\max(1.4826 \times \text{MAD}, \epsilon)}$$
  - If a patient's readings are identical ($\text{MAD} = 0, \text{std} = 0$), the denominator is safely bounded by $\epsilon = 10^{-6}$, preventing floating-point zero-division exceptions.

---

## 4. Complete Ordered Feature Schema (`v1.0.0`)

The feature vector contains **43 ordered features** (38 physiological + 5 data quality):

| Index | Feature Name | Source Metric | Units | Description & Formula |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `hr_latest` | Heart Rate | bpm | Latest valid reading in window |
| **1** | `hr_rolling_mean` | Heart Rate | bpm | Mean over window |
| **2** | `hr_rolling_median` | Heart Rate | bpm | Median over window |
| **3** | `hr_rolling_std` | Heart Rate | bpm | Sample std dev ($N \ge 2$) |
| **4** | `hr_rolling_mad` | Heart Rate | bpm | $\text{median}(\|x - \text{median}(x)\|)$ |
| **5** | `hr_window_min` | Heart Rate | bpm | Minimum value in window |
| **6** | `hr_window_max` | Heart Rate | bpm | Maximum value in window |
| **7** | `hr_abs_change` | Heart Rate | bpm | $x_t - x_{t-1}$ |
| **8** | `hr_rel_change` | Heart Rate | ratio | $(x_t - x_{t-1}) / \|x_{t-1}\|$ |
| **9** | `hr_trend_slope` | Heart Rate | bpm/hr | $(x_t - x_{t-1}) / \Delta t_{\text{hours}}$ |
| **10** | `hr_baseline_dev` | Heart Rate | bpm | $x_t - \text{Median}_{\text{base}}$ |
| **11** | `hr_norm_baseline_dev` | Heart Rate | z-score | $(x_t - \text{Median}_{\text{base}}) / (1.4826 \times \text{MAD}_{\text{base}})$ |
| **12** | `spo2_latest` | SpO₂ | % | Latest valid reading in window |
| **13** | `spo2_rolling_mean` | SpO₂ | % | Mean over window |
| **14** | `spo2_rolling_median` | SpO₂ | % | Median over window |
| **15** | `spo2_rolling_std` | SpO₂ | % | Sample std dev ($N \ge 2$) |
| **16** | `spo2_rolling_mad` | SpO₂ | % | Median absolute deviation |
| **17** | `spo2_window_min` | SpO₂ | % | Minimum value in window |
| **18** | `spo2_window_max` | SpO₂ | % | Maximum value in window |
| **19** | `spo2_abs_change` | SpO₂ | % | $x_t - x_{t-1}$ |
| **20** | `spo2_rel_change` | SpO₂ | ratio | $(x_t - x_{t-1}) / \|x_{t-1}\|$ |
| **21** | `spo2_trend_slope` | SpO₂ | %/hr | $(x_t - x_{t-1}) / \Delta t_{\text{hours}}$ |
| **22** | `spo2_baseline_dev` | SpO₂ | % | $x_t - \text{Median}_{\text{base}}$ |
| **23** | `spo2_norm_baseline_dev`| SpO₂ | z-score | $(x_t - \text{Median}_{\text{base}}) / (1.4826 \times \text{MAD}_{\text{base}})$ |
| **24** | `temp_latest` | Temperature | °F | Latest valid reading in window |
| **25** | `temp_rolling_mean` | Temperature | °F | Mean over window |
| **26** | `temp_rolling_median` | Temperature | °F | Median over window |
| **27** | `temp_rolling_std` | Temperature | °F | Sample std dev ($N \ge 2$) |
| **28** | `temp_rolling_mad` | Temperature | °F | Median absolute deviation |
| **29** | `temp_window_min` | Temperature | °F | Minimum value in window |
| **30** | `temp_window_max` | Temperature | °F | Maximum value in window |
| **31** | `temp_abs_change` | Temperature | °F | $x_t - x_{t-1}$ |
| **32** | `temp_rel_change` | Temperature | ratio | $(x_t - x_{t-1}) / \|x_{t-1}\|$ |
| **33** | `temp_trend_slope` | Temperature | °F/hr | $(x_t - x_{t-1}) / \Delta t_{\text{hours}}$ |
| **34** | `temp_baseline_dev` | Temperature | °F | $x_t - \text{Median}_{\text{base}}$ |
| **35** | `temp_norm_baseline_dev`| Temperature | z-score | $(x_t - \text{Median}_{\text{base}}) / (1.4826 \times \text{MAD}_{\text{base}})$ |
| **36** | `cross_hypoxia_tachycardia_ratio` | Cross | ratio | $\text{HR} / \text{SpO}_2$ |
| **37** | `cross_vital_strain_index` | Cross | composite | $\frac{\text{HR}-80}{15} + \frac{\text{Temp}-98.6}{1.0} - \frac{\text{SpO}_2-97}{2.5}$ |
| **38** | `quality_sample_count` | Quality | count | Number of observations in current window |
| **39** | `quality_missing_count` | Quality | count | Number of missing metrics in reading |
| **40** | `quality_elapsed_minutes`| Quality | min | Elapsed time from prior reading |
| **41** | `quality_gap_exceeded` | Quality | binary | $1.0$ if $\Delta t > 180\text{m}$, else $0.0$ |
| **42** | `quality_has_baseline` | Quality | binary | $1.0$ if baseline established, else $0.0$ |

---

## 5. Automated Verification & Test Results

The test suite executed via `pytest` and standard Python `unittest`:

```bash
python -m pytest tests/ -v
# 20 passed in 1.03s

python -m unittest discover -s tests -p "test_*.py" -v
# 20 passed in 0.204s
```

### Verified Test Matrix:
1. `test_temperature_unit_conversions`: Exact conversion ($32^\circ\text{F} = 0^\circ\text{C}, 98.6^\circ\text{F} = 37.0^\circ\text{C}, 212^\circ\text{F} = 100^\circ\text{C}$), array and NaN preservation.
2. `test_hand_calculated_window_statistics`: Verified manual hand calculations for rolling mean, median, std, MAD, and trend slope.
3. `test_hand_calculated_end_to_end_observation`: End-to-end verification of 2-reading scenario.
4. `test_chronological_ordering_is_enforced`: Shuffled rows are automatically sorted chronologically.
5. `test_multiple_patients_never_share_baselines_or_windows`: Patient P101 and P102 state isolation verified.
6. `test_constant_measurements_and_zero_variability_baselines`: Zero-division safety verified when MAD and std are 0.
7. `test_missing_and_non_finite_readings`: Missing vitals flagged, not imputed with zero.
8. `test_irregular_sampling_intervals_and_gap_exceeded`: Trend slope accounts for $\Delta t$; gaps $> 180\text{m}$ decouple slopes.
9. `test_insufficient_history_behavior`: $k < 5$ returns $0.0$ deviation and `INSUFFICIENT_HISTORY` quality state.
10. `test_prevention_of_future_data_leakage`: Fit on training split leaves frozen baselines for future spikes.
11. `test_streaming_and_batch_equivalence`: `StreamingFeatureExtractor` and batch `FeatureExtractor` produce mathematically identical outputs.
12. `test_determinism_repeated_execution`: Bit-for-bit idempotence verified on repeated runs.
13. `test_actual_kaggle_raw_dataset_file`: Invariant checks on the raw 1,500-row CSV.

---

## 6. Real-World Execution on Kaggle Dataset

The feature extractor was executed against the entire 1,500-row historical CSV:
- **Execution Time:** **817.1 ms** (total for 1,500 rows)
- **Per-Record Latency:** **0.54 ms / record** (sub-millisecond, orders of magnitude below the 25ms real-time streaming budget)
- **Quality State Breakdown:** 1,500 `VALID` records (100% eligible for scoring).
- **Extracted Dataset:** Generated and saved to [`data/processed/features_kaggle_v1.csv`](file:///C:/Users/HARNOOR%20KANT/.gemini/antigravity/scratch/smart-health/data/processed/features_kaggle_v1.csv) (49 total columns).

---

## 7. Remaining Limitations

1. **Unsupervised Nature:** The historical CSV provides no ground-truth anomaly labels. Unsupervised Isolation Forest training will require synthetic anomaly injection fixtures to benchmark detection recall.
2. **Cold-Start for New Patients:** A patient's first 4 readings operate in `INSUFFICIENT_HISTORY` mode until their 5th reading establishes an individualized baseline. Deterministic clinical rules (e.g. fixed vital bounds) will safeguard patient triage during this cold-start interval.

# Risks, Assumptions & Unresolved Decisions Register

**System:** Smart Health Monitoring System  
**Component:** Member 3: ML Analysis, Integration, QA and Deployment  
**Version:** 3.0  
**Status:** Working Assessment Document  

---

## 1. Project Risks & Mitigation Matrix

| Risk ID | Category | Risk Description | Severity | Likelihood | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R-01** | **Evaluation & Viva** | **False Claims of Clinical Accuracy:** Claiming accuracy/recall on an unlabelled Kaggle dataset will cause immediate academic viva penalization. | **Critical** | High | Explicitly document that Isolation Forest is **unsupervised**. Test recall exclusively via **controlled synthetic anomaly injection fixtures**. Never claim clinical diagnostic efficacy. |
| **R-02** | **Data Leakage** | **Time-Series / Patient Data Leakage:** Using random train/test splits where adjacent readings from the same patient leak across partitions. | **High** | High | Enforce **Chronological Temporal Splits** (earlier 70% train / later 30% test) and **GroupKFold by patient_id**. Never use naive random `train_test_split()`. |
| **R-03** | **Feature Engineering** | **Sampling Gap Distortion:** Intervals between readings range from 5 minutes to over 5 hours (median ~35 mins). A row-count rolling window (`window=10`) spans wildly varying durations. | **High** | High | Implement **time-aware rolling windows** (e.g. `window='2h'`, `on='timestamp'`) rather than index-based row windows. Explicitly calculate `sampling_gap_minutes` as a feature. |
| **R-04** | **Cross-Member Drift** | **Temperature Unit Inconsistency:** Frontend displaying Celsius while Backend stores Fahrenheit or vice versa. | **Medium** | Medium | Maintain raw Fahrenheit in the feature pipeline. Expose clear conversion helpers (`(T - 32) * 5/9`) with unit suffixes (`temperature_f`, `temperature_c`) in contracts. |
| **R-05** | **Dataset Boundary** | **Fabricating Respiratory Rate / BP:** Accidentally imputing or simulating missing vitals in historical records to match simulator features. | **High** | Medium | Strict validation checks in `validation.py` rejecting or flagging any historical record with non-null BP or respiratory rate. |
| **R-06** | **Latency & Performance** | **Cold-Start / Inference Overhead:** Excessive latency (>200ms) during real-time event ingestion when recalculating rolling features. | **Medium** | Low | Maintain an in-memory patient circular buffer ($N=50$ most recent readings) for $O(1)$ sliding window statistics. Target < 25ms inference latency. |

---

## 2. Core Working Assumptions

1. **Dataset Integrity:**
   - The verified 1,500 rows in `kaggle_health_dataset.csv` represent the definitive historical ground truth for historical baseline demonstrations.
   - All 5 patients (`P101`–`P105`) represent distinct monitoring subjects.
2. **Timestamp Policy:**
   - Timestamps are naive wall-clock time (`YYYY-MM-DD HH:MM:SS`). They will not be transformed to UTC with arbitrary timezone offsets, preserving historical consistency.
3. **Role of Machine Learning:**
   - Machine learning is a **decision-support heuristic**, functioning alongside deterministic rule-based triggers (e.g. SpO₂ < 90% or Heart Rate > 120 bpm).
   - In any critical event, deterministic clinical safety rules take precedence over statistical model predictions.
4. **Architectural Handoff:**
   - Member 1 will invoke `packages/health_analysis` synchronously or asynchronously via a stable Python interface (`assess_event()`), persisting assessments into MySQL `model_assessments` and mirroring to MongoDB Atlas.

---

## 3. Unresolved Decisions & Open Questions

| Item | Topic | Current Recommendation | Trade-off / Decision Needed from Team |
| :--- | :--- | :--- | :--- |
| **UD-01** | **Contamination Rate ($c$) in Isolation Forest** | Set default to `0.05` (5%). | Lower contamination (0.01) generates fewer false alarms but may miss subtle shifts; higher (0.10) increases sensitivity but may generate alert fatigue. To be tuned using the synthetic injection fixture. |
| **UD-02** | **Inference Execution Timing** | Ingest event -> Compute features -> Execute inference on every single incoming event. | Real-time event scoring provides immediate UI feedback (<50ms) but incurs per-event compute. Alternative is batched 10-minute window scoring. **Recommendation:** Real-time scoring using in-memory sliding buffer. |
| **UD-03** | **Minimum History Threshold for Baselines** | Require at least $k = 5$ observations before calculating patient baseline deviations. | When $k < 5$, baseline deviation is set to `0.0` and marked with `insufficient_history` flag. |
| **UD-04** | **Temperature Storage in MongoDB Mirror** | Mirror both `temperature_f` and derived `temperature_c` in `health_events.measurements`. | Simplifies Streamlit and React frontend queries without requiring frontend arithmetic. |

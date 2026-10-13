SMART HEALTH MONITORING SYSTEM — EXPANDED DA3 TEAM PACK
Version 2.0 — 13 October 2026

This pack retains the original project implementation handbooks and adds a detailed DA3 implementation addendum to every document.

FILES
00_Master_Three_Member_Project_Plan.docx
  Team architecture, source-of-truth policy, requirement-to-evidence matrix, owners, milestone gates, repository layout, environment variables, benchmark protocol and submission checklist.
01_Member_1_Simulator_Backend_MySQL.docx
  Backend/simulator/MySQL ownership plus MongoDB Atlas design, CRUD, indexes, explain plans, mirror/retry strategy and DB/API access control.
02_Member_2_Frontend_Applications.docx
  Flutter and React integration, API contract handoff, stale/disconnected states, authorization, reliability and UI evidence.
03_Member_3_ML_Integration_Deployment.docx
  ML integration plus Mongo aggregation, SQL-vs-NoSQL benchmarking, Streamlit analytics, backup/restore, end-to-end tests and release evidence.
04_Technical_Handbook_and_Viva.docx
  SQL-vs-NoSQL architecture rationale, DA3 definitions, viva questions and demonstration sequence.

ARCHITECTURE DECISION
- MySQL remains authoritative for users, roles, assignments, normalized canonical readings, alerts, messages and audit history.
- MongoDB Atlas is the DA3 document database for a derived/mirrored synthetic health-event stream and optional model-assessment snapshots.
- FastAPI is the only gateway used by Flutter and React. Frontends never connect directly to either database.
- A stable source_event_id makes mirroring idempotent. Do not claim atomic cross-database writes; use a retryable pending/outbox mechanism.
- Streamlit is the recommended DA3 analytics/benchmark dashboard. The course plan allows Power BI or Streamlit and lists multiple cloud options; MongoDB Atlas is this team's recommended implementation choice, not a stated course mandate.

IMPORTANT
All health readings must be marked SIMULATED. This is not a medical device or clinically validated system.
Never commit .env files, database connection strings, tokens, backup dumps, or real personal/medical data.
Benchmark values must come from actual measured runs; do not invent or pre-fill results.
Atlas feature availability depends on the selected tier. Document what was actually configured and tested.

SUGGESTED FIRST ACTIONS
1. Team reads the master plan and agrees to the architecture/source-of-truth policy.
2. Member 1 commits event contract, MySQL schema and Mongo document example.
3. Member 3 prepares the benchmark/aggregation/restore evidence folders and Streamlit skeleton.
4. Member 2 integrates only against the FastAPI contract.
5. Team runs the gates in the master plan and saves evidence continuously.


DATASET ALIGNMENT UPDATE — VERSION 3.0
The uploaded CSV is the dataset basis for historical records. Its observed schema is:
  patient_id, heart_rate, spo2, temperature_f, timestamp
Rows: 1500
Patient IDs/counts: {'P101': 293, 'P102': 302, 'P103': 296, 'P104': 325, 'P105': 284}
Timestamp strings: 2023-10-01 08:13:00 to 2023-10-11 13:35:00
No missing values in the supplied columns; no duplicate full rows or duplicate patient_id+timestamp pairs.
Observed source ranges:
  heart_rate: 60–135 bpm
  spo2: 88.1–99.9 %
  temperature_f: 97.5–103.0 °F

IMPORTANT DATASET BOUNDARIES
- Historical Kaggle-backed records contain only heart_rate, spo2 and temperature_f.
- Respiratory rate and systolic/diastolic BP remain optional features of the pre-existing live simulator only; do not invent them for CSV history.
- The CSV timestamps do not specify a timezone. Document an explicit source-timezone assumption before converting to UTC; otherwise preserve the source timestamp and label the timezone unknown.
- The CSV has no diagnostic/anomaly labels. Isolation Forest is unsupervised demonstration only; do not claim diagnostic accuracy.
- The CSV's clinical provenance is not established by its columns. Keep NOT FOR MEDICAL USE throughout; label live simulator records SIMULATED and imported records as Kaggle historical data with provenance unverified until the source page is checked.
- MySQL remains the authoritative store; MongoDB Atlas remains the DA3 analytics mirror. The two database writes are not one atomic transaction; use idempotency and retry/outbox.
- Benchmark the actual 1,500 source rows first. A 10,000+ row generated/scaled experiment must be reported separately from the Kaggle dataset.
- Preserve the original CSV, record a checksum, create a repeatable importer, and verify 1,500 source events / 4,500 normalized metric rows if using one MySQL metric row per source metric.
- Do not commit secrets or database dumps. Check Kaggle license before redistributing the raw CSV in a public repository.

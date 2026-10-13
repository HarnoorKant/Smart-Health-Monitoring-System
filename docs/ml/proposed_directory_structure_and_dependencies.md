# Proposed Directory Structure & Dependency Requirements

**System:** Smart Health Monitoring System  
**Component:** Member 3: ML Analysis, Integration, QA and Deployment  
**Version:** 3.0  
**Status:** Approved Layout Specification  

---

## 1. Project Directory Structure

The structure directly adheres to the Master Three-Member Coordination Plan (`00_Master_Three_Member_Project_Plan.docx`), with the ML subsystem isolated in `packages/health_analysis/`:

```
smart-health/
├── .gitignore                          # Ignores .venv, .env, __pycache__, dumps, .pytest_cache
├── README.md                           # Master project guide and setup instructions
├── requirements.txt                    # Python dependencies
├── pytest.ini                          # Pytest configuration with pythonpath = packages
│
├── data/
│   └── raw/
│       └── kaggle_health_dataset.csv   # Verified 1,500-row historical CSV (SHA-256 verified)
│
├── packages/
│   ├── health_analysis/                # [MEMBER 3 OWNED] ML Analysis Package
│   │   ├── __init__.py                 # Package version and export definitions
│   │   ├── validation.py               # Dataset audit, schema checking, data profiling
│   │   ├── features.py                 # [Next Phase] Time-aware rolling features, MAD, trend
│   │   ├── models.py                   # [Next Phase] Isolation Forest wrapper & scoring
│   │   └── pipeline.py                 # [Next Phase] End-to-end event assessment interface
│   └── contracts/                      # Pydantic schemas shared across services
│       └── __init__.py
│
├── apps/
│   ├── analytics_streamlit/            # [MEMBER 3 OWNED] DA3 Analytics Dashboard
│   ├── doctor_web/                     # [MEMBER 2 OWNED] React + TypeScript Dashboard
│   └── patient_flutter/                # [MEMBER 2 OWNED] Flutter Patient App
│
├── services/
│   ├── api/                            # [MEMBER 1 OWNED] FastAPI backend gateway
│   ├── simulator/                      # [MEMBER 1 OWNED] Synthetic telemetry generator
│   └── retry_mirror/                   # [MEMBER 1 OWNED] MySQL-to-Mongo idempotent mirror
│
├── database/
│   ├── mysql/                          # [MEMBER 1 OWNED] Migrations, seeds, procedures, queries
│   └── mongo/                          # [MEMBER 1 & 3] Collections, indexes, aggregations
│
├── scripts/
│   ├── audit_dataset.py                # [MEMBER 3 OWNED] Reproducible dataset audit tool
│   ├── benchmark_sql_nosql.py          # [MEMBER 3 OWNED] DA3 performance comparison runner
│   ├── backup_restore_test.py          # [MEMBER 3 OWNED] Backup/recovery verification
│   └── smoke_test.py                   # Health check runner
│
├── tests/
│   ├── conftest.py
│   ├── test_data_validation.py         # Unit tests for dataset validation & profiling
│   ├── test_features.py                # [Next Phase] Unit tests for feature extraction
│   ├── test_isolation_forest.py        # [Next Phase] Unit tests for anomaly scoring
│   └── test_synthetic_injection.py     # [Next Phase] Synthetic anomaly recall verification
│
└── docs/
    ├── architecture/                   # System diagrams and API contracts
    └── ml/                             # [MEMBER 3 OWNED] ML Specifications and Evidence
        ├── dataset_profiling_report.md
        ├── dataset_profile.json
        ├── data_dictionary.md
        ├── ml_architecture_and_feature_schema.md
        ├── risks_assumptions_decisions.md
        ├── proposed_directory_structure_and_dependencies.md
        └── implementation_checklist.md
```

---

## 2. Structural Compatibility Analysis

1. **Intended ML Package Location:** `packages/health_analysis/`.
2. **Prior State of the Repository:**
   - Prior to this phase, no code repository existed in the workspace; only the unzipped documentation pack (`D:\Downloads\Smart_Health_Kaggle_Dataset_Aligned_Documentation_Pack`) and loose files in `D:\Downloads\` were present.
   - Establishing `packages/health_analysis/` within `smart-health/` establishes the exact layout mandated by the Master Coordination Plan without requiring migrations, renames, or breaking changes to other components.
3. **Isolation & Clean Handoff:**
   - Member 1's backend imports `health_analysis` as a pure Python module via `sys.path` or editable install (`pip install -e packages/health_analysis`).
   - Frontends interact solely with FastAPI endpoints (`/api/v1/...`) and remain insulated from internal ML classes.

---

## 3. Dependency Requirements & Environment

### Python Runtime
- **Target Version:** Python 3.12 (Installed: Python 3.12.10 on Windows).

### Core Package Manifest (`requirements.txt`)
```text
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.12.0
scikit-learn>=1.4.0
pytest>=8.0.0
streamlit>=1.35.0
pymongo>=4.6.0
```

### Virtual Environment Setup Protocol
```bash
# Navigate to project root
cd "C:\Users\HARNOOR KANT\.gemini\antigravity\scratch\smart-health"

# Create virtual environment (optional/isolated)
python -m venv .venv

# Activate
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

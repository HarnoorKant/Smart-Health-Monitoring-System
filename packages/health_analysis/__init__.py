"""
health_analysis: Smart Health Monitoring System ML & Analytics Package.
Member 3: ML Analysis, Integration, QA and Deployment.
"""

from health_analysis.units import (
    fahrenheit_to_celsius,
    celsius_to_fahrenheit,
    normalize_temperature,
)
from health_analysis.schema import (
    FEATURE_SCHEMA_VERSION,
    PHYSIOLOGICAL_FEATURE_NAMES_V1,
    DATA_QUALITY_FEATURE_NAMES_V1,
    ALL_FEATURE_NAMES_V1,
    DataQualityState,
    FeatureExtractorConfig,
    FeatureRecord,
)
from health_analysis.baseline import (
    MetricBaseline,
    PatientBaseline,
    BaselineManager,
    IncrementalBaselineTracker,
    calculate_metric_baseline,
)
from health_analysis.features import (
    FeatureExtractor,
    StreamingFeatureExtractor,
    compute_window_statistics,
)
from health_analysis.validation import (
    audit_dataset_file,
    audit_dataframe,
    EXPECTED_COLUMNS,
    FORBIDDEN_HISTORICAL_COLUMNS,
)
from health_analysis.split import (
    SplitMetadata,
    chronological_patient_split,
    leave_patient_out_split,
)
from health_analysis.models import (
    ModelConfig,
    AssessmentOutput,
    IsolationForestDetector,
)
from health_analysis.rules import (
    RuleFinding,
    evaluate_safety_rules,
)
from health_analysis.service import (
    AssessmentResult,
    HealthAnalysisService,
    get_analysis_service,
    assess_health_event,
)

__version__ = "0.1.0"

__all__ = [
    # Units
    "fahrenheit_to_celsius",
    "celsius_to_fahrenheit",
    "normalize_temperature",
    # Schema
    "FEATURE_SCHEMA_VERSION",
    "PHYSIOLOGICAL_FEATURE_NAMES_V1",
    "DATA_QUALITY_FEATURE_NAMES_V1",
    "ALL_FEATURE_NAMES_V1",
    "DataQualityState",
    "FeatureExtractorConfig",
    "FeatureRecord",
    # Baseline
    "MetricBaseline",
    "PatientBaseline",
    "BaselineManager",
    "IncrementalBaselineTracker",
    "calculate_metric_baseline",
    # Features
    "FeatureExtractor",
    "StreamingFeatureExtractor",
    "compute_window_statistics",
    # Validation
    "audit_dataset_file",
    "audit_dataframe",
    "EXPECTED_COLUMNS",
    "FORBIDDEN_HISTORICAL_COLUMNS",
    # Split
    "SplitMetadata",
    "chronological_patient_split",
    "leave_patient_out_split",
    # Models
    "ModelConfig",
    "AssessmentOutput",
    "IsolationForestDetector",
    # Rules
    "RuleFinding",
    "evaluate_safety_rules",
    # Service Integration Interface
    "AssessmentResult",
    "HealthAnalysisService",
    "get_analysis_service",
    "assess_health_event",
]

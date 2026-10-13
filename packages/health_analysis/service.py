"""
Health Analysis Service and Integration Interface.
Member 3: ML Analysis, Integration, QA and Deployment.

Provides the canonical, importable assessment interface for the Smart Health
Monitoring System backend without depending on FastAPI request objects or MySQL models.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from health_analysis.baseline import BaselineManager
from health_analysis.features import StreamingFeatureExtractor
from health_analysis.models import IsolationForestDetector, ModelConfig
from health_analysis.schema import (
    FEATURE_SCHEMA_VERSION,
    PHYSIOLOGICAL_FEATURE_NAMES_V1,
    DataQualityState,
    FeatureExtractorConfig,
    FeatureRecord,
)

logger = logging.getLogger("health_analysis.service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class AssessmentResult:
    """
    Agreed assessment result contract matching Member 1's relational schema
    (model_assessments table) and MongoDB Atlas collection.
    """
    model_version: str
    score: float  # [0.0, 1.0] where higher is more anomalous
    result: str  # "normal_like" | "anomalous" | "insufficient_data"
    feature_summary: Dict[str, Any]
    time_window_start: datetime
    time_window_end: datetime
    explanation: List[str]
    patient_id: str
    is_simulated: bool = True
    assessment_id: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a clean JSON-serializable dictionary."""
        return {
            "assessment_id": self.assessment_id,
            "patient_id": self.patient_id,
            "model_version": self.model_version,
            "score": round(float(self.score), 4),
            "result": self.result,
            "feature_summary": self.feature_summary,
            "time_window_start": self.time_window_start.isoformat(),
            "time_window_end": self.time_window_end.isoformat(),
            "explanation": self.explanation,
            "is_simulated": self.is_simulated,
            "latency_ms": round(float(self.latency_ms), 2),
        }


class HealthAnalysisService:
    """
    Importable ML analysis service providing real-time telemetry assessment.
    Encapsulates feature extraction, frozen baseline management, model scoring,
    and simulator compatibility.
    """

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        feature_config: Optional[FeatureExtractorConfig] = None,
        baseline_manager: Optional[BaselineManager] = None,
    ):
        self.artifact_dir = artifact_dir
        self.feature_config = feature_config or FeatureExtractorConfig()
        self.baseline_manager = baseline_manager
        self.detector: Optional[IsolationForestDetector] = None

        # Streaming feature extractor instance
        self.streaming_extractor = StreamingFeatureExtractor(
            config=self.feature_config,
            baseline_manager=self.baseline_manager,
        )

        # Attempt to load model if artifact directory is provided
        if self.artifact_dir:
            self._load_model_safe(self.artifact_dir)

    def _load_model_safe(self, artifact_dir: str) -> bool:
        """Safely load model artifacts with error handling and logging."""
        try:
            if not os.path.exists(artifact_dir):
                logger.warning(f"Model artifact directory not found at {artifact_dir}")
                return False
            self.detector = IsolationForestDetector.load(artifact_dir)
            logger.info(
                f"Loaded IsolationForest model version {self.detector.config.model_version} "
                f"from {artifact_dir}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load Isolation Forest artifact from {artifact_dir}: {e}", exc_info=True)
            self.detector = None
            return False

    def assess_telemetry(
        self,
        patient_id: str,
        timestamp: Union[str, datetime, pd.Timestamp],
        heart_rate: Optional[float],
        spo2: Optional[float],
        temperature_f: Optional[float],
        # Simulator optional extensions (handled safely without model contamination)
        respiratory_rate: Optional[float] = None,
        systolic_bp: Optional[float] = None,
        diastolic_bp: Optional[float] = None,
        source_event_id: Optional[str] = None,
        is_simulated: bool = True,
    ) -> AssessmentResult:
        """
        Main assessment entrypoint called by backend during event ingestion.

        Guarantees:
        - Never raises uncaught exceptions.
        - Returns 'insufficient_data' if inputs are invalid or history is too short.
        - Never pollutes the 3-metric feature vector with live simulator BP/respiratory rate.
        - Returns fully JSON-serializable AssessmentResult.
        """
        t0 = time.time()
        pid = str(patient_id).strip()

        # Parse timestamp safely
        try:
            ts_parsed = pd.to_datetime(timestamp, errors="coerce")
            if pd.isnull(ts_parsed):
                raise ValueError("Invalid timestamp value")
            ts_dt = ts_parsed.to_pydatetime()
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        except Exception:
            window_end = datetime.now(timezone.utc)
            return AssessmentResult(
                model_version=self.detector.config.model_version if self.detector else "unknown",
                score=0.0,
                result="insufficient_data",
                feature_summary={"error": "invalid_timestamp"},
                time_window_start=window_end,
                time_window_end=window_end,
                explanation=["Timestamp is invalid or malformed."],
                patient_id=pid,
                is_simulated=is_simulated,
                assessment_id=f"asmt_{source_event_id}" if source_event_id else None,
                latency_ms=(time.time() - t0) * 1000.0,
            )

        window_start = ts_dt - pd.Timedelta(minutes=self.feature_config.window_duration_minutes).to_pytimedelta()

        # Validate core inputs
        core_missing = []
        if heart_rate is None or np.isnan(heart_rate) or np.isinf(heart_rate):
            core_missing.append("heart_rate")
        if spo2 is None or np.isnan(spo2) or np.isinf(spo2):
            core_missing.append("spo2")
        if temperature_f is None or np.isnan(temperature_f) or np.isinf(temperature_f):
            core_missing.append("temperature_f")

        if core_missing:
            return AssessmentResult(
                model_version=self.detector.config.model_version if self.detector else "unknown",
                score=0.0,
                result="insufficient_data",
                feature_summary={"missing_metrics": core_missing},
                time_window_start=window_start,
                time_window_end=ts_dt,
                explanation=[f"Required measurement(s) missing or non-finite: {', '.join(core_missing)}."],
                patient_id=pid,
                is_simulated=is_simulated,
                assessment_id=f"asmt_{source_event_id}" if source_event_id else None,
                latency_ms=(time.time() - t0) * 1000.0,
            )

        # Check model availability
        if self.detector is None or not self.detector.is_fitted:
            return AssessmentResult(
                model_version="unavailable",
                score=0.0,
                result="insufficient_data",
                feature_summary={"error": "model_not_loaded"},
                time_window_start=window_start,
                time_window_end=ts_dt,
                explanation=["ML anomaly detection model is not loaded or unavailable."],
                patient_id=pid,
                is_simulated=is_simulated,
                assessment_id=f"asmt_{source_event_id}" if source_event_id else None,
                latency_ms=(time.time() - t0) * 1000.0,
            )

        # Process reading through streaming extractor
        # (Respiratory rate and BP are intentionally omitted from historical model vector)
        try:
            record: FeatureRecord = self.streaming_extractor.process_reading(
                patient_id=pid,
                timestamp=ts_parsed,
                heart_rate=heart_rate,
                spo2=spo2,
                temperature_f=temperature_f,
            )
        except Exception as e:
            logger.error(f"Error during feature extraction for patient {pid}: {e}", exc_info=True)
            return AssessmentResult(
                model_version=self.detector.config.model_version,
                score=0.0,
                result="insufficient_data",
                feature_summary={"error": "feature_extraction_failure"},
                time_window_start=window_start,
                time_window_end=ts_dt,
                explanation=["Feature extraction failed due to internal error."],
                patient_id=pid,
                is_simulated=is_simulated,
                assessment_id=f"asmt_{source_event_id}" if source_event_id else None,
                latency_ms=(time.time() - t0) * 1000.0,
            )

        # Handle data quality / insufficient history states
        if record.quality_state == DataQualityState.INSUFFICIENT_HISTORY:
            return AssessmentResult(
                model_version=self.detector.config.model_version,
                score=0.0,
                result="insufficient_data",
                feature_summary={
                    "heart_rate": heart_rate,
                    "spo2": spo2,
                    "temperature_f": temperature_f,
                    "history_samples_observed": record.features.get("quality_sample_count", 0),
                    "min_required_samples": self.feature_config.min_baseline_samples,
                },
                time_window_start=window_start,
                time_window_end=ts_dt,
                explanation=["Insufficient historical observations (< 5 samples) to establish individualized patient baseline."],
                patient_id=pid,
                is_simulated=is_simulated,
                assessment_id=f"asmt_{source_event_id}" if source_event_id else None,
                latency_ms=(time.time() - t0) * 1000.0,
            )

        if not record.is_eligible_for_scoring or record.quality_state != DataQualityState.VALID:
            return AssessmentResult(
                model_version=self.detector.config.model_version,
                score=0.0,
                result="insufficient_data",
                feature_summary={"quality_state": record.quality_state.value, "flags": record.quality_flags},
                time_window_start=window_start,
                time_window_end=ts_dt,
                explanation=[f"Data quality preconditions not satisfied: {'; '.join(record.quality_flags)}."],
                patient_id=pid,
                is_simulated=is_simulated,
                assessment_id=f"asmt_{source_event_id}" if source_event_id else None,
                latency_ms=(time.time() - t0) * 1000.0,
            )

        # Model Assessment
        try:
            model_assessment = self.detector.assess_record(record)
        except Exception as e:
            logger.error(f"Error during model scoring for patient {pid}: {e}", exc_info=True)
            return AssessmentResult(
                model_version=self.detector.config.model_version,
                score=0.0,
                result="insufficient_data",
                feature_summary={"error": "model_scoring_failure"},
                time_window_start=window_start,
                time_window_end=ts_dt,
                explanation=["Isolation Forest inference failed due to internal error."],
                patient_id=pid,
                is_simulated=is_simulated,
                assessment_id=f"asmt_{source_event_id}" if source_event_id else None,
                latency_ms=(time.time() - t0) * 1000.0,
            )

        score = float(model_assessment.anomaly_score)
        result_str = "anomalous" if model_assessment.is_anomaly else "normal_like"

        # Construct explanations based on actual feature evidence
        explanations = self._build_evidence_explanations(
            record=record,
            assessment=model_assessment,
            is_anomalous=model_assessment.is_anomaly,
        )

        # Compact feature summary for database storage
        summary = {
            "heart_rate_latest": round(float(heart_rate), 1),
            "spo2_latest": round(float(spo2), 1),
            "temperature_f_latest": round(float(temperature_f), 1),
            "heart_rate_rolling_mean": round(record.features.get("hr_rolling_mean", 0.0), 1),
            "spo2_rolling_min": round(record.features.get("spo2_window_min", 0.0), 1),
            "temp_rolling_max": round(record.features.get("temp_window_max", 0.0), 1),
            "vital_strain_index": round(record.features.get("cross_vital_strain_index", 0.0), 2),
            "window_sample_count": int(record.features.get("quality_sample_count", 1)),
        }

        # Include simulator context if supplied (for traceability only)
        if respiratory_rate is not None:
            summary["respiratory_rate_unmodeled"] = round(float(respiratory_rate), 1)
        if systolic_bp is not None and diastolic_bp is not None:
            summary["blood_pressure_unmodeled"] = f"{int(systolic_bp)}/{int(diastolic_bp)}"

        elapsed_ms = (time.time() - t0) * 1000.0

        return AssessmentResult(
            model_version=self.detector.config.model_version,
            score=round(score, 4),
            result=result_str,
            feature_summary=summary,
            time_window_start=window_start,
            time_window_end=ts_dt,
            explanation=explanations,
            patient_id=pid,
            is_simulated=is_simulated,
            assessment_id=f"asmt_{source_event_id}" if source_event_id else None,
            latency_ms=round(elapsed_ms, 2),
        )

    def _build_evidence_explanations(
        self,
        record: FeatureRecord,
        assessment: Any,
        is_anomalous: bool,
    ) -> List[str]:
        """Generate concise, factual evidence strings without inventing diagnoses."""
        if not is_anomalous:
            return ["All vital signs and rolling metrics are within normative baseline clusters."]

        explanations: List[str] = []
        for contrib in assessment.top_contributing_features:
            feat = contrib["feature"]
            val = contrib["value"]
            med = contrib["baseline_median"]
            dev = contrib["robust_deviation"]

            if "spo2" in feat:
                explanations.append(
                    f"SpO2 metric ({feat}) measured {val}% vs historical baseline median of {med}% ({dev:+.1f}x MAD)."
                )
            elif "hr" in feat:
                explanations.append(
                    f"Heart rate metric ({feat}) measured {val} bpm vs historical baseline median of {med} bpm ({dev:+.1f}x MAD)."
                )
            elif "temp" in feat:
                explanations.append(
                    f"Temperature metric ({feat}) measured {val} F vs historical baseline median of {med} F ({dev:+.1f}x MAD)."
                )
            elif "cross" in feat:
                explanations.append(
                    f"Multivariate vital strain index elevated to {val} (composite strain deviation)."
                )

        if not explanations:
            explanations.append("Multivariate anomaly detected across combined physiological feature vector.")

        return explanations


# Module-level singleton accessor
_GLOBAL_SERVICE: Optional[HealthAnalysisService] = None


def get_analysis_service(
    artifact_dir: Optional[str] = None,
    baseline_manager: Optional[BaselineManager] = None,
) -> HealthAnalysisService:
    """Return the initialized HealthAnalysisService instance."""
    global _GLOBAL_SERVICE
    if _GLOBAL_SERVICE is None:
        target_dir = artifact_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models", "iforest_v1"
        )
        _GLOBAL_SERVICE = HealthAnalysisService(
            artifact_dir=target_dir,
            baseline_manager=baseline_manager,
        )
    return _GLOBAL_SERVICE


def assess_health_event(
    patient_id: str,
    timestamp: Union[str, datetime, pd.Timestamp],
    heart_rate: Optional[float],
    spo2: Optional[float],
    temperature_f: Optional[float],
    respiratory_rate: Optional[float] = None,
    systolic_bp: Optional[float] = None,
    diastolic_bp: Optional[float] = None,
    source_event_id: Optional[str] = None,
    is_simulated: bool = True,
    service: Optional[HealthAnalysisService] = None,
) -> AssessmentResult:
    """
    Convenience functional API for single event assessment.
    """
    svc = service or get_analysis_service()
    return svc.assess_telemetry(
        patient_id=patient_id,
        timestamp=timestamp,
        heart_rate=heart_rate,
        spo2=spo2,
        temperature_f=temperature_f,
        respiratory_rate=respiratory_rate,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        source_event_id=source_event_id,
        is_simulated=is_simulated,
    )

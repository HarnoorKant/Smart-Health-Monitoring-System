"""
Isolation Forest unsupervised anomaly detector for health telemetry.
Member 3: ML Analysis, Integration, QA and Deployment.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from health_analysis.schema import (
    FEATURE_SCHEMA_VERSION,
    PHYSIOLOGICAL_FEATURE_NAMES_V1,
    DataQualityState,
    FeatureRecord,
)


@dataclass
class ModelConfig:
    """Hyperparameters and configuration for Isolation Forest Anomaly Detector."""
    model_version: str = "1.0.0"
    n_estimators: int = 100
    max_samples: Union[int, float, str] = 256
    contamination: float = 0.05
    max_features: float = 1.0
    random_state: int = 42
    # Threshold in [0.0, 1.0] above which an observation is flagged as an anomaly
    decision_threshold: float = 0.50
    # Sigmoid temperature scaling factor to map raw -decision_function to [0, 1]
    score_scaling_temperature: float = 0.15
    # Minimum training samples required
    min_training_samples: int = 50
    # Feature schema version and unit conventions
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    temperature_unit: str = "F"
    feature_names: List[str] = field(default_factory=lambda: list(PHYSIOLOGICAL_FEATURE_NAMES_V1))


@dataclass
class AssessmentOutput:
    """Standardized output for an assessed telemetry event."""
    patient_id: str
    timestamp: str
    model_version: str
    anomaly_score: float  # [0.0, 1.0] where higher is more anomalous
    is_anomaly: bool  # True if anomaly_score >= decision_threshold
    result_status: str  # "normal_like" | "elevated_variability" | "mild_anomaly" | "severe_anomaly" | "insufficient_history"
    raw_decision_score: float  # Scikit-learn native decision_function value
    top_contributing_features: List[Dict[str, Any]]
    quality_state: str
    is_simulated: bool = True
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IsolationForestDetector:
    """
    Unsupervised Anomaly Detector wrapping scikit-learn's Isolation Forest.
    Calibrates score to [0.0, 1.0] where higher indicates higher anomaly confidence,
    extracts top contributing features for clinical explainability, and enforces
    strict feature-order invariance.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model: Optional[IsolationForest] = None
        self.is_fitted: bool = False
        self.training_feature_medians: Dict[str, float] = {}
        self.training_feature_mads: Dict[str, float] = {}
        self.training_metadata: Dict[str, Any] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "IsolationForestDetector":
        """
        Train the Isolation Forest model on eligible feature matrices.
        """
        feats = feature_names or self.config.feature_names

        if isinstance(X, pd.DataFrame):
            missing_cols = set(feats) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Training DataFrame missing required feature columns: {missing_cols}")
            X_arr = X[feats].to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.shape[1] != len(feats):
                raise ValueError(
                    f"Feature dimension mismatch: expected {len(feats)} features, got {X_arr.shape[1]}"
                )

        # Validate finite values
        if np.isnan(X_arr).any() or np.isinf(X_arr).any():
            raise ValueError("Training matrix contains NaN or Infinite values. Clean input before fitting.")

        n_samples = X_arr.shape[0]
        if n_samples < self.config.min_training_samples:
            raise ValueError(
                f"Insufficient training samples: got {n_samples}, requires at least {self.config.min_training_samples}"
            )

        # Determine max_samples safely
        max_samples_val = self.config.max_samples
        if isinstance(max_samples_val, int) and max_samples_val > n_samples:
            max_samples_val = n_samples

        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=max_samples_val,
            contamination=self.config.contamination,
            max_features=self.config.max_features,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        self.model.fit(X_arr)
        self.is_fitted = True

        # Compute training medians and MADs for explainability attribution
        self.training_feature_medians.clear()
        self.training_feature_mads.clear()
        for i, col in enumerate(feats):
            col_vals = X_arr[:, i]
            med = float(np.median(col_vals))
            mad = float(np.median(np.abs(col_vals - med)))
            self.training_feature_medians[col] = med
            self.training_feature_mads[col] = mad if mad > 1e-6 else 1e-3

        self.training_metadata = {
            "training_samples": n_samples,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "extra": metadata or {},
        }

        return self

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score a batch of samples.
        Returns:
            (anomaly_scores, raw_decision_functions)
            where anomaly_scores is in [0.0, 1.0] and higher indicates greater anomalousness.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() or load() before scoring.")

        feats = self.config.feature_names
        if isinstance(X, pd.DataFrame):
            missing_cols = set(feats) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Input DataFrame missing required feature columns: {missing_cols}")
            X_arr = X[feats].to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        # Scikit-learn decision_function: lower means more anomalous
        # decision_function ~ 0 at contamination decision boundary
        raw_decisions = self.model.decision_function(X_arr)

        # Map raw decision function so higher means more anomalous
        # Negate: raw_inverted > 0 means anomalous, < 0 means normal
        raw_inverted = -raw_decisions

        # Sigmoid calibration: 1 / (1 + exp(-raw_inverted / temperature))
        # At decision_function = 0: raw_inverted = 0 -> anomaly_score = 0.50
        temp = self.config.score_scaling_temperature
        calibrated_scores = 1.0 / (1.0 + np.exp(-raw_inverted / temp))

        return calibrated_scores, raw_decisions

    def assess_record(self, record: FeatureRecord) -> AssessmentOutput:
        """
        Evaluate a single telemetry feature record in real-time.
        Emits an AssessmentOutput matching the handoff contract.
        """
        t0 = time.time()
        pid = record.patient_id
        ts = record.timestamp

        # Check eligibility and history
        if not record.is_eligible_for_scoring or record.quality_state != DataQualityState.VALID:
            return AssessmentOutput(
                patient_id=pid,
                timestamp=ts,
                model_version=self.config.model_version,
                anomaly_score=0.0,
                is_anomaly=False,
                result_status="insufficient_history" if record.quality_state == DataQualityState.INSUFFICIENT_HISTORY else "data_quality_error",
                raw_decision_score=0.0,
                top_contributing_features=[],
                quality_state=record.quality_state.value,
                is_simulated=True,
                latency_ms=(time.time() - t0) * 1000.0,
            )

        # Extract ordered feature vector
        vector = record.to_feature_vector(self.config.feature_names)
        X_mat = np.array([vector], dtype=float)

        scores, raw_decisions = self.score_samples(X_mat)
        anomaly_score = float(scores[0])
        raw_dec = float(raw_decisions[0])

        is_anom = anomaly_score >= self.config.decision_threshold

        # Result status categorization
        if anomaly_score < 0.40:
            status = "normal_like"
        elif anomaly_score < 0.50:
            status = "elevated_variability"
        elif anomaly_score < 0.70:
            status = "mild_anomaly"
        else:
            status = "severe_anomaly"

        # Attribute top contributing features
        top_features = self._extract_top_contributions(vector)

        elapsed_ms = (time.time() - t0) * 1000.0

        return AssessmentOutput(
            patient_id=pid,
            timestamp=ts,
            model_version=self.config.model_version,
            anomaly_score=round(anomaly_score, 4),
            is_anomaly=is_anom,
            result_status=status,
            raw_decision_score=round(raw_dec, 4),
            top_contributing_features=top_features,
            quality_state=record.quality_state.value,
            is_simulated=True,
            latency_ms=round(elapsed_ms, 2),
        )

    def _extract_top_contributions(self, feature_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Identify which features deviate most from normative training baseline."""
        contributions = []
        for i, feat_name in enumerate(self.config.feature_names):
            val = feature_vector[i]
            med = self.training_feature_medians.get(feat_name, 0.0)
            mad = self.training_feature_mads.get(feat_name, 1.0)
            robust_z = abs(val - med) / (mad * 1.4826)
            contributions.append({
                "feature": feat_name,
                "value": round(val, 2),
                "baseline_median": round(med, 2),
                "robust_deviation": round(robust_z, 2),
            })

        contributions.sort(key=lambda x: x["robust_deviation"], reverse=True)
        return contributions[:top_k]

    def save(self, artifact_dir: str) -> None:
        """
        Persist trained model weights and machine-readable metadata.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted model.")

        os.makedirs(artifact_dir, exist_ok=True)
        model_path = os.path.join(artifact_dir, "model.joblib")
        metadata_path = os.path.join(artifact_dir, "metadata.json")

        joblib.dump(self.model, model_path)

        metadata = {
            "model_version": self.config.model_version,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "hyperparameters": {
                "n_estimators": self.config.n_estimators,
                "max_samples": self.config.max_samples,
                "contamination": self.config.contamination,
                "max_features": self.config.max_features,
                "random_state": self.config.random_state,
                "decision_threshold": self.config.decision_threshold,
                "score_scaling_temperature": self.config.score_scaling_temperature,
            },
            "feature_schema_version": self.config.feature_schema_version,
            "feature_names": self.config.feature_names,
            "feature_count": len(self.config.feature_names),
            "temperature_unit": self.config.temperature_unit,
            "training_feature_medians": self.training_feature_medians,
            "training_feature_mads": self.training_feature_mads,
            "training_details": self.training_metadata,
            "dependencies": {
                "joblib": joblib.__version__,
                "scikit-learn": "1.9.1",
                "numpy": np.__version__,
            },
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, artifact_dir: str) -> "IsolationForestDetector":
        """
        Load persisted model and validate feature schema compatibility.
        """
        model_path = os.path.join(artifact_dir, "model.joblib")
        metadata_path = os.path.join(artifact_dir, "metadata.json")

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model artifacts missing in {artifact_dir}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Validate schema version
        saved_schema_ver = metadata.get("feature_schema_version")
        if saved_schema_ver != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: saved={saved_schema_ver}, current={FEATURE_SCHEMA_VERSION}"
            )

        hp = metadata["hyperparameters"]
        config = ModelConfig(
            model_version=metadata["model_version"],
            n_estimators=hp["n_estimators"],
            max_samples=hp["max_samples"],
            contamination=hp["contamination"],
            max_features=hp["max_features"],
            random_state=hp["random_state"],
            decision_threshold=hp["decision_threshold"],
            score_scaling_temperature=hp.get("score_scaling_temperature", 0.15),
            feature_schema_version=metadata["feature_schema_version"],
            temperature_unit=metadata.get("temperature_unit", "F"),
            feature_names=metadata["feature_names"],
        )

        detector = cls(config=config)
        detector.model = joblib.load(model_path)
        detector.is_fitted = True
        detector.training_feature_medians = metadata.get("training_feature_medians", {})
        detector.training_feature_mads = metadata.get("training_feature_mads", {})
        detector.training_metadata = metadata.get("training_details", {})

        return detector

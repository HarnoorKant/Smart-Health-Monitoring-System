"""
Deterministic clinical safety rule engine.
Member 3: ML Analysis, Integration, QA and Deployment.

Independent of machine learning; guarantees patient safety monitoring
even if ML models are unavailable or return insufficient_data.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RuleFinding:
    rule_code: str
    severity: str  # "CRITICAL" | "WARNING" | "INFO"
    message: str
    metric_code: str
    observed_value: float
    threshold_value: float


def evaluate_safety_rules(
    heart_rate: Optional[float],
    spo2: Optional[float],
    temperature_f: Optional[float],
) -> List[RuleFinding]:
    """
    Evaluate deterministic clinical safety bounds on instantaneous readings.
    Executes independently of any machine learning model or database.
    """
    findings: List[RuleFinding] = []

    # Rule 1: Critical Hypoxemia
    if spo2 is not None and spo2 < 90.0:
        findings.append(RuleFinding(
            rule_code="CRITICAL_HYPOXEMIA",
            severity="CRITICAL",
            message=f"Oxygen saturation severely depressed: {spo2:.1f}% (< 90.0%)",
            metric_code="spo2",
            observed_value=float(spo2),
            threshold_value=90.0,
        ))
    elif spo2 is not None and spo2 < 94.0:
        findings.append(RuleFinding(
            rule_code="MILD_HYPOXEMIA",
            severity="WARNING",
            message=f"Oxygen saturation below normal baseline: {spo2:.1f}% (< 94.0%)",
            metric_code="spo2",
            observed_value=float(spo2),
            threshold_value=94.0,
        ))

    # Rule 2: Critical Tachycardia
    if heart_rate is not None and heart_rate > 130.0:
        findings.append(RuleFinding(
            rule_code="CRITICAL_TACHYCARDIA",
            severity="CRITICAL",
            message=f"Severe tachycardia detected: {heart_rate:.1f} bpm (> 130.0 bpm)",
            metric_code="heart_rate",
            observed_value=float(heart_rate),
            threshold_value=130.0,
        ))
    elif heart_rate is not None and heart_rate > 100.0:
        findings.append(RuleFinding(
            rule_code="MILD_TACHYCARDIA",
            severity="WARNING",
            message=f"Elevated resting heart rate: {heart_rate:.1f} bpm (> 100.0 bpm)",
            metric_code="heart_rate",
            observed_value=float(heart_rate),
            threshold_value=100.0,
        ))

    # Rule 3: High Fever
    if temperature_f is not None and temperature_f >= 102.0:
        findings.append(RuleFinding(
            rule_code="HIGH_FEVER",
            severity="CRITICAL",
            message=f"High fever detected: {temperature_f:.1f} F (>= 102.0 F)",
            metric_code="temperature_f",
            observed_value=float(temperature_f),
            threshold_value=102.0,
        ))
    elif temperature_f is not None and temperature_f >= 100.4:
        findings.append(RuleFinding(
            rule_code="MILD_FEVER",
            severity="WARNING",
            message=f"Pyrexia/mild fever detected: {temperature_f:.1f} F (>= 100.4 F)",
            metric_code="temperature_f",
            observed_value=float(temperature_f),
            threshold_value=100.4,
        ))

    return findings

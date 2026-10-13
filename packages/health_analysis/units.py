"""
Unit normalization and conversion utilities for health telemetry.
Member 3: ML Analysis, Integration, QA and Deployment.
"""

from typing import Union
import numpy as np


def fahrenheit_to_celsius(temp_f: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert Fahrenheit to Celsius using exact linear formula:
    T_c = (T_f - 32) * 5 / 9

    Preserves NaN and non-finite values without raising errors.
    """
    if isinstance(temp_f, (float, int)):
        if np.isnan(temp_f) or np.isinf(temp_f):
            return float(temp_f)
        return (float(temp_f) - 32.0) * 5.0 / 9.0
    arr = np.asarray(temp_f, dtype=float)
    return (arr - 32.0) * 5.0 / 9.0


def celsius_to_fahrenheit(temp_c: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert Celsius to Fahrenheit using exact linear formula:
    T_f = (T_c * 9 / 5) + 32

    Preserves NaN and non-finite values without raising errors.
    """
    if isinstance(temp_c, (float, int)):
        if np.isnan(temp_c) or np.isinf(temp_c):
            return float(temp_c)
        return (float(temp_c) * 9.0 / 5.0) + 32.0
    arr = np.asarray(temp_c, dtype=float)
    return (arr * 9.0 / 5.0) + 32.0


def normalize_temperature(
    val: float,
    source_unit: str = "F",
    target_unit: str = "F"
) -> float:
    """
    Normalize temperature value between 'F' and 'C'.
    """
    if np.isnan(val) or np.isinf(val):
        return float(val)

    src = source_unit.upper()
    tgt = target_unit.upper()

    if src == tgt:
        return float(val)
    if src == "F" and tgt == "C":
        return fahrenheit_to_celsius(val)
    if src == "C" and tgt == "F":
        return celsius_to_fahrenheit(val)
    raise ValueError(f"Unsupported temperature unit conversion: {src} -> {tgt}")

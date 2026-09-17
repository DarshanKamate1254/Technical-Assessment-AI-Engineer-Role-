"""Anomaly detection package for operational business rules and statistical IQR analysis."""

from app.anomalies.detector import detect_anomalies
from app.anomalies.rules import detect_business_rule_anomalies
from app.anomalies.statistical import (
    calculate_iqr_thresholds,
    detect_statistical_anomalies,
)

__all__ = [
    "detect_anomalies",
    "detect_business_rule_anomalies",
    "detect_statistical_anomalies",
    "calculate_iqr_thresholds",
]

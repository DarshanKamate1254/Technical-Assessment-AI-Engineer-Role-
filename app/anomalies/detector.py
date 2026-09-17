"""Anomaly Detector Coordinator Module for Customer Support Tickets.

Orchestrates both business-rule and statistical anomaly detectors, computes
summary distributions, and aggregates structured evidence for LangChain/LLM consumption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.anomalies.rules import detect_business_rule_anomalies
from app.anomalies.statistical import detect_statistical_anomalies


def detect_anomalies(db_path: str | Path | None = None) -> dict[str, Any]:
    """Execute complete anomaly detection suite (business rules + statistical IQR).

    Preserves multiple anomaly records when a ticket triggers multiple rules.

    Args:
        db_path: Optional path to SQLite database.

    Returns:
        Structured dictionary containing:
            - summary: Metric breakdown (total_anomalies, critical, high, medium, by_type, unique_tickets_flagged)
            - statistical_thresholds: IQR metrics (Q1, Q3, IQR, upper_bound, sample_size)
            - anomalies: Comprehensive list of structured anomaly records with evidence
    """
    business_anomalies = detect_business_rule_anomalies(db_path)
    thresholds, stat_anomalies = detect_statistical_anomalies(db_path)

    all_anomalies = business_anomalies + stat_anomalies

    # Compute summary counts
    critical_count = sum(1 for a in all_anomalies if a.get("severity") == "critical")
    high_count = sum(1 for a in all_anomalies if a.get("severity") == "high")
    medium_count = sum(1 for a in all_anomalies if a.get("severity") == "medium")

    by_type: dict[str, int] = {}
    unique_ticket_ids: set[str] = set()

    for a in all_anomalies:
        atype = a.get("anomaly_type", "UNKNOWN")
        by_type[atype] = by_type.get(atype, 0) + 1
        tid = a.get("ticket_id")
        if tid:
            unique_ticket_ids.add(tid)

    summary = {
        "total_anomalies": len(all_anomalies),
        "critical": critical_count,
        "high": high_count,
        "medium": medium_count,
        "by_type": by_type,
        "unique_tickets_flagged": len(unique_ticket_ids),
    }

    return {
        "summary": summary,
        "statistical_thresholds": thresholds,
        "anomalies": all_anomalies,
    }

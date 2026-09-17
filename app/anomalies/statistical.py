"""Statistical Anomaly Detection Module for Customer Support Tickets.

Implements deterministic statistical outlier detection using the Interquartile Range
(IQR) method on resolution times, calculating Q1, Q3, IQR, and the upper bound threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import numpy as np

from app.database.connection import DEFAULT_DB_PATH, get_connection
from app.database.schema import TABLE_NAME


def calculate_iqr_thresholds(values: Sequence[float]) -> dict[str, Any]:
    """Calculate deterministic Interquartile Range (IQR) outlier thresholds.

    Formula:
        - Q1 = 25th percentile
        - Q3 = 75th percentile
        - IQR = Q3 - Q1
        - upper_bound = Q3 + 1.5 * IQR
        - lower_bound = max(0.0, Q1 - 1.5 * IQR)

    Args:
        values: Sequence of numeric values (non-null).

    Returns:
        Dictionary containing q1, q3, iqr, upper_bound, lower_bound, and sample_size.
    """
    clean_values = [float(v) for v in values if v is not None and not np.isnan(v)]
    sample_size = len(clean_values)

    if sample_size < 4:
        return {
            "q1": None,
            "q3": None,
            "iqr": None,
            "upper_bound": None,
            "lower_bound": None,
            "sample_size": sample_size,
            "has_sufficient_data": False,
        }

    q1 = float(np.percentile(clean_values, 25))
    q3 = float(np.percentile(clean_values, 75))
    iqr = float(q3 - q1)
    upper_bound = float(q3 + 1.5 * iqr)
    lower_bound = float(max(0.0, q1 - 1.5 * iqr))

    return {
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "upper_bound": round(upper_bound, 4),
        "lower_bound": round(lower_bound, 4),
        "sample_size": sample_size,
        "has_sufficient_data": True,
    }


def detect_statistical_anomalies(
    db_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Detect statistical resolution-time outliers using the IQR method.

    Returns:
        Tuple of (threshold_dictionary, list_of_statistical_anomalies).
    """
    anomalies: list[dict[str, Any]] = []
    db_target = db_path if db_path is not None else DEFAULT_DB_PATH

    with get_connection(db_target) as conn:
        # Fetch all non-null resolution times
        sql_fetch = (
            f"SELECT resolution_time_hrs FROM {TABLE_NAME} "
            "WHERE resolution_time_hrs IS NOT NULL;"
        )
        rows = conn.execute(sql_fetch).fetchall()
        res_values = [float(r["resolution_time_hrs"]) for r in rows]

        thresholds = calculate_iqr_thresholds(res_values)
        if not thresholds.get("has_sufficient_data") or thresholds.get("upper_bound") is None:
            return thresholds, []

        upper_bound = thresholds["upper_bound"]

        # Find tickets exceeding upper bound
        sql_outliers = f"""
        SELECT
            ticket_id, priority, status, resolution_time_hrs, agent_id, issue_summary
        FROM {TABLE_NAME}
        WHERE resolution_time_hrs > ?
        ORDER BY resolution_time_hrs DESC;
        """
        outlier_rows = conn.execute(sql_outliers, (upper_bound,)).fetchall()

        for row in outlier_rows:
            actual_res = float(row["resolution_time_hrs"])
            anomalies.append(
                {
                    "ticket_id": str(row["ticket_id"]),
                    "anomaly_type": "STATISTICAL_LONG_RESOLUTION",
                    "severity": "medium",
                    "reason": (
                        f"Resolution time ({actual_res:.1f}h) exceeds the statistical "
                        f"IQR upper bound threshold ({upper_bound:.2f}h)."
                    ),
                    "evidence": {
                        "actual_resolution_time_hrs": actual_res,
                        "q1": thresholds["q1"],
                        "q3": thresholds["q3"],
                        "iqr": thresholds["iqr"],
                        "upper_bound": upper_bound,
                        "priority": str(row["priority"]),
                        "status": str(row["status"]),
                        "agent_id": str(row["agent_id"]),
                    },
                }
            )

    return thresholds, anomalies

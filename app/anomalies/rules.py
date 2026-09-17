"""Business-Rule Anomaly Detection Module for Customer Support Tickets.

Implements deterministic rule-based operational anomaly detectors:
- Rule 1: HIGH_PRIORITY_UNRESOLVED_OVER_24H (Severity: High)
- Rule 2: CRITICAL_UNRESOLVED (Severity: Critical)
- Rule 3: LONG_RESOLUTION_TIME (Severity: Medium, threshold > 48h)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.database.connection import DEFAULT_DB_PATH, get_connection
from app.database.schema import TABLE_NAME


def detect_business_rule_anomalies(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Execute all deterministic business rule anomaly detection checks.

    Returns:
        List of structured anomaly dictionaries containing:
            - ticket_id
            - anomaly_type
            - severity
            - reason
            - evidence
    """
    anomalies: list[dict[str, Any]] = []
    db_target = db_path if db_path is not None else DEFAULT_DB_PATH

    with get_connection(db_target) as conn:
        # ---------------------------------------------------------------------
        # Rule 1: HIGH_PRIORITY_UNRESOLVED_OVER_24H
        # ---------------------------------------------------------------------
        sql_rule1 = f"""
        SELECT
            ticket_id, priority, status, created_at, ticket_age_hours,
            unresolved_age_hours, agent_id, issue_summary
        FROM {TABLE_NAME}
        WHERE priority IN ('High', 'Critical')
          AND is_unresolved = 1
          AND unresolved_age_hours > 24.0
        ORDER BY unresolved_age_hours DESC;
        """
        for row in conn.execute(sql_rule1).fetchall():
            anomalies.append(
                {
                    "ticket_id": str(row["ticket_id"]),
                    "anomaly_type": "HIGH_PRIORITY_UNRESOLVED_OVER_24H",
                    "severity": "high",
                    "reason": "High-priority ticket remains unresolved for more than 24 hours.",
                    "evidence": {
                        "priority": str(row["priority"]),
                        "status": str(row["status"]),
                        "unresolved_age_hours": float(row["unresolved_age_hours"]),
                        "created_at": str(row["created_at"]),
                        "agent_id": str(row["agent_id"]),
                        "issue_summary": str(row["issue_summary"]),
                    },
                }
            )

        # ---------------------------------------------------------------------
        # Rule 2: CRITICAL_UNRESOLVED
        # ---------------------------------------------------------------------
        sql_rule2 = f"""
        SELECT
            ticket_id, priority, status, created_at, ticket_age_hours,
            unresolved_age_hours, agent_id, issue_summary
        FROM {TABLE_NAME}
        WHERE priority = 'Critical'
          AND is_unresolved = 1
        ORDER BY ticket_age_hours DESC;
        """
        for row in conn.execute(sql_rule2).fetchall():
            anomalies.append(
                {
                    "ticket_id": str(row["ticket_id"]),
                    "anomaly_type": "CRITICAL_UNRESOLVED",
                    "severity": "critical",
                    "reason": "Critical priority ticket remains in an unresolved state.",
                    "evidence": {
                        "priority": str(row["priority"]),
                        "status": str(row["status"]),
                        "unresolved_age_hours": float(row["unresolved_age_hours"])
                        if row["unresolved_age_hours"] is not None
                        else None,
                        "created_at": str(row["created_at"]),
                        "agent_id": str(row["agent_id"]),
                        "issue_summary": str(row["issue_summary"]),
                    },
                }
            )

        # ---------------------------------------------------------------------
        # Rule 3: LONG_RESOLUTION_TIME (> 48 hours)
        # ---------------------------------------------------------------------
        sql_rule3 = f"""
        SELECT
            ticket_id, priority, status, resolution_time_hrs, agent_id, issue_summary
        FROM {TABLE_NAME}
        WHERE resolution_time_hrs > 48.0
        ORDER BY resolution_time_hrs DESC;
        """
        for row in conn.execute(sql_rule3).fetchall():
            res_time = float(row["resolution_time_hrs"])
            anomalies.append(
                {
                    "ticket_id": str(row["ticket_id"]),
                    "anomaly_type": "LONG_RESOLUTION_TIME",
                    "severity": "medium",
                    "reason": f"Resolution time ({res_time:.1f}h) exceeds 48-hour operational threshold.",
                    "evidence": {
                        "resolution_time_hrs": res_time,
                        "priority": str(row["priority"]),
                        "status": str(row["status"]),
                        "agent_id": str(row["agent_id"]),
                        "threshold_hrs": 48.0,
                    },
                }
            )

    return anomalies

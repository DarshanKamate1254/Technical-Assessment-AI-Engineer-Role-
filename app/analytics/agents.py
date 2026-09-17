"""Agent Performance Analytics Module for Customer Support Tickets.

Calculates structured, objective operational metrics per support agent:
ticket volume, resolution status, response/resolution times, and ratings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.database.connection import DEFAULT_DB_PATH, get_connection
from app.database.schema import TABLE_NAME


def get_agent_performance(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Calculate structured operational performance statistics for each agent.

    Returns:
        List of dictionaries containing:
            - agent_id (str)
            - total_tickets (int)
            - resolved_tickets (int)
            - unresolved_tickets (int)
            - average_response_time (float | None)
            - average_resolution_time (float | None)
            - average_customer_rating (float | None)
            - rated_ticket_count (int)
    """
    sql = f"""
    SELECT
        agent_id,
        COUNT(*) AS total_tickets,
        SUM(CASE WHEN is_resolved = 1 THEN 1 ELSE 0 END) AS resolved_tickets,
        SUM(CASE WHEN is_unresolved = 1 THEN 1 ELSE 0 END) AS unresolved_tickets,
        ROUND(AVG(response_time_hrs), 2) AS average_response_time,
        ROUND(AVG(resolution_time_hrs), 2) AS average_resolution_time,
        ROUND(AVG(customer_rating), 4) AS average_customer_rating,
        COUNT(customer_rating) AS rated_ticket_count
    FROM {TABLE_NAME}
    GROUP BY agent_id
    ORDER BY total_tickets DESC, agent_id ASC;
    """
    with get_connection(db_path if db_path is not None else DEFAULT_DB_PATH) as conn:
        rows = conn.execute(sql).fetchall()
        return [
            {
                "agent_id": str(r["agent_id"]),
                "total_tickets": int(r["total_tickets"]),
                "resolved_tickets": int(r["resolved_tickets"]),
                "unresolved_tickets": int(r["unresolved_tickets"]),
                "average_response_time": float(r["average_response_time"])
                if r["average_response_time"] is not None
                else None,
                "average_resolution_time": float(r["average_resolution_time"])
                if r["average_resolution_time"] is not None
                else None,
                "average_customer_rating": float(r["average_customer_rating"])
                if r["average_customer_rating"] is not None
                else None,
                "rated_ticket_count": int(r["rated_ticket_count"]),
            }
            for r in rows
        ]

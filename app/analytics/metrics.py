"""Analytics Metrics Module for Customer Support Tickets.

Provides deterministic functions for ticket volume, unresolved tickets,
response-time statistics, resolution-time statistics, customer ratings,
and SLA/temporal age metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from app.database.connection import DEFAULT_DB_PATH, get_connection
from app.database.schema import TABLE_NAME


def _get_db_conn(db_path: str | Path | None = None):
    return get_connection(db_path if db_path is not None else DEFAULT_DB_PATH)


# =============================================================================
# 1. Ticket Volume Metrics
# =============================================================================


def get_total_tickets(db_path: str | Path | None = None) -> int:
    """Return the total number of support tickets in the database."""
    sql = f"SELECT COUNT(*) AS count FROM {TABLE_NAME};"
    with _get_db_conn(db_path) as conn:
        row = conn.execute(sql).fetchone()
        return int(row["count"]) if row else 0


def get_ticket_count_by_status(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return ticket volume breakdown grouped by status."""
    sql = f"""
    SELECT status, COUNT(*) AS count
    FROM {TABLE_NAME}
    GROUP BY status
    ORDER BY count DESC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [{"status": str(r["status"]), "count": int(r["count"])} for r in rows]


def get_ticket_count_by_priority(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return ticket volume breakdown grouped by priority."""
    sql = f"""
    SELECT priority, COUNT(*) AS count
    FROM {TABLE_NAME}
    GROUP BY priority
    ORDER BY count DESC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [{"priority": str(r["priority"]), "count": int(r["count"])} for r in rows]


def get_ticket_count_by_category(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return ticket volume breakdown grouped by category."""
    sql = f"""
    SELECT category, COUNT(*) AS count
    FROM {TABLE_NAME}
    GROUP BY category
    ORDER BY count DESC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [{"category": str(r["category"]), "count": int(r["count"])} for r in rows]


def get_ticket_count_by_agent(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return ticket volume breakdown grouped by agent."""
    sql = f"""
    SELECT agent_id, COUNT(*) AS count
    FROM {TABLE_NAME}
    GROUP BY agent_id
    ORDER BY count DESC, agent_id ASC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [{"agent_id": str(r["agent_id"]), "count": int(r["count"])} for r in rows]


# =============================================================================
# 2. Unresolved Ticket Analytics
# =============================================================================


def get_unresolved_ticket_count(db_path: str | Path | None = None) -> int:
    """Return the total count of unresolved (Open or Escalated) tickets."""
    sql = f"SELECT COUNT(*) AS count FROM {TABLE_NAME} WHERE is_unresolved = 1;"
    with _get_db_conn(db_path) as conn:
        row = conn.execute(sql).fetchone()
        return int(row["count"]) if row else 0


def get_unresolved_by_priority(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return unresolved ticket breakdown grouped by priority."""
    sql = f"""
    SELECT priority, COUNT(*) AS count
    FROM {TABLE_NAME}
    WHERE is_unresolved = 1
    GROUP BY priority
    ORDER BY count DESC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [{"priority": str(r["priority"]), "count": int(r["count"])} for r in rows]


def get_unresolved_by_category(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return unresolved ticket breakdown grouped by category."""
    sql = f"""
    SELECT category, COUNT(*) AS count
    FROM {TABLE_NAME}
    WHERE is_unresolved = 1
    GROUP BY category
    ORDER BY count DESC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [{"category": str(r["category"]), "count": int(r["count"])} for r in rows]


def get_unresolved_by_agent(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return unresolved ticket breakdown grouped by agent."""
    sql = f"""
    SELECT agent_id, COUNT(*) AS count
    FROM {TABLE_NAME}
    WHERE is_unresolved = 1
    GROUP BY agent_id
    ORDER BY count DESC, agent_id ASC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [{"agent_id": str(r["agent_id"]), "count": int(r["count"])} for r in rows]


def get_critical_unresolved_count(db_path: str | Path | None = None) -> int:
    """Return the count of Critical priority tickets that are currently unresolved."""
    sql = f"SELECT COUNT(*) AS count FROM {TABLE_NAME} WHERE is_critical_unresolved = 1;"
    with _get_db_conn(db_path) as conn:
        row = conn.execute(sql).fetchone()
        return int(row["count"]) if row else 0


def get_high_priority_unresolved_count(db_path: str | Path | None = None) -> int:
    """Return the count of High or Critical priority tickets that are unresolved."""
    sql = (
        f"SELECT COUNT(*) AS count FROM {TABLE_NAME} "
        "WHERE priority IN ('High', 'Critical') AND is_unresolved = 1;"
    )
    with _get_db_conn(db_path) as conn:
        row = conn.execute(sql).fetchone()
        return int(row["count"]) if row else 0


# =============================================================================
# 3. Response-Time Analytics
# =============================================================================


def get_average_response_time(
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate overall average response time in hours, excluding missing values."""
    sql = f"""
    SELECT
        ROUND(AVG(response_time_hrs), 2) AS average_response_time_hrs,
        COUNT(response_time_hrs) AS observed_count
    FROM {TABLE_NAME}
    WHERE response_time_hrs IS NOT NULL;
    """
    with _get_db_conn(db_path) as conn:
        row = conn.execute(sql).fetchone()
        if row and row["observed_count"] > 0:
            return {
                "average_response_time_hrs": float(row["average_response_time_hrs"]),
                "observed_count": int(row["observed_count"]),
            }
        return {"average_response_time_hrs": None, "observed_count": 0}


def get_median_response_time(
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate overall median response time in hours from observed response times."""
    sql = f"SELECT response_time_hrs FROM {TABLE_NAME} WHERE response_time_hrs IS NOT NULL;"
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        values = [float(r["response_time_hrs"]) for r in rows]
        if values:
            median_val = round(float(np.median(values)), 2)
            return {"median_response_time_hrs": median_val, "observed_count": len(values)}
        return {"median_response_time_hrs": None, "observed_count": 0}


def get_response_time_by_priority(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return average response time grouped by priority."""
    sql = f"""
    SELECT
        priority,
        ROUND(AVG(response_time_hrs), 2) AS average_response_time_hrs,
        COUNT(response_time_hrs) AS observed_count
    FROM {TABLE_NAME}
    WHERE response_time_hrs IS NOT NULL
    GROUP BY priority
    ORDER BY average_response_time_hrs ASC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [
            {
                "priority": str(r["priority"]),
                "average_response_time_hrs": float(r["average_response_time_hrs"]),
                "observed_count": int(r["observed_count"]),
            }
            for r in rows
        ]


def get_response_time_by_category(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return average response time grouped by category."""
    sql = f"""
    SELECT
        category,
        ROUND(AVG(response_time_hrs), 2) AS average_response_time_hrs,
        COUNT(response_time_hrs) AS observed_count
    FROM {TABLE_NAME}
    WHERE response_time_hrs IS NOT NULL
    GROUP BY category
    ORDER BY average_response_time_hrs ASC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [
            {
                "category": str(r["category"]),
                "average_response_time_hrs": float(r["average_response_time_hrs"]),
                "observed_count": int(r["observed_count"]),
            }
            for r in rows
        ]


def get_response_time_by_agent(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return average response time grouped by agent."""
    sql = f"""
    SELECT
        agent_id,
        ROUND(AVG(response_time_hrs), 2) AS average_response_time_hrs,
        COUNT(response_time_hrs) AS observed_count
    FROM {TABLE_NAME}
    WHERE response_time_hrs IS NOT NULL
    GROUP BY agent_id
    ORDER BY average_response_time_hrs ASC, agent_id ASC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [
            {
                "agent_id": str(r["agent_id"]),
                "average_response_time_hrs": float(r["average_response_time_hrs"]),
                "observed_count": int(r["observed_count"]),
            }
            for r in rows
        ]


# =============================================================================
# 4. Resolution-Time Analytics
# =============================================================================


def get_average_resolution_time(
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate overall average resolution time in hours, excluding NULLs."""
    sql = f"""
    SELECT
        ROUND(AVG(resolution_time_hrs), 2) AS average_resolution_time_hrs,
        COUNT(resolution_time_hrs) AS resolved_ticket_count
    FROM {TABLE_NAME}
    WHERE resolution_time_hrs IS NOT NULL;
    """
    with _get_db_conn(db_path) as conn:
        row = conn.execute(sql).fetchone()
        if row and row["resolved_ticket_count"] > 0:
            return {
                "average_resolution_time_hrs": float(row["average_resolution_time_hrs"]),
                "resolved_ticket_count": int(row["resolved_ticket_count"]),
            }
        return {"average_resolution_time_hrs": None, "resolved_ticket_count": 0}


def get_median_resolution_time(
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate overall median resolution time in hours for resolved tickets."""
    sql = f"SELECT resolution_time_hrs FROM {TABLE_NAME} WHERE resolution_time_hrs IS NOT NULL;"
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        values = [float(r["resolution_time_hrs"]) for r in rows]
        if values:
            median_val = round(float(np.median(values)), 2)
            return {"median_resolution_time_hrs": median_val, "resolved_ticket_count": len(values)}
        return {"median_resolution_time_hrs": None, "resolved_ticket_count": 0}


def get_resolution_time_by_priority(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Calculate average resolution time grouped by priority."""
    sql = f"""
    SELECT
        priority,
        ROUND(AVG(resolution_time_hrs), 2) AS average_resolution_time_hrs,
        COUNT(resolution_time_hrs) AS resolved_ticket_count
    FROM {TABLE_NAME}
    WHERE resolution_time_hrs IS NOT NULL
    GROUP BY priority
    ORDER BY average_resolution_time_hrs DESC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [
            {
                "priority": str(r["priority"]),
                "average_resolution_time_hrs": float(r["average_resolution_time_hrs"]),
                "resolved_ticket_count": int(r["resolved_ticket_count"]),
            }
            for r in rows
        ]


def get_resolution_time_by_category(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Calculate average resolution time grouped by category."""
    sql = f"""
    SELECT
        category,
        ROUND(AVG(resolution_time_hrs), 2) AS average_resolution_time_hrs,
        COUNT(resolution_time_hrs) AS resolved_ticket_count
    FROM {TABLE_NAME}
    WHERE resolution_time_hrs IS NOT NULL
    GROUP BY category
    ORDER BY average_resolution_time_hrs DESC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [
            {
                "category": str(r["category"]),
                "average_resolution_time_hrs": float(r["average_resolution_time_hrs"]),
                "resolved_ticket_count": int(r["resolved_ticket_count"]),
            }
            for r in rows
        ]


def get_resolution_time_by_agent(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Calculate average resolution time grouped by agent."""
    sql = f"""
    SELECT
        agent_id,
        ROUND(AVG(resolution_time_hrs), 2) AS average_resolution_time_hrs,
        COUNT(resolution_time_hrs) AS resolved_ticket_count
    FROM {TABLE_NAME}
    WHERE resolution_time_hrs IS NOT NULL
    GROUP BY agent_id
    ORDER BY average_resolution_time_hrs ASC, agent_id ASC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [
            {
                "agent_id": str(r["agent_id"]),
                "average_resolution_time_hrs": float(r["average_resolution_time_hrs"]),
                "resolved_ticket_count": int(r["resolved_ticket_count"]),
            }
            for r in rows
        ]


# =============================================================================
# 5. Customer-Rating Analytics
# =============================================================================


def get_average_customer_rating(
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate overall average customer rating, excluding missing ratings."""
    sql = f"""
    SELECT
        ROUND(AVG(customer_rating), 4) AS average_customer_rating,
        COUNT(customer_rating) AS rated_ticket_count
    FROM {TABLE_NAME}
    WHERE customer_rating IS NOT NULL;
    """
    with _get_db_conn(db_path) as conn:
        row = conn.execute(sql).fetchone()
        if row and row["rated_ticket_count"] > 0:
            return {
                "average_customer_rating": float(row["average_customer_rating"]),
                "rated_ticket_count": int(row["rated_ticket_count"]),
            }
        return {"average_customer_rating": None, "rated_ticket_count": 0}


def get_rating_distribution(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return distribution of integer star customer ratings (1 to 5)."""
    sql = f"""
    SELECT
        CAST(ROUND(customer_rating) AS INTEGER) AS rating,
        COUNT(*) AS count
    FROM {TABLE_NAME}
    WHERE customer_rating IS NOT NULL
    GROUP BY rating
    ORDER BY rating ASC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [{"rating": int(r["rating"]), "count": int(r["count"])} for r in rows]


def get_average_rating_by_agent(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Calculate average customer rating grouped by agent."""
    sql = f"""
    SELECT
        agent_id,
        ROUND(AVG(customer_rating), 4) AS average_rating,
        COUNT(customer_rating) AS rated_ticket_count
    FROM {TABLE_NAME}
    WHERE customer_rating IS NOT NULL
    GROUP BY agent_id
    ORDER BY average_rating ASC, agent_id ASC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [
            {
                "agent_id": str(r["agent_id"]),
                "average_rating": float(r["average_rating"]),
                "rated_ticket_count": int(r["rated_ticket_count"]),
            }
            for r in rows
        ]


def get_average_rating_by_priority(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Calculate average customer rating grouped by priority."""
    sql = f"""
    SELECT
        priority,
        ROUND(AVG(customer_rating), 4) AS average_rating,
        COUNT(customer_rating) AS rated_ticket_count
    FROM {TABLE_NAME}
    WHERE customer_rating IS NOT NULL
    GROUP BY priority
    ORDER BY average_rating DESC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [
            {
                "priority": str(r["priority"]),
                "average_rating": float(r["average_rating"]),
                "rated_ticket_count": int(r["rated_ticket_count"]),
            }
            for r in rows
        ]


def get_average_rating_by_category(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Calculate average customer rating grouped by category."""
    sql = f"""
    SELECT
        category,
        ROUND(AVG(customer_rating), 4) AS average_rating,
        COUNT(customer_rating) AS rated_ticket_count
    FROM {TABLE_NAME}
    WHERE customer_rating IS NOT NULL
    GROUP BY category
    ORDER BY average_rating DESC;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [
            {
                "category": str(r["category"]),
                "average_rating": float(r["average_rating"]),
                "rated_ticket_count": int(r["rated_ticket_count"]),
            }
            for r in rows
        ]


# =============================================================================
# 6. SLA & Age Analytics
# =============================================================================


def get_unresolved_over_24h(
    limit: int = 100, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Retrieve all unresolved tickets older than 24 hours."""
    sql = f"""
    SELECT
        ticket_id,
        priority,
        status,
        created_at,
        ticket_age_hours,
        unresolved_age_hours,
        agent_id,
        issue_summary
    FROM {TABLE_NAME}
    WHERE is_unresolved = 1 AND unresolved_age_hours > 24.0
    ORDER BY unresolved_age_hours DESC
    LIMIT ?;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_high_priority_unresolved_over_24h(
    limit: int = 100, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Retrieve High and Critical priority unresolved tickets older than 24 hours."""
    sql = f"""
    SELECT
        ticket_id,
        priority,
        status,
        created_at,
        ticket_age_hours,
        unresolved_age_hours,
        agent_id,
        issue_summary
    FROM {TABLE_NAME}
    WHERE priority IN ('High', 'Critical')
      AND is_unresolved = 1
      AND unresolved_age_hours > 24.0
    ORDER BY unresolved_age_hours DESC
    LIMIT ?;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_critical_unresolved_over_24h(
    limit: int = 100, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Retrieve Critical priority unresolved tickets older than 24 hours."""
    sql = f"""
    SELECT
        ticket_id,
        priority,
        status,
        created_at,
        ticket_age_hours,
        unresolved_age_hours,
        agent_id,
        issue_summary
    FROM {TABLE_NAME}
    WHERE priority = 'Critical'
      AND is_unresolved = 1
      AND unresolved_age_hours > 24.0
    ORDER BY unresolved_age_hours DESC
    LIMIT ?;
    """
    with _get_db_conn(db_path) as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in rows]

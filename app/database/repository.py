"""Query Repository Module for Customer Support Tickets.

Provides deterministic, typed, SQL-based repository functions for business queries,
aggregations, agent performance metrics, SLA monitoring, and controlled generic
filtering with strict whitelisting to eliminate SQL injection risks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.database.connection import DEFAULT_DB_PATH, get_connection
from app.database.schema import ALL_COLUMNS, TABLE_NAME

# Whitelists for SQL security
ALLOWED_COLUMNS: frozenset[str] = frozenset(ALL_COLUMNS)
ALLOWED_SORT_COLUMNS: frozenset[str] = ALLOWED_COLUMNS
ALLOWED_OPERATORS: frozenset[str] = frozenset(
    {"eq", "ne", "gt", "gte", "lt", "lte", "like", "in", "isnull"}
)

OPERATOR_SQL_MAP: dict[str, str] = {
    "eq": "=",
    "ne": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "like": "LIKE",
}


class TicketRepository:
    """Encapsulates SQL query operations against the support_tickets table."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = db_path if db_path is not None else DEFAULT_DB_PATH

    def _get_conn(self):
        return get_connection(self.db_path)

    # =========================================================================
    # 1. Required Core Business Queries
    # =========================================================================

    def get_critical_unresolved_count(self) -> int:
        """Query 1: Return the count of critical tickets that are unresolved.

        Logic: priority = 'Critical' AND status != 'Resolved' (is_critical_unresolved = 1).
        """
        sql = (
            f"SELECT COUNT(*) AS count FROM {TABLE_NAME} "
            "WHERE priority = 'Critical' AND status != 'Resolved';"
        )
        with self._get_conn() as conn:
            row = conn.execute(sql).fetchone()
            return int(row["count"]) if row else 0

    def get_lowest_average_rating_agent(self) -> dict[str, Any] | None:
        """Query 2: Find the agent with the lowest average customer rating.

        Excludes NULL ratings, groups by agent_id, and returns agent_id,
        average_rating, and rated_ticket_count.
        """
        sql = f"""
        SELECT
            agent_id,
            ROUND(AVG(customer_rating), 4) AS average_rating,
            COUNT(customer_rating) AS rated_ticket_count
        FROM {TABLE_NAME}
        WHERE customer_rating IS NOT NULL
        GROUP BY agent_id
        ORDER BY average_rating ASC, agent_id ASC
        LIMIT 1;
        """
        with self._get_conn() as conn:
            row = conn.execute(sql).fetchone()
            if row:
                return {
                    "agent_id": str(row["agent_id"]),
                    "average_rating": float(row["average_rating"]),
                    "rated_ticket_count": int(row["rated_ticket_count"]),
                }
            return None

    def get_unresolved_over_24h_count(self) -> int:
        """Query 3: Return the count of unresolved tickets older than 24 hours."""
        sql = (
            f"SELECT COUNT(*) AS count FROM {TABLE_NAME} "
            "WHERE is_unresolved = 1 AND unresolved_age_hours > 24.0;"
        )
        with self._get_conn() as conn:
            row = conn.execute(sql).fetchone()
            return int(row["count"]) if row else 0

    def get_high_priority_unresolved_over_24h(
        self, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query 4: Retrieve high-priority and critical unresolved tickets older than 24 hours.

        Returns key diagnostic evidence columns ordered by age descending.
        """
        sql = f"""
        SELECT
            ticket_id,
            priority,
            status,
            created_at,
            ticket_age_hours,
            agent_id,
            issue_summary
        FROM {TABLE_NAME}
        WHERE priority IN ('High', 'Critical')
          AND is_unresolved = 1
          AND unresolved_age_hours > 24.0
        ORDER BY ticket_age_hours DESC
        LIMIT ?;
        """
        with self._get_conn() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
            return [dict(row) for row in rows]

    def get_average_resolution_time_by_priority(self) -> list[dict[str, Any]]:
        """Query 5: Calculate average resolution time (in hours) grouped by priority.

        Excludes NULL resolution times (unresolved tickets).
        """
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
        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()
            return [
                {
                    "priority": str(row["priority"]),
                    "average_resolution_time_hrs": float(row["average_resolution_time_hrs"]),
                    "resolved_ticket_count": int(row["resolved_ticket_count"]),
                }
                for row in rows
            ]

    def get_ticket_counts_by_category(self) -> list[dict[str, Any]]:
        """Query 6: Return the distribution of tickets grouped by category."""
        sql = f"""
        SELECT
            category,
            COUNT(*) AS count
        FROM {TABLE_NAME}
        GROUP BY category
        ORDER BY count DESC;
        """
        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()
            return [
                {"category": str(row["category"]), "count": int(row["count"])}
                for row in rows
            ]

    def get_ticket_counts_by_status(self) -> list[dict[str, Any]]:
        """Query 7: Return the distribution of tickets grouped by status."""
        sql = f"""
        SELECT
            status,
            COUNT(*) AS count
        FROM {TABLE_NAME}
        GROUP BY status
        ORDER BY count DESC;
        """
        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()
            return [
                {"status": str(row["status"]), "count": int(row["count"])}
                for row in rows
            ]

    def get_average_customer_rating_by_agent(self) -> list[dict[str, Any]]:
        """Query 8: Return the average customer rating for each agent.

        Excludes tickets with NULL ratings and orders by lowest to highest average rating.
        """
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
        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()
            return [
                {
                    "agent_id": str(row["agent_id"]),
                    "average_rating": float(row["average_rating"]),
                    "rated_ticket_count": int(row["rated_ticket_count"]),
                }
                for row in rows
            ]

    # =========================================================================
    # 2. General Query & Aggregation Helpers
    # =========================================================================

    def get_ticket_count(self) -> int:
        """Return total number of tickets stored in the database."""
        sql = f"SELECT COUNT(*) AS count FROM {TABLE_NAME};"
        with self._get_conn() as conn:
            row = conn.execute(sql).fetchone()
            return int(row["count"]) if row else 0

    def get_unresolved_ticket_count(self) -> int:
        """Return total count of unresolved (Open or Escalated) tickets."""
        sql = f"SELECT COUNT(*) AS count FROM {TABLE_NAME} WHERE is_unresolved = 1;"
        with self._get_conn() as conn:
            row = conn.execute(sql).fetchone()
            return int(row["count"]) if row else 0

    def get_ticket_counts_by_priority(self) -> list[dict[str, Any]]:
        """Return ticket counts grouped by priority."""
        sql = f"""
        SELECT
            priority,
            COUNT(*) AS count
        FROM {TABLE_NAME}
        GROUP BY priority
        ORDER BY count DESC;
        """
        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()
            return [
                {"priority": str(row["priority"]), "count": int(row["count"])}
                for row in rows
            ]

    def get_average_resolution_time(self) -> float | None:
        """Calculate overall average resolution time in hours for resolved tickets."""
        sql = (
            f"SELECT ROUND(AVG(resolution_time_hrs), 2) AS avg_res "
            f"FROM {TABLE_NAME} WHERE resolution_time_hrs IS NOT NULL;"
        )
        with self._get_conn() as conn:
            row = conn.execute(sql).fetchone()
            return float(row["avg_res"]) if row and row["avg_res"] is not None else None

    def get_average_response_time(self) -> float | None:
        """Calculate overall average response time in hours."""
        sql = (
            f"SELECT ROUND(AVG(response_time_hrs), 2) AS avg_resp "
            f"FROM {TABLE_NAME} WHERE response_time_hrs IS NOT NULL;"
        )
        with self._get_conn() as conn:
            row = conn.execute(sql).fetchone()
            return float(row["avg_resp"]) if row and row["avg_resp"] is not None else None

    def get_tickets_by_agent(
        self, agent_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch all tickets assigned to a specific agent."""
        sql = f"SELECT * FROM {TABLE_NAME} WHERE agent_id = ? LIMIT ?;"
        with self._get_conn() as conn:
            rows = conn.execute(sql, (agent_id, limit)).fetchall()
            return [dict(row) for row in rows]

    def get_tickets_by_priority(
        self, priority: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch tickets matching a specific priority."""
        sql = f"SELECT * FROM {TABLE_NAME} WHERE priority = ? LIMIT ?;"
        with self._get_conn() as conn:
            rows = conn.execute(sql, (priority, limit)).fetchall()
            return [dict(row) for row in rows]

    def get_tickets_by_status(
        self, status: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch tickets matching a specific status."""
        sql = f"SELECT * FROM {TABLE_NAME} WHERE status = ? LIMIT ?;"
        with self._get_conn() as conn:
            rows = conn.execute(sql, (status, limit)).fetchall()
            return [dict(row) for row in rows]

    def get_ticket_by_id(self, ticket_id: str) -> dict[str, Any] | None:
        """Fetch a single ticket record by its primary key ticket_id."""
        sql = f"SELECT * FROM {TABLE_NAME} WHERE ticket_id = ? LIMIT 1;"
        with self._get_conn() as conn:
            row = conn.execute(sql, (ticket_id,)).fetchone()
            return dict(row) if row else None

    # =========================================================================
    # 3. Controlled Generic Filtering Interface (SQL-Injection Safe)
    # =========================================================================

    def query_tickets(
        self,
        filters: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        order_by: str | None = None,
        descending: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized query with strict column/operator whitelisting.

        Security constraints:
        - All selected columns, sort columns, and filter columns are strictly validated against ALLOWED_COLUMNS.
        - All values are passed using parameterized SQL placeholders (?).

        Supported filter formats:
        - {"status": "Open"}                  -> status = 'Open'
        - {"priority__in": ["High", "Critical"]} -> priority IN ('High', 'Critical')
        - {"ticket_age_hours__gt": 24.0}       -> ticket_age_hours > 24.0
        - {"resolution_time_hrs__isnull": True} -> resolution_time_hrs IS NULL

        Args:
            filters: Dictionary mapping column names (or column__operator) to values.
            columns: Optional list of columns to retrieve. Defaults to ALL_COLUMNS.
            order_by: Optional column name to sort by. Must be in ALLOWED_SORT_COLUMNS.
            descending: If True, sort DESC; otherwise ASC.
            limit: Maximum rows to return (default: 100).
            offset: Rows offset for pagination (default: 0).

        Returns:
            List of matching records as dictionaries.

        Raises:
            ValueError: If an unwhitelisted column, sort key, or operator is requested.
        """
        # Validate columns
        if columns is not None:
            for col in columns:
                if col not in ALLOWED_COLUMNS:
                    raise ValueError(
                        f"Disallowed column '{col}'. Allowed: {sorted(ALLOWED_COLUMNS)}"
                    )
            select_clause = ", ".join(columns)
        else:
            select_clause = ", ".join(ALL_COLUMNS)

        where_clauses: list[str] = []
        params: list[Any] = []

        if filters:
            for key, val in filters.items():
                if "__" in key:
                    col_name, op = key.split("__", 1)
                else:
                    col_name, op = key, "eq"

                if col_name not in ALLOWED_COLUMNS:
                    raise ValueError(
                        f"Disallowed filter column '{col_name}'. Allowed: {sorted(ALLOWED_COLUMNS)}"
                    )

                if op not in ALLOWED_OPERATORS:
                    raise ValueError(
                        f"Disallowed filter operator '{op}'. Allowed: {sorted(ALLOWED_OPERATORS)}"
                    )

                if op == "isnull":
                    if val:
                        where_clauses.append(f"{col_name} IS NULL")
                    else:
                        where_clauses.append(f"{col_name} IS NOT NULL")
                elif op == "in":
                    if not isinstance(val, (list, tuple, set)) or len(val) == 0:
                        where_clauses.append("1 = 0")  # Empty IN match nothing
                    else:
                        placeholders = ", ".join(["?"] * len(val))
                        where_clauses.append(f"{col_name} IN ({placeholders})")
                        params.extend(list(val))
                elif op == "eq" and val is None:
                    where_clauses.append(f"{col_name} IS NULL")
                elif op == "ne" and val is None:
                    where_clauses.append(f"{col_name} IS NOT NULL")
                else:
                    sql_op = OPERATOR_SQL_MAP.get(op, "=")
                    where_clauses.append(f"{col_name} {sql_op} ?")
                    params.append(val)

        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Validate order_by
        order_sql = ""
        if order_by is not None:
            if order_by not in ALLOWED_SORT_COLUMNS:
                raise ValueError(
                    f"Disallowed sort column '{order_by}'. Allowed: {sorted(ALLOWED_SORT_COLUMNS)}"
                )
            direction = "DESC" if descending else "ASC"
            order_sql = f" ORDER BY {order_by} {direction}"

        # Pagination
        limit_val = max(1, int(limit))
        offset_val = max(0, int(offset))
        limit_sql = f" LIMIT ? OFFSET ?"
        params.extend([limit_val, offset_val])

        full_sql = f"SELECT {select_clause} FROM {TABLE_NAME}{where_sql}{order_sql}{limit_sql};"

        with self._get_conn() as conn:
            rows = conn.execute(full_sql, params).fetchall()
            return [dict(row) for row in rows]


# =============================================================================
# 4. Module-Level Convenience Functions (using default db_path)
# =============================================================================


def get_critical_unresolved_count(db_path: str | Path | None = None) -> int:
    """Return count of unresolved critical tickets."""
    return TicketRepository(db_path).get_critical_unresolved_count()


def get_lowest_average_rating_agent(
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return agent with lowest average customer rating."""
    return TicketRepository(db_path).get_lowest_average_rating_agent()


def get_unresolved_over_24h_count(db_path: str | Path | None = None) -> int:
    """Return count of unresolved tickets older than 24h."""
    return TicketRepository(db_path).get_unresolved_over_24h_count()


def get_high_priority_unresolved_over_24h(
    limit: int = 100, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Return high-priority unresolved tickets older than 24h."""
    return TicketRepository(db_path).get_high_priority_unresolved_over_24h(limit=limit)


def get_average_resolution_time_by_priority(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return average resolution time by priority."""
    return TicketRepository(db_path).get_average_resolution_time_by_priority()


def get_ticket_counts_by_category(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return ticket counts grouped by category."""
    return TicketRepository(db_path).get_ticket_counts_by_category()


def get_ticket_counts_by_status(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return ticket counts grouped by status."""
    return TicketRepository(db_path).get_ticket_counts_by_status()


def get_average_customer_rating_by_agent(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return average customer rating by agent."""
    return TicketRepository(db_path).get_average_customer_rating_by_agent()


def get_ticket_count(db_path: str | Path | None = None) -> int:
    """Return total count of tickets."""
    return TicketRepository(db_path).get_ticket_count()


def get_unresolved_ticket_count(db_path: str | Path | None = None) -> int:
    """Return count of unresolved tickets."""
    return TicketRepository(db_path).get_unresolved_ticket_count()


def get_ticket_counts_by_priority(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return ticket counts by priority."""
    return TicketRepository(db_path).get_ticket_counts_by_priority()


def get_average_resolution_time(db_path: str | Path | None = None) -> float | None:
    """Return overall average resolution time."""
    return TicketRepository(db_path).get_average_resolution_time()


def get_average_response_time(db_path: str | Path | None = None) -> float | None:
    """Return overall average response time."""
    return TicketRepository(db_path).get_average_response_time()


def get_tickets_by_agent(
    agent_id: str, limit: int = 100, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Fetch tickets by agent."""
    return TicketRepository(db_path).get_tickets_by_agent(agent_id, limit=limit)


def get_tickets_by_priority(
    priority: str, limit: int = 100, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Fetch tickets by priority."""
    return TicketRepository(db_path).get_tickets_by_priority(priority, limit=limit)


def get_tickets_by_status(
    status: str, limit: int = 100, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Fetch tickets by status."""
    return TicketRepository(db_path).get_tickets_by_status(status, limit=limit)


def get_ticket_by_id(
    ticket_id: str, db_path: str | Path | None = None
) -> dict[str, Any] | None:
    """Fetch ticket by ID."""
    return TicketRepository(db_path).get_ticket_by_id(ticket_id)


def query_tickets(
    filters: dict[str, Any] | None = None,
    columns: list[str] | None = None,
    order_by: str | None = None,
    descending: bool = False,
    limit: int = 100,
    offset: int = 0,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Execute a parameterized query with strict column/operator whitelisting."""
    return TicketRepository(db_path).query_tickets(
        filters=filters,
        columns=columns,
        order_by=order_by,
        descending=descending,
        limit=limit,
        offset=offset,
    )

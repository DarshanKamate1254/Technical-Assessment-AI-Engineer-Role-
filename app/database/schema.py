"""Database Schema and Initialization Module for Customer Support Tickets.

Defines the SQLite DDL schema, table definitions, indexes, and initialization
routines for storing both raw cleaned fields and all Stage 3 engineered features.
"""

from __future__ import annotations

from pathlib import Path

from app.database.connection import DEFAULT_DB_PATH, get_db_connection

TABLE_NAME = "support_tickets"

SCHEMA_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    ticket_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL,
    response_time_hrs REAL NOT NULL,
    resolution_time_hrs REAL,
    agent_id TEXT NOT NULL,
    customer_rating REAL,
    issue_summary TEXT NOT NULL,
    is_resolved INTEGER NOT NULL,
    is_unresolved INTEGER NOT NULL,
    is_high_priority INTEGER NOT NULL,
    is_critical INTEGER NOT NULL,
    priority_level INTEGER NOT NULL,
    response_time_bucket TEXT NOT NULL,
    resolution_time_bucket TEXT,
    ticket_age_hours REAL NOT NULL,
    unresolved_age_hours REAL,
    is_unresolved_over_24h INTEGER NOT NULL,
    is_critical_unresolved INTEGER NOT NULL,
    has_customer_rating INTEGER NOT NULL,
    low_customer_rating INTEGER NOT NULL,
    has_resolution_time INTEGER NOT NULL,
    long_resolution INTEGER NOT NULL,
    priority_category TEXT NOT NULL
);
"""

INDEXES_DDL = [
    f"CREATE INDEX IF NOT EXISTS idx_tickets_status ON {TABLE_NAME} (status);",
    f"CREATE INDEX IF NOT EXISTS idx_tickets_priority ON {TABLE_NAME} (priority);",
    f"CREATE INDEX IF NOT EXISTS idx_tickets_agent_id ON {TABLE_NAME} (agent_id);",
    f"CREATE INDEX IF NOT EXISTS idx_tickets_category ON {TABLE_NAME} (category);",
    f"CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON {TABLE_NAME} (created_at);",
    f"CREATE INDEX IF NOT EXISTS idx_tickets_is_unresolved ON {TABLE_NAME} (is_unresolved);",
    f"CREATE INDEX IF NOT EXISTS idx_tickets_is_critical_unres ON {TABLE_NAME} (is_critical_unresolved);",
]

ALL_COLUMNS: tuple[str, ...] = (
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
    "is_resolved",
    "is_unresolved",
    "is_high_priority",
    "is_critical",
    "priority_level",
    "response_time_bucket",
    "resolution_time_bucket",
    "ticket_age_hours",
    "unresolved_age_hours",
    "is_unresolved_over_24h",
    "is_critical_unresolved",
    "has_customer_rating",
    "low_customer_rating",
    "has_resolution_time",
    "long_resolution",
    "priority_category",
)


def initialize_database(db_path: str | Path | None = None) -> None:
    """Idempotently initialize the SQLite database table and indexes.

    Args:
        db_path: Path to the SQLite database file. Defaults to DEFAULT_DB_PATH.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(SCHEMA_DDL)
        for index_ddl in INDEXES_DDL:
            cursor.execute(index_ddl)

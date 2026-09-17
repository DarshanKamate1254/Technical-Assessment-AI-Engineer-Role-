"""Database Connection Management Module for Customer Support Tickets.

Provides thread-safe connection pooling/factory and context managers for
SQLite database access with standard row factories and foreign key constraints.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Generator

DEFAULT_DB_PATH = Path("data/support_tickets.db")


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Create and return a configured SQLite connection.

    Args:
        db_path: Path to the SQLite database file. Defaults to DEFAULT_DB_PATH.

    Returns:
        Configured sqlite3.Connection with sqlite3.Row row_factory.
    """
    path_obj = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path_obj))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_db_connection(
    db_path: str | Path | None = None,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connections handling commits, rollbacks, and closure.

    Args:
        db_path: Path to the SQLite database file.

    Yields:
        sqlite3.Connection: Active database connection.
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

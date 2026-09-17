"""Database Loader Module for Customer Support Tickets.

Handles persisting validated and feature-engineered DataFrames into SQLite,
managing type coercions, NULL preservation, and atomic batch insertion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from app.database.connection import DEFAULT_DB_PATH, get_db_connection
from app.database.schema import ALL_COLUMNS, TABLE_NAME, initialize_database
from app.pipeline.feature_engineer import engineer_features
from app.pipeline.parser import parse_csv
from app.pipeline.pipeline import validate_and_clean


def prepare_records_for_sqlite(df: pd.DataFrame) -> list[tuple[Any, ...]]:
    """Convert DataFrame rows into SQLite-compatible Python tuples with preserved NULLs.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        List of tuples matching ALL_COLUMNS ordering.
    """
    records: list[tuple[Any, ...]] = []

    for _, row in df.iterrows():
        record_values: list[Any] = []
        for col in ALL_COLUMNS:
            val = row.get(col, None)

            # Preserve NULLs for NA / NaN / NaT
            if pd.isna(val) or val is pd.NA:
                record_values.append(None)
            elif col == "created_at" and isinstance(val, (pd.Timestamp, np.datetime64)):
                record_values.append(pd.to_datetime(val).strftime("%Y-%m-%d %H:%M:%S"))
            elif isinstance(val, (bool, np.bool_)):
                record_values.append(1 if val else 0)
            elif isinstance(val, (np.integer, int)):
                record_values.append(int(val))
            elif isinstance(val, (np.floating, float)):
                record_values.append(float(val))
            else:
                record_values.append(str(val))

        records.append(tuple(record_values))

    return records


def store_tickets(
    df: pd.DataFrame,
    db_path: str | Path | None = None,
    if_exists: str = "replace",
) -> int:
    """Persist the validated and feature-engineered DataFrame into SQLite.

    Policy:
        - If `if_exists == "replace"`, clears existing records before inserting.
        - Preserves all original and engineered columns.
        - Preserves NULL values.
        - Uses safe parameterized bulk inserts.

    Args:
        df: Input DataFrame containing all 26 required columns.
        db_path: SQLite database path. Defaults to DEFAULT_DB_PATH.
        if_exists: Action if table already contains data ("replace" or "append").

    Returns:
        Number of records inserted.
    """
    initialize_database(db_path)
    records = prepare_records_for_sqlite(df)

    columns_clause = ", ".join(ALL_COLUMNS)
    placeholders = ", ".join(["?"] * len(ALL_COLUMNS))
    insert_sql = (
        f"INSERT INTO {TABLE_NAME} ({columns_clause}) VALUES ({placeholders});"
    )

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if if_exists == "replace":
            cursor.execute(f"DELETE FROM {TABLE_NAME};")

        cursor.executemany(insert_sql, records)
        return len(records)


def build_database(
    df: pd.DataFrame,
    db_path: str | Path | None = None,
) -> int:
    """Build or rebuild the SQLite database from a feature-engineered DataFrame.

    Args:
        df: Feature-engineered DataFrame from Stage 3.
        db_path: SQLite database path. Defaults to DEFAULT_DB_PATH.

    Returns:
        Number of records stored in the database.
    """
    return store_tickets(df, db_path=db_path, if_exists="replace")


def build_database_from_csv(
    csv_path: str | Path,
    db_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], int]:
    """Execute the end-to-end pipeline from CSV to SQLite database.

    Pipeline:
        CSV -> Parse (Stage 1) -> Validate & Clean (Stage 2) -> Feature Engineer (Stage 3) -> Store (Stage 4)

    Args:
        csv_path: Path to the raw CSV file.
        db_path: Target SQLite database path.

    Returns:
        Tuple of (feature_engineered_df, validation_report, rows_stored).
    """
    raw_df = parse_csv(csv_path)
    cleaned_df, report = validate_and_clean(raw_df)
    featured_df = engineer_features(cleaned_df)
    rows_stored = build_database(featured_df, db_path=db_path)
    return featured_df, report, rows_stored

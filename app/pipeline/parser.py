"""Customer Support Tickets Data Parser Module.

Provides robust, modular, and reusable functions to load and parse CSV datasets
into appropriate Python/Pandas data types with full diagnostics and validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

EXPECTED_COLUMNS: tuple[str, ...] = (
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
)

STRING_COLUMNS: tuple[str, ...] = (
    "ticket_id",
    "category",
    "priority",
    "status",
    "agent_id",
    "issue_summary",
)

DATETIME_COLUMNS: tuple[str, ...] = ("created_at",)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "response_time_hrs",
    "resolution_time_hrs",
    "customer_rating",
)


class ParserError(Exception):
    """Base exception for parsing errors."""


class CSVReadError(ParserError):
    """Raised when the CSV file cannot be located, read, or decoded."""


class MissingColumnError(ParserError):
    """Raised when one or more expected columns are missing from the dataset."""


@dataclass(frozen=True)
class ParsingDiagnostics:
    """Encapsulates parsing metadata and diagnostic statistics."""

    file_name: str
    row_count: int
    column_count: int
    column_names: list[str]
    data_types: dict[str, str]
    missing_values_per_column: dict[str, int]
    failed_conversions_per_column: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostic metrics as a Python dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Return a formatted human-readable diagnostic report."""
        lines = [
            "=" * 60,
            "PARSING DIAGNOSTICS REPORT",
            "=" * 60,
            f"File Name      : {self.file_name}",
            f"Row Count      : {self.row_count}",
            f"Column Count   : {self.column_count}",
            "-" * 60,
            f"{'Column Name':<24} {'Parsed Type':<16} {'Missing':<10} {'Failed Conv':<10}",
            "-" * 60,
        ]
        for col in self.column_names:
            dtype_str = self.data_types.get(col, "unknown")
            missing = self.missing_values_per_column.get(col, 0)
            failed = self.failed_conversions_per_column.get(col, 0)
            lines.append(f"{col:<24} {dtype_str:<16} {missing:<10} {failed:<10}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def read_raw_csv(
    file_path: str | Path,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Read raw CSV file preserving all rows and column structures as strings.

    Args:
        file_path: Path to the CSV file.
        encoding: File character encoding (defaults to utf-8).

    Returns:
        Raw pandas DataFrame with all values read as strings.

    Raises:
        CSVReadError: If the file is missing, empty, or cannot be read/decoded.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise CSVReadError(f"CSV file not found: {file_path}")
    if not path_obj.is_file():
        raise CSVReadError(f"Specified path is not a file: {file_path}")
    if os.path.getsize(path_obj) == 0:
        raise CSVReadError(f"CSV file is empty: {file_path}")

    try:
        df_raw = pd.read_csv(
            path_obj,
            encoding=encoding,
            dtype=str,
            keep_default_na=False,
        )
        return df_raw
    except UnicodeDecodeError as err:
        raise CSVReadError(
            f"Failed to decode CSV file with {encoding} encoding: {err}"
        ) from err
    except pd.errors.EmptyDataError as err:
        raise CSVReadError(f"CSV file contains no readable data: {err}") from err
    except Exception as err:
        raise CSVReadError(f"Error reading CSV file '{file_path}': {err}") from err


def validate_columns(
    df: pd.DataFrame,
    expected_columns: Sequence[str] = EXPECTED_COLUMNS,
) -> None:
    """Verify that all required columns are present in the DataFrame.

    Args:
        df: DataFrame to validate.
        expected_columns: Collection of required column names.

    Raises:
        MissingColumnError: If any expected columns are absent.
    """
    actual_columns = set(df.columns)
    missing_columns = [col for col in expected_columns if col not in actual_columns]
    if missing_columns:
        raise MissingColumnError(
            f"Missing required columns in CSV: {missing_columns}"
        )


def convert_column_types(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert columns into appropriate Python/Pandas types and track conversion failures.

    Type specifications:
        - String columns: pandas StringDtype (`pd.StringDtype()`)
        - Datetime columns: `datetime64[ns]`
        - Numeric columns: `float64`

    Missing values (empty strings or NaNs) are preserved without imputation or zero-filling.
    Raw non-empty values that fail conversion to datetime or numeric are coerced to NaN/NaT
    and tracked in the failed conversion count.

    Args:
        df: Raw DataFrame containing string-typed columns.

    Returns:
        A tuple of (parsed DataFrame, dictionary of failed conversion counts per column).
    """
    parsed_df = df.copy()
    failed_conversions: dict[str, int] = {col: 0 for col in df.columns}

    # 1. Parse String columns
    for col in STRING_COLUMNS:
        if col in parsed_df.columns:
            series = parsed_df[col]
            # Replace empty strings with NA and cast to StringDtype
            parsed_df[col] = (
                series.replace("", pd.NA)
                if isinstance(series, pd.Series)
                else series
            ).astype("string")
            failed_conversions[col] = 0

    # 2. Parse Datetime columns safely
    for col in DATETIME_COLUMNS:
        if col in parsed_df.columns:
            raw_series = df[col].astype(str).str.strip()
            raw_has_value = raw_series.notna() & (raw_series != "")
            parsed_series = pd.to_datetime(parsed_df[col], errors="coerce")
            # Count values that were not blank but failed datetime parsing
            failed_count = int((raw_has_value & parsed_series.isna()).sum())
            failed_conversions[col] = failed_count
            parsed_df[col] = parsed_series

    # 3. Parse Numeric columns safely
    for col in NUMERIC_COLUMNS:
        if col in parsed_df.columns:
            raw_series = df[col].astype(str).str.strip()
            raw_has_value = raw_series.notna() & (raw_series != "")
            parsed_series = pd.to_numeric(parsed_df[col], errors="coerce").astype(
                "float64"
            )
            # Count values that were not blank but failed numeric parsing
            failed_count = int((raw_has_value & parsed_series.isna()).sum())
            failed_conversions[col] = failed_count
            parsed_df[col] = parsed_series

    return parsed_df, failed_conversions


def parse_csv_with_diagnostics(
    file_path: str | Path,
    expected_columns: Sequence[str] = EXPECTED_COLUMNS,
    encoding: str = "utf-8",
) -> tuple[pd.DataFrame, ParsingDiagnostics]:
    """Parse a support tickets CSV file and compute parsing diagnostics.

    Args:
        file_path: Path to the target CSV file.
        expected_columns: Optional sequence of required columns.
        encoding: Character encoding for the CSV file.

    Returns:
        A tuple of (parsed pd.DataFrame, ParsingDiagnostics).

    Raises:
        CSVReadError: If the CSV cannot be read or decoded.
        MissingColumnError: If required columns are missing.
    """
    raw_df = read_raw_csv(file_path, encoding=encoding)
    validate_columns(raw_df, expected_columns=expected_columns)
    parsed_df, failed_conversions = convert_column_types(raw_df)

    data_types = {col: str(parsed_df[col].dtype) for col in parsed_df.columns}
    missing_values = {col: int(parsed_df[col].isna().sum()) for col in parsed_df.columns}

    file_name = Path(file_path).name
    diagnostics = ParsingDiagnostics(
        file_name=file_name,
        row_count=len(parsed_df),
        column_count=len(parsed_df.columns),
        column_names=list(parsed_df.columns),
        data_types=data_types,
        missing_values_per_column=missing_values,
        failed_conversions_per_column=failed_conversions,
    )

    return parsed_df, diagnostics


def parse_csv(
    file_path: str | Path,
    expected_columns: Sequence[str] = EXPECTED_COLUMNS,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Parse a CSV file and convert raw CSV values into typed Python/Pandas representations.

    Args:
        file_path: Path to the CSV file.
        expected_columns: Optional sequence of required columns.
        encoding: Character encoding for the CSV file.

    Returns:
        Parsed DataFrame with properly typed columns.

    Raises:
        CSVReadError: If the CSV cannot be read or decoded.
        MissingColumnError: If required columns are missing.
    """
    parsed_df, _ = parse_csv_with_diagnostics(
        file_path, expected_columns=expected_columns, encoding=encoding
    )
    return parsed_df

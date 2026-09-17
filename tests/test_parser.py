"""Unit tests for the CSV data parsing module."""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from app.pipeline.parser import (
    CSVReadError,
    EXPECTED_COLUMNS,
    MissingColumnError,
    ParsingDiagnostics,
    convert_column_types,
    parse_csv,
    parse_csv_with_diagnostics,
    read_raw_csv,
    validate_columns,
)

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "support_tickets.csv"


def test_valid_csv_parsing_actual_dataset():
    """Test loading and parsing the actual support_tickets.csv dataset."""
    assert DATASET_PATH.exists(), f"Dataset file not found at {DATASET_PATH}"

    df, diag = parse_csv_with_diagnostics(DATASET_PATH)

    # 1. Row count & column count
    assert len(df) == 500, f"Expected 500 rows, got {len(df)}"
    assert len(df.columns) == 10, f"Expected 10 columns, got {len(df.columns)}"
    assert list(df.columns) == list(EXPECTED_COLUMNS)

    # 2. Diagnostics
    assert isinstance(diag, ParsingDiagnostics)
    assert diag.row_count == 500
    assert diag.column_count == 10
    assert diag.file_name == "support_tickets.csv"
    assert sum(diag.failed_conversions_per_column.values()) == 0

    # 3. Direct parse_csv output matches
    df_direct = parse_csv(DATASET_PATH)
    pd.testing.assert_frame_equal(df, df_direct)


def test_datetime_conversion():
    """Test that created_at column is converted to proper datetime format."""
    df = parse_csv(DATASET_PATH)

    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])
    assert df["created_at"].isna().sum() == 0

    # Test datetime properties
    min_date = df["created_at"].min()
    max_date = df["created_at"].max()
    assert min_date == pd.Timestamp("2024-01-01 08:54:00")
    assert max_date == pd.Timestamp("2024-03-30 18:06:00")
    assert (df["created_at"].dt.year == 2024).all()


def test_numeric_conversion():
    """Test that numeric fields are converted to float dtypes and support numeric operations."""
    df = parse_csv(DATASET_PATH)

    for col in ["response_time_hrs", "resolution_time_hrs", "customer_rating"]:
        assert pd.api.types.is_float_dtype(df[col]), f"{col} is not float dtype"

    # Verify min/max values
    assert df["response_time_hrs"].min() == pytest.approx(0.2)
    assert df["response_time_hrs"].max() == pytest.approx(5.0)

    # Only resolved tickets have resolution time and ratings
    resolved_df = df[df["status"] == "Resolved"]
    assert resolved_df["resolution_time_hrs"].min() == pytest.approx(1.2)
    assert resolved_df["resolution_time_hrs"].max() == pytest.approx(119.7)
    assert resolved_df["customer_rating"].min() == pytest.approx(1.0)
    assert resolved_df["customer_rating"].max() == pytest.approx(5.0)


def test_preservation_of_missing_values():
    """Test that missing values remain NaN and are NOT replaced with 0 or imputed."""
    df = parse_csv(DATASET_PATH)

    # Open and Escalated tickets should have missing resolution_time_hrs and customer_rating
    non_resolved_mask = df["status"].isin(["Open", "Escalated"])
    assert non_resolved_mask.sum() == 173

    # Check resolution_time_hrs missing values
    assert df["resolution_time_hrs"].isna().sum() == 173
    assert df.loc[non_resolved_mask, "resolution_time_hrs"].isna().all()

    # Check customer_rating missing values
    assert df["customer_rating"].isna().sum() == 173
    assert df.loc[non_resolved_mask, "customer_rating"].isna().all()

    # Crucial check: zeros are NOT inserted for missing values
    assert (df.loc[non_resolved_mask, "resolution_time_hrs"] == 0).sum() == 0
    assert (df.loc[non_resolved_mask, "customer_rating"] == 0).sum() == 0


def test_missing_required_column_detection():
    """Test that MissingColumnError is raised when required columns are absent."""
    csv_content = "ticket_id,created_at,category\nTKT-001,2024-01-01 10:00,General\n"

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp.write(csv_content)
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(MissingColumnError) as exc_info:
            parse_csv(tmp_path)
        assert "priority" in str(exc_info.value)
        assert "status" in str(exc_info.value)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_invalid_date_numeric_parsing_diagnostics():
    """Test that invalid dates and non-numeric values are safely coerced and tracked in diagnostics."""
    csv_content = (
        "ticket_id,created_at,category,priority,status,response_time_hrs,resolution_time_hrs,agent_id,customer_rating,issue_summary\n"
        "TKT-001,2024-02-05 11:14,General,Low,Resolved,3.7,7.8,AGT-03,4,Valid row\n"
        "TKT-002,INVALID_DATE_STR,Billing,Low,Resolved,CORRUPT_NUM,13.7,AGT-09,BAD_RATING,Invalid row\n"
        "TKT-003,2024-03-09 09:59,Billing,High,Open,0.6,,AGT-07,,Missing row\n"
    )

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp.write(csv_content)
        tmp_path = Path(tmp.name)

    try:
        df, diag = parse_csv_with_diagnostics(tmp_path)

        # Total rows preserved without dropping
        assert len(df) == 3
        assert diag.row_count == 3

        # Row 2 should have coerced NaNs
        assert pd.isna(df.loc[1, "created_at"])
        assert pd.isna(df.loc[1, "response_time_hrs"])
        assert pd.isna(df.loc[1, "customer_rating"])

        # Row 3 should have valid date, valid numeric, and legitimate NaNs
        assert not pd.isna(df.loc[2, "created_at"])
        assert df.loc[2, "response_time_hrs"] == pytest.approx(0.6)
        assert pd.isna(df.loc[2, "resolution_time_hrs"])
        assert pd.isna(df.loc[2, "customer_rating"])

        # Diagnostics must capture failed conversions (Row 2), but not legitimate blanks (Row 3)
        assert diag.failed_conversions_per_column["created_at"] == 1
        assert diag.failed_conversions_per_column["response_time_hrs"] == 1
        assert diag.failed_conversions_per_column["customer_rating"] == 1
        assert diag.failed_conversions_per_column["resolution_time_hrs"] == 0

        # Missing values count (both failed + empty)
        assert diag.missing_values_per_column["created_at"] == 1
        assert diag.missing_values_per_column["response_time_hrs"] == 1
        assert diag.missing_values_per_column["resolution_time_hrs"] == 1
        assert diag.missing_values_per_column["customer_rating"] == 2

        # Summary string formatting check
        summary_str = diag.summary()
        assert "PARSING DIAGNOSTICS REPORT" in summary_str
        assert "created_at" in summary_str
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_file_read_errors():
    """Test that CSVReadError is raised for non-existent and empty files."""
    # 1. Non-existent file
    with pytest.raises(CSVReadError) as exc_info:
        parse_csv("non_existent_file_xyz_123.csv")
    assert "not found" in str(exc_info.value).lower()

    # 2. Empty file
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(CSVReadError) as exc_info:
            parse_csv(tmp_path)
        assert "empty" in str(exc_info.value).lower()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_string_column_casting():
    """Test that string columns are cast to pandas string dtype."""
    df = parse_csv(DATASET_PATH)

    string_cols = [
        "ticket_id",
        "category",
        "priority",
        "status",
        "agent_id",
        "issue_summary",
    ]
    for col in string_cols:
        assert isinstance(df[col].dtype, pd.StringDtype) or df[col].dtype == "string"

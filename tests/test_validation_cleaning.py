"""Unit and integration tests for Stage 2: Data Validation and Cleaning."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.pipeline.cleaner import DataCleaner
from app.pipeline.parser import parse_csv
from app.pipeline.pipeline import (
    validate_and_clean,
    validate_and_clean_with_report,
)
from app.pipeline.validator import (
    REQUIRED_COLUMNS,
    DataValidator,
    SchemaValidationError,
    ValidationReport,
)

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "support_tickets.csv"


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Fixture returning a standard valid DataFrame for support tickets."""
    return pd.DataFrame(
        {
            "ticket_id": pd.Series(["TKT-001", "TKT-002", "TKT-003"], dtype="string"),
            "created_at": pd.to_datetime(
                ["2024-01-01 10:00:00", "2024-01-02 11:30:00", "2024-01-03 14:15:00"]
            ),
            "category": pd.Series(["General", "Billing", "Technical"], dtype="string"),
            "priority": pd.Series(["Low", "Medium", "High"], dtype="string"),
            "status": pd.Series(["Resolved", "Open", "Escalated"], dtype="string"),
            "response_time_hrs": pd.Series([1.5, 0.8, 2.3], dtype="float64"),
            "resolution_time_hrs": pd.Series([12.0, np.nan, np.nan], dtype="float64"),
            "agent_id": pd.Series(["AGT-01", "AGT-02", "AGT-03"], dtype="string"),
            "customer_rating": pd.Series([4.0, np.nan, np.nan], dtype="float64"),
            "issue_summary": pd.Series(
                ["Password reset request", "Invoice inquiry", "System error 500"],
                dtype="string",
            ),
        }
    )


# 1. Missing required column
def test_missing_required_column_raises_error(sample_valid_df):
    """Test that SchemaValidationError is raised when required columns are absent."""
    df_missing = sample_valid_df.drop(columns=["priority", "status"])

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_and_clean(df_missing)

    err_msg = str(exc_info.value)
    assert "priority" in err_msg
    assert "status" in err_msg


# 2. Duplicate ticket IDs
def test_duplicate_ticket_ids_handling(sample_valid_df):
    """Test that duplicate ticket IDs are detected, conflicting ones flagged, and first occurrence kept."""
    # Row 3 has same ticket_id as Row 0 but different category/issue (conflicting duplicate)
    row_conflict = sample_valid_df.iloc[0].copy()
    row_conflict["category"] = "Technical"
    row_conflict["issue_summary"] = "Different issue"

    df_with_dups = pd.concat([sample_valid_df, pd.DataFrame([row_conflict])], ignore_index=True)

    cleaned_df, report = validate_and_clean(df_with_dups)

    assert report["rows_before"] == 4
    assert report["rows_after"] == 3
    assert report["duplicate_rows_removed"] == 1
    assert report["duplicate_ticket_ids"] == 1
    assert "TKT-001" in report["conflicting_ticket_ids"]
    assert len(cleaned_df) == 3
    # Check that first occurrence is preserved
    assert cleaned_df.loc[0, "category"] == "General"


# 3. Exact duplicate rows
def test_exact_duplicate_rows_deduplication(sample_valid_df):
    """Test that identical duplicate rows are removed and counted properly."""
    df_with_exact_dup = pd.concat(
        [sample_valid_df, sample_valid_df.iloc[[0]]], ignore_index=True
    )

    cleaned_df, report = validate_and_clean(df_with_exact_dup)

    assert report["rows_before"] == 4
    assert report["rows_after"] == 3
    assert report["duplicate_rows_removed"] == 1
    assert report["duplicate_ticket_ids"] == 1
    assert len(report["conflicting_ticket_ids"]) == 0
    pd.testing.assert_frame_equal(cleaned_df, sample_valid_df)


# 4. Invalid category
def test_invalid_category_detection(sample_valid_df):
    """Test that invalid category values are detected and reported."""
    sample_valid_df.loc[0, "category"] = "UnknownCategory"

    _, report = validate_and_clean(sample_valid_df)

    assert report["invalid_categories"] == 1


# 5. Invalid priority
def test_invalid_priority_detection(sample_valid_df):
    """Test that invalid priority values are detected and reported."""
    sample_valid_df.loc[1, "priority"] = "UrgentPlus"

    _, report = validate_and_clean(sample_valid_df)

    assert report["invalid_priorities"] == 1


# 6. Invalid status
def test_invalid_status_detection(sample_valid_df):
    """Test that invalid status values are detected and reported."""
    sample_valid_df.loc[2, "status"] = "PendingReview"

    _, report = validate_and_clean(sample_valid_df)

    assert report["invalid_statuses"] == 1


# 7. Negative response time
def test_negative_response_time_detection(sample_valid_df):
    """Test that negative response_time_hrs values are detected as invalid."""
    sample_valid_df.loc[0, "response_time_hrs"] = -1.5

    _, report = validate_and_clean(sample_valid_df)

    assert report["invalid_response_times"] == 1


# 8. Negative resolution time
def test_negative_resolution_time_detection(sample_valid_df):
    """Test that negative resolution_time_hrs values are detected as invalid."""
    sample_valid_df.loc[0, "resolution_time_hrs"] = -5.0

    _, report = validate_and_clean(sample_valid_df)

    assert report["invalid_resolution_times"] == 1


# 9. Rating below 1
def test_rating_below_one_detection(sample_valid_df):
    """Test that customer_rating < 1.0 is detected as invalid."""
    sample_valid_df.loc[0, "customer_rating"] = 0.5

    _, report = validate_and_clean(sample_valid_df)

    assert report["invalid_ratings"] == 1


# 10. Rating above 5
def test_rating_above_five_detection(sample_valid_df):
    """Test that customer_rating > 5.0 is detected as invalid."""
    sample_valid_df.loc[0, "customer_rating"] = 5.5

    _, report = validate_and_clean(sample_valid_df)

    assert report["invalid_ratings"] == 1


# 11. Invalid date
def test_invalid_date_detection(sample_valid_df):
    """Test that missing / NaT dates are detected as invalid."""
    sample_valid_df.loc[0, "created_at"] = pd.NaT

    _, report = validate_and_clean(sample_valid_df)

    assert report["invalid_dates"] == 1


# 12. Preservation of legitimate missing values
def test_preservation_of_legitimate_missing_values(sample_valid_df):
    """Test that missing resolution_time_hrs and customer_rating remain NaN and are not filled with 0 or imputed."""
    cleaned_df, report = validate_and_clean(sample_valid_df)

    # Missing counts in report
    assert report["missing_values"]["resolution_time_hrs"] == 2
    assert report["missing_values"]["customer_rating"] == 2

    # Check that open/escalated tickets retain NaN and NOT 0.0
    assert pd.isna(cleaned_df.loc[1, "resolution_time_hrs"])
    assert pd.isna(cleaned_df.loc[1, "customer_rating"])
    assert pd.isna(cleaned_df.loc[2, "resolution_time_hrs"])
    assert pd.isna(cleaned_df.loc[2, "customer_rating"])

    # Ensure no imputation with 0 occurred
    assert (cleaned_df["resolution_time_hrs"] == 0).sum() == 0
    assert (cleaned_df["customer_rating"] == 0).sum() == 0


# 13. Resolved ticket with missing resolution time
def test_resolved_ticket_missing_resolution_time(sample_valid_df):
    """Test that a Resolved ticket with NULL resolution_time_hrs is flagged in cross_field_issues."""
    # Row 0 is Resolved, clear its resolution_time_hrs
    sample_valid_df.loc[0, "resolution_time_hrs"] = np.nan

    _, report = validate_and_clean(sample_valid_df)

    assert report["cross_field_issues"] == 1


# 14. Unresolved ticket with missing resolution time
def test_unresolved_ticket_missing_resolution_time_is_valid(sample_valid_df):
    """Test that Open and Escalated tickets with NULL resolution_time_hrs are valid and NOT flagged."""
    # Row 1 is Open (res_time is NaN), Row 2 is Escalated (res_time is NaN)
    # Row 0 is Resolved with resolution_time_hrs = 12.0
    _, report = validate_and_clean(sample_valid_df)

    assert report["cross_field_issues"] == 0


# 15. Whitespace normalization and safe casing
def test_whitespace_normalization_and_casing():
    """Test that leading/trailing whitespaces, consecutive spaces, and casing are normalized safely."""
    df_raw = pd.DataFrame(
        {
            "ticket_id": pd.Series(["  TKT-100  "], dtype="string"),
            "created_at": pd.to_datetime(["2024-02-01 10:00:00"]),
            "category": pd.Series(["  billing  "], dtype="string"),
            "priority": pd.Series(["  hIgh  "], dtype="string"),
            "status": pd.Series(["  reSolVed  "], dtype="string"),
            "response_time_hrs": pd.Series([2.0], dtype="float64"),
            "resolution_time_hrs": pd.Series([10.0], dtype="float64"),
            "agent_id": pd.Series(["  AGT-09  "], dtype="string"),
            "customer_rating": pd.Series([4.0], dtype="float64"),
            "issue_summary": pd.Series(
                ["  Payment   failed    with   code   404.  "],
                dtype="string",
            ),
        }
    )

    cleaned_df, report = validate_and_clean(df_raw)

    assert cleaned_df.loc[0, "ticket_id"] == "TKT-100"
    assert cleaned_df.loc[0, "category"] == "Billing"
    assert cleaned_df.loc[0, "priority"] == "High"
    assert cleaned_df.loc[0, "status"] == "Resolved"
    assert cleaned_df.loc[0, "agent_id"] == "AGT-09"
    assert cleaned_df.loc[0, "issue_summary"] == "Payment failed with code 404."
    assert report["invalid_categories"] == 0
    assert report["invalid_priorities"] == 0
    assert report["invalid_statuses"] == 0


# 16. Integration test on actual support_tickets.csv dataset
def test_validation_and_cleaning_actual_dataset():
    """Test end-to-end validation and cleaning on the actual support_tickets.csv dataset."""
    assert DATASET_PATH.exists(), f"Dataset file not found at {DATASET_PATH}"

    parsed_df = parse_csv(DATASET_PATH)
    cleaned_df, report = validate_and_clean(parsed_df)

    assert report["rows_before"] == 500
    assert report["rows_after"] == 500
    assert report["duplicate_rows_removed"] == 0
    assert report["duplicate_ticket_ids"] == 0
    assert len(report["conflicting_ticket_ids"]) == 0
    assert report["invalid_dates"] == 0
    assert report["invalid_response_times"] == 0
    assert report["invalid_resolution_times"] == 0
    assert report["invalid_ratings"] == 0
    assert report["invalid_categories"] == 0
    assert report["invalid_priorities"] == 0
    assert report["invalid_statuses"] == 0
    assert report["cross_field_issues"] == 0
    assert report["missing_values"]["resolution_time_hrs"] == 173
    assert report["missing_values"]["customer_rating"] == 173
    assert report["missing_values"]["ticket_id"] == 0
    assert report["missing_values"]["created_at"] == 0
    assert report["missing_values"]["category"] == 0
    assert report["missing_values"]["priority"] == 0
    assert report["missing_values"]["status"] == 0
    assert report["missing_values"]["response_time_hrs"] == 0
    assert report["missing_values"]["agent_id"] == 0
    assert report["missing_values"]["issue_summary"] == 0


# 17. ValidationReport object summary and to_dict methods
def test_validation_report_methods(sample_valid_df):
    """Test summary() and to_dict() methods on ValidationReport."""
    _, report_obj = validate_and_clean_with_report(sample_valid_df)

    assert isinstance(report_obj, ValidationReport)
    report_dict = report_obj.to_dict()
    assert isinstance(report_dict, dict)
    assert "rows_before" in report_dict
    assert "missing_values" in report_dict

    summary_text = report_obj.summary()
    assert "VALIDATION & CLEANING REPORT" in summary_text
    assert "Rows Before Deduplication" in summary_text

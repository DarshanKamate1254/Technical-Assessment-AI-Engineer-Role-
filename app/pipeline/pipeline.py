"""Pipeline Coordinator for Data Validation and Cleaning (Stage 2).

Provides the high-level orchestration function `validate_and_clean` to take a
parsed DataFrame, apply deterministic schema verification, duplicate resolution,
casing and whitespace normalization, business rule validation, and produce a
clean DataFrame along with a structured validation report.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.pipeline.cleaner import DataCleaner
from app.pipeline.validator import (
    DataValidator,
    SchemaValidationError,
    ValidationError,
    ValidationReport,
)


def validate_and_clean_with_report(
    df: pd.DataFrame,
    validator: DataValidator | None = None,
    cleaner: DataCleaner | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean a support tickets DataFrame, returning the cleaned

    DataFrame and a ValidationReport instance.

    Args:
        df: Input parsed DataFrame.
        validator: Optional custom DataValidator instance.
        cleaner: Optional custom DataCleaner instance.

    Returns:
        Tuple of (cleaned_df, ValidationReport).

    Raises:
        SchemaValidationError: If required columns are missing.
    """
    if validator is None:
        validator = DataValidator()
    if cleaner is None:
        cleaner = DataCleaner()

    # Step 1: Validate Schema
    validator.validate_schema(df)

    rows_before = len(df)

    # Step 2: Apply safe deterministic cleaning (whitespace, casing, deduplication)
    cleaned_df, removed_count, conflicting_ids, dup_id_count = cleaner.clean(df)
    rows_after = len(cleaned_df)

    # Step 3: Run comprehensive domain and business rule validations on cleaned data
    invalid_dates = validator.validate_dates(cleaned_df)
    invalid_resp = validator.validate_response_times(cleaned_df)
    invalid_res = validator.validate_resolution_times(cleaned_df)
    invalid_rat = validator.validate_ratings(cleaned_df)
    invalid_cats = validator.validate_categories(cleaned_df)
    invalid_priorities = validator.validate_priorities(cleaned_df)
    invalid_statuses = validator.validate_statuses(cleaned_df)
    missing_vals = validator.analyze_missing_values(cleaned_df)
    cross_field = validator.validate_cross_field(cleaned_df)

    report = ValidationReport(
        rows_before=rows_before,
        rows_after=rows_after,
        duplicate_rows_removed=removed_count,
        duplicate_ticket_ids=dup_id_count,
        invalid_dates=invalid_dates,
        invalid_response_times=invalid_resp,
        invalid_resolution_times=invalid_res,
        invalid_ratings=invalid_rat,
        invalid_categories=invalid_cats,
        invalid_priorities=invalid_priorities,
        invalid_statuses=invalid_statuses,
        missing_values=missing_vals,
        cross_field_issues=cross_field,
        conflicting_ticket_ids=conflicting_ids,
    )

    return cleaned_df, report


def validate_and_clean(
    df: pd.DataFrame,
    validator: DataValidator | None = None,
    cleaner: DataCleaner | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stage 2 primary interface.

    Validates and cleans the parsed DataFrame, returning a clean DataFrame
    and a structured validation dictionary.

    Args:
        df: Input parsed DataFrame.
        validator: Optional custom DataValidator instance.
        cleaner: Optional custom DataCleaner instance.

    Returns:
        Tuple of (cleaned_df, validation_report_dict).

    Raises:
        SchemaValidationError: If required columns are missing.
    """
    cleaned_df, report = validate_and_clean_with_report(
        df, validator=validator, cleaner=cleaner
    )
    return cleaned_df, report.to_dict()

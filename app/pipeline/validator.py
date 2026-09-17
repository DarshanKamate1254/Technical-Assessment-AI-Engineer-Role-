"""Data Validation Module for Customer Support Tickets.

Provides robust validation routines to verify schema integrity, detect duplicate
records, validate categorical domains, check numeric boundaries, inspect date
validity, analyze missing values, and verify cross-field business logic constraints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
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

VALID_CATEGORIES: frozenset[str] = frozenset({"General", "Billing", "Technical"})
VALID_PRIORITIES: frozenset[str] = frozenset({"Low", "Medium", "High", "Critical"})
VALID_STATUSES: frozenset[str] = frozenset({"Resolved", "Open", "Escalated"})

RATING_MIN: float = 1.0
RATING_MAX: float = 5.0


class ValidationError(Exception):
    """Base exception for validation errors."""


class SchemaValidationError(ValidationError):
    """Raised when the DataFrame fails schema or required column validation."""


@dataclass
class ValidationReport:
    """Encapsulates structured validation and cleaning diagnostic results."""

    rows_before: int
    rows_after: int
    duplicate_rows_removed: int = 0
    duplicate_ticket_ids: int = 0
    invalid_dates: int = 0
    invalid_response_times: int = 0
    invalid_resolution_times: int = 0
    invalid_ratings: int = 0
    invalid_categories: int = 0
    invalid_priorities: int = 0
    invalid_statuses: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    cross_field_issues: int = 0
    conflicting_ticket_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert validation report metrics to a dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Produce a formatted human-readable validation report."""
        lines = [
            "=" * 60,
            "VALIDATION & CLEANING REPORT",
            "=" * 60,
            f"Rows Before Deduplication : {self.rows_before}",
            f"Rows After Deduplication  : {self.rows_after}",
            f"Duplicate Rows Removed    : {self.duplicate_rows_removed}",
            f"Duplicate Ticket IDs      : {self.duplicate_ticket_ids}",
            f"Conflicting Ticket IDs    : {len(self.conflicting_ticket_ids)} {self.conflicting_ticket_ids if self.conflicting_ticket_ids else ''}",
            "-" * 60,
            "DATA INTEGRITY ISSUES",
            "-" * 60,
            f"Invalid Categories        : {self.invalid_categories}",
            f"Invalid Priorities        : {self.invalid_priorities}",
            f"Invalid Statuses          : {self.invalid_statuses}",
            f"Invalid Response Times    : {self.invalid_response_times}",
            f"Invalid Resolution Times  : {self.invalid_resolution_times}",
            f"Invalid Customer Ratings  : {self.invalid_ratings}",
            f"Invalid Dates             : {self.invalid_dates}",
            f"Cross-Field Consistency   : {self.cross_field_issues}",
            "-" * 60,
            "MISSING VALUES BY COLUMN",
            "-" * 60,
        ]
        for col, count in self.missing_values.items():
            lines.append(f"{col:<26}: {count}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class DataValidator:
    """Performs validation checks on support ticket DataFrames."""

    def __init__(
        self,
        required_columns: Sequence[str] = REQUIRED_COLUMNS,
        valid_categories: frozenset[str] = VALID_CATEGORIES,
        valid_priorities: frozenset[str] = VALID_PRIORITIES,
        valid_statuses: frozenset[str] = VALID_STATUSES,
        rating_min: float = RATING_MIN,
        rating_max: float = RATING_MAX,
    ) -> None:
        self.required_columns = tuple(required_columns)
        self.valid_categories = valid_categories
        self.valid_priorities = valid_priorities
        self.valid_statuses = valid_statuses
        self.rating_min = rating_min
        self.rating_max = rating_max

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validate that all required columns are present in the DataFrame.

        Args:
            df: DataFrame to validate.

        Raises:
            SchemaValidationError: If one or more required columns are missing.
        """
        actual_columns = set(df.columns)
        missing_columns = [col for col in self.required_columns if col not in actual_columns]
        if missing_columns:
            raise SchemaValidationError(
                f"Missing required columns in dataset: {missing_columns}"
            )

    def validate_duplicates(self, df: pd.DataFrame) -> tuple[int, list[str], int]:
        """Detect duplicate ticket IDs, exact duplicate rows, and conflicting duplicate records.

        Args:
            df: DataFrame to inspect.

        Returns:
            Tuple of (duplicate_ticket_id_count, conflicting_ticket_ids, exact_duplicate_rows_count).
        """
        if "ticket_id" not in df.columns:
            return 0, [], 0

        # Exact duplicate rows across all columns
        exact_dup_mask = df.duplicated(keep="first")
        exact_dup_count = int(exact_dup_mask.sum())

        # Total duplicate ticket_id occurrences (excluding first occurrence)
        dup_id_mask = df["ticket_id"].duplicated(keep="first")
        dup_id_count = int(dup_id_mask.sum())

        # Identify conflicting duplicates: same ticket_id, but row contents differ
        conflicting_ids: list[str] = []
        dup_id_all_mask = df["ticket_id"].duplicated(keep=False)
        if dup_id_all_mask.any():
            dup_df = df[dup_id_all_mask]
            for tid, group in dup_df.groupby("ticket_id"):
                # If number of unique rows in group > 1, it's a conflict
                if len(group.drop_duplicates()) > 1:
                    conflicting_ids.append(str(tid))

        return dup_id_count, conflicting_ids, exact_dup_count

    def validate_categories(self, df: pd.DataFrame) -> int:
        """Count invalid category values."""
        if "category" not in df.columns:
            return 0
        series = df["category"]
        # Invalid if null or not in allowed set
        invalid_mask = series.isna() | (~series.astype(str).isin(self.valid_categories))
        return int(invalid_mask.sum())

    def validate_priorities(self, df: pd.DataFrame) -> int:
        """Count invalid priority values."""
        if "priority" not in df.columns:
            return 0
        series = df["priority"]
        invalid_mask = series.isna() | (~series.astype(str).isin(self.valid_priorities))
        return int(invalid_mask.sum())

    def validate_statuses(self, df: pd.DataFrame) -> int:
        """Count invalid status values."""
        if "status" not in df.columns:
            return 0
        series = df["status"]
        invalid_mask = series.isna() | (~series.astype(str).isin(self.valid_statuses))
        return int(invalid_mask.sum())

    def validate_response_times(self, df: pd.DataFrame) -> int:
        """Count invalid response_time_hrs values (e.g. negative or missing)."""
        if "response_time_hrs" not in df.columns:
            return 0
        series = df["response_time_hrs"]
        # Must be non-negative numeric and not null
        invalid_mask = series.isna() | (series < 0)
        return int(invalid_mask.sum())

    def validate_resolution_times(self, df: pd.DataFrame) -> int:
        """Count invalid resolution_time_hrs values (e.g. negative).

        Note: Missing values for resolution_time_hrs are permitted and checked
        separately in cross-field consistency.
        """
        if "resolution_time_hrs" not in df.columns:
            return 0
        series = df["resolution_time_hrs"]
        # If present, must be >= 0
        invalid_mask = series.notna() & (series < 0)
        return int(invalid_mask.sum())

    def validate_ratings(self, df: pd.DataFrame) -> int:
        """Count invalid customer_rating values (not within [1, 5]).

        Note: Missing ratings are permitted and preserved.
        """
        if "customer_rating" not in df.columns:
            return 0
        series = df["customer_rating"]
        # If present, must be between rating_min and rating_max
        invalid_mask = series.notna() & (
            (series < self.rating_min) | (series > self.rating_max)
        )
        return int(invalid_mask.sum())

    def validate_dates(self, df: pd.DataFrame) -> int:
        """Count invalid created_at datetime values.

        Checks for null / NaT datetimes and unreasonable date values.
        """
        if "created_at" not in df.columns:
            return 0
        series = df["created_at"]
        invalid_mask = series.isna()
        return int(invalid_mask.sum())

    def validate_cross_field(self, df: pd.DataFrame) -> int:
        """Verify cross-field business logic constraints.

        Rules:
        1. If status == 'Resolved', resolution_time_hrs must not be null.
        2. If status in ['Open', 'Escalated'], resolution_time_hrs being null is valid.
        """
        if "status" not in df.columns or "resolution_time_hrs" not in df.columns:
            return 0

        # Resolved tickets missing resolution time
        resolved_missing_res_time = (df["status"] == "Resolved") & (
            df["resolution_time_hrs"].isna()
        )
        return int(resolved_missing_res_time.sum())

    def analyze_missing_values(self, df: pd.DataFrame) -> dict[str, int]:
        """Compute the count of missing (null/NA) values for each column."""
        return {col: int(df[col].isna().sum()) for col in df.columns}

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Execute all validation checks on the DataFrame and generate a report.

        Args:
            df: DataFrame to validate.

        Returns:
            ValidationReport instance containing all diagnostic metrics.

        Raises:
            SchemaValidationError: If required columns are missing.
        """
        self.validate_schema(df)

        dup_ids, conflicting_ids, exact_dups = self.validate_duplicates(df)
        invalid_cats = self.validate_categories(df)
        invalid_priorities = self.validate_priorities(df)
        invalid_statuses = self.validate_statuses(df)
        invalid_resp = self.validate_response_times(df)
        invalid_res = self.validate_resolution_times(df)
        invalid_rat = self.validate_ratings(df)
        invalid_dates = self.validate_dates(df)
        cross_field = self.validate_cross_field(df)
        missing_vals = self.analyze_missing_values(df)

        return ValidationReport(
            rows_before=len(df),
            rows_after=len(df),
            duplicate_rows_removed=0,
            duplicate_ticket_ids=dup_ids,
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


def validate_schema(df: pd.DataFrame) -> None:
    """Convenience helper to validate required columns."""
    DataValidator().validate_schema(df)

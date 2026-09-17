"""Data Cleaning Module for Customer Support Tickets.

Provides safe, deterministic data cleaning routines:
- Whitespace stripping and normalization
- Safe categorical casing standardization
- Deterministic deduplication (keeping first occurrence, tracking conflicts)
- Preservation of legitimate missing values
"""

from __future__ import annotations

import pandas as pd

from app.pipeline.validator import (
    REQUIRED_COLUMNS,
    VALID_CATEGORIES,
    VALID_PRIORITIES,
    VALID_STATUSES,
)

# Canonical casing mappings (lowercase -> canonical)
CATEGORY_MAP: dict[str, str] = {val.lower(): val for val in VALID_CATEGORIES}
PRIORITY_MAP: dict[str, str] = {val.lower(): val for val in VALID_PRIORITIES}
STATUS_MAP: dict[str, str] = {val.lower(): val for val in VALID_STATUSES}


class DataCleaner:
    """Performs deterministic cleaning and normalization on support ticket DataFrames."""

    def __init__(
        self,
        category_map: dict[str, str] = CATEGORY_MAP,
        priority_map: dict[str, str] = PRIORITY_MAP,
        status_map: dict[str, str] = STATUS_MAP,
    ) -> None:
        self.category_map = category_map
        self.priority_map = priority_map
        self.status_map = status_map

    def normalize_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace on all string columns and collapse

        consecutive whitespace in text fields.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with normalized whitespace.
        """
        cleaned_df = df.copy()

        # Normalize all string or object columns
        for col in cleaned_df.columns:
            orig_dtype = cleaned_df[col].dtype
            if pd.api.types.is_string_dtype(orig_dtype) or pd.api.types.is_object_dtype(
                orig_dtype
            ):
                stripped = cleaned_df[col].astype("string").str.strip()
                cleaned_df[col] = stripped.astype(orig_dtype)

        # Collapse internal multiple whitespace in issue_summary
        if "issue_summary" in cleaned_df.columns:
            orig_dtype = cleaned_df["issue_summary"].dtype
            collapsed = (
                cleaned_df["issue_summary"]
                .astype("string")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            cleaned_df["issue_summary"] = collapsed.astype(orig_dtype)

        return cleaned_df

    def normalize_categorical_casing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize casing for known categorical values.

        If a value matches a canonical category, priority, or status (case-insensitively),
        it is mapped to the canonical PascalCase value. Unrecognized values are left as-is
        for validation reporting.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with normalized categorical casing.
        """
        cleaned_df = df.copy()

        for col, mapping in (
            ("category", self.category_map),
            ("priority", self.priority_map),
            ("status", self.status_map),
        ):
            if col in cleaned_df.columns:
                orig_dtype = cleaned_df[col].dtype
                s_clean = cleaned_df[col].astype("string").str.strip()
                s_lower = s_clean.str.lower()
                s_mapped = s_lower.map(mapping)
                # If s_mapped is NA but original was not NA (unrecognized value), retain s_clean
                retained = s_mapped.where(s_mapped.notna() | s_clean.isna(), s_clean)
                cleaned_df[col] = retained.astype(orig_dtype)

        return cleaned_df

    def deduplicate(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, int, list[str], int]:
        """Apply deterministic deduplication policy.

        Policy:
        - Keep the first occurrence of each ticket_id.
        - Remove subsequent exact duplicate rows.
        - Remove subsequent conflicting duplicate rows while tracking their ticket_ids.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (deduplicated_df, duplicate_rows_removed_count, conflicting_ticket_ids, duplicate_ticket_ids_count).
        """
        if "ticket_id" not in df.columns or len(df) == 0:
            return df.copy(), 0, [], 0

        # Exact duplicate rows
        exact_dup_mask = df.duplicated(keep="first")
        exact_dup_count = int(exact_dup_mask.sum())

        # Redundant duplicate ticket_ids
        dup_id_mask = df["ticket_id"].duplicated(keep="first")
        dup_id_count = int(dup_id_mask.sum())

        # Detect conflicting duplicates
        conflicting_ids: list[str] = []
        dup_id_all_mask = df["ticket_id"].duplicated(keep=False)
        if dup_id_all_mask.any():
            dup_df = df[dup_id_all_mask]
            for tid, group in dup_df.groupby("ticket_id"):
                if len(group.drop_duplicates()) > 1:
                    conflicting_ids.append(str(tid))

        # Deterministic deduplication: keep first occurrence of ticket_id
        cleaned_df = df.drop_duplicates(subset=["ticket_id"], keep="first").reset_index(
            drop=True
        )
        rows_removed = len(df) - len(cleaned_df)

        return cleaned_df, rows_removed, conflicting_ids, dup_id_count

    def clean(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, int, list[str], int]:
        """Run all cleaning steps in order.

        1. Whitespace normalization
        2. Categorical casing normalization
        3. Deduplication

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (cleaned_df, duplicate_rows_removed, conflicting_ticket_ids, duplicate_ticket_ids).
        """
        df_step1 = self.normalize_whitespace(df)
        df_step2 = self.normalize_categorical_casing(df_step1)
        cleaned_df, removed_count, conflicting_ids, dup_id_count = self.deduplicate(df_step2)
        return cleaned_df, removed_count, conflicting_ids, dup_id_count

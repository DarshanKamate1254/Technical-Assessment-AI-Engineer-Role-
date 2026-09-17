"""Feature Engineering Module for Customer Support Tickets.

Provides deterministic, explainable feature engineering routines on top of
validated and cleaned support ticket datasets to support analytics, NLP querying,
SLA monitoring, and anomaly detection.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

PRIORITY_LEVEL_MAP: dict[str, int] = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}

RESPONSE_TIME_BINS: list[float] = [-float("inf"), 1.0, 4.0, 12.0, 24.0, float("inf")]
RESPONSE_TIME_LABELS: list[str] = ["<1h", "1-4h", "4-12h", "12-24h", ">24h"]

RESOLUTION_TIME_BINS: list[float] = [-float("inf"), 4.0, 12.0, 24.0, 48.0, float("inf")]
RESOLUTION_TIME_LABELS: list[str] = ["<4h", "4-12h", "12-24h", "24-48h", ">48h"]

FEATURE_METADATA: dict[str, dict[str, str]] = {
    "is_resolved": {
        "type": "boolean",
        "description": "Whether the ticket status is Resolved (True for Resolved, False otherwise)",
    },
    "is_unresolved": {
        "type": "boolean",
        "description": "Whether the ticket status is Open or Escalated",
    },
    "is_high_priority": {
        "type": "boolean",
        "description": "Whether the ticket priority is High or Critical",
    },
    "is_critical": {
        "type": "boolean",
        "description": "Whether the ticket priority is Critical",
    },
    "priority_level": {
        "type": "integer",
        "description": "Numeric priority ranking: Low (1), Medium (2), High (3), Critical (4)",
    },
    "response_time_bucket": {
        "type": "string",
        "description": "Categorical response time interval (<1h, 1-4h, 4-12h, 12-24h, >24h)",
    },
    "resolution_time_bucket": {
        "type": "string",
        "description": "Categorical resolution time interval (<4h, 4-12h, 12-24h, 24-48h, >48h) or NULL for unresolved",
    },
    "ticket_age_hours": {
        "type": "float",
        "description": "Hours between ticket creation and dataset reference timestamp",
    },
    "unresolved_age_hours": {
        "type": "float",
        "description": "Hours since creation for unresolved tickets; NULL for resolved tickets",
    },
    "is_unresolved_over_24h": {
        "type": "boolean",
        "description": "Whether the ticket is unresolved and older than 24 hours relative to reference time",
    },
    "is_critical_unresolved": {
        "type": "boolean",
        "description": "Whether the ticket is Critical priority and currently unresolved",
    },
    "has_customer_rating": {
        "type": "boolean",
        "description": "Whether a customer rating is recorded for the ticket",
    },
    "low_customer_rating": {
        "type": "boolean",
        "description": "Whether the customer rating is strictly below 3.0 (False if rating is missing)",
    },
    "has_resolution_time": {
        "type": "boolean",
        "description": "Whether a resolution time is recorded for the ticket",
    },
    "long_resolution": {
        "type": "boolean",
        "description": "Whether the resolution time strictly exceeds 48 hours (False if resolution time is missing)",
    },
    "priority_category": {
        "type": "string",
        "description": "Composite interaction feature combining priority and category (e.g. Critical_Technical)",
    },
}


class FeatureEngineer:
    """Transforms validated/cleaned support ticket data into feature-enriched DataFrames."""

    def __init__(
        self,
        priority_level_map: dict[str, int] = PRIORITY_LEVEL_MAP,
        response_time_bins: list[float] = RESPONSE_TIME_BINS,
        response_time_labels: list[str] = RESPONSE_TIME_LABELS,
        resolution_time_bins: list[float] = RESOLUTION_TIME_BINS,
        resolution_time_labels: list[str] = RESOLUTION_TIME_LABELS,
        feature_metadata: dict[str, dict[str, str]] = FEATURE_METADATA,
    ) -> None:
        self.priority_level_map = priority_level_map
        self.response_time_bins = response_time_bins
        self.response_time_labels = response_time_labels
        self.resolution_time_bins = resolution_time_bins
        self.resolution_time_labels = resolution_time_labels
        self.feature_metadata = feature_metadata

    def get_reference_timestamp(
        self, df: pd.DataFrame, custom_reference: pd.Timestamp | None = None
    ) -> pd.Timestamp:
        """Determine the deterministic reference timestamp.

        If custom_reference is supplied, it is returned. Otherwise, derives the
        maximum non-null timestamp in `created_at`.
        """
        if custom_reference is not None:
            return pd.to_datetime(custom_reference)

        if "created_at" in df.columns and df["created_at"].notna().any():
            return pd.to_datetime(df["created_at"].dropna().max())

        return pd.Timestamp.now()

    def engineer_features(
        self,
        df: pd.DataFrame,
        reference_time: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Derive all deterministic features for the input DataFrame.

        Preserves all existing columns and appends newly engineered features.

        Args:
            df: Validated and cleaned support tickets DataFrame.
            reference_time: Optional explicit reference timestamp for temporal features.
                           If None, defaults to `max(created_at)`.

        Returns:
            New DataFrame with original columns preserved and engineered features appended.
        """
        if len(df) == 0:
            featured_df = df.copy()
            for col in self.feature_metadata:
                featured_df[col] = pd.Series(dtype="object")
            return featured_df

        featured_df = df.copy()

        # 1. Resolution Status Features
        status_col = (
            featured_df["status"].astype("string")
            if "status" in featured_df.columns
            else pd.Series(dtype="string")
        )
        is_resolved = (status_col == "Resolved").fillna(False).astype(bool)
        is_unresolved = status_col.isin(["Open", "Escalated"]).fillna(False).astype(bool)
        featured_df["is_resolved"] = is_resolved
        featured_df["is_unresolved"] = is_unresolved

        # 2. Priority Features
        priority_col = (
            featured_df["priority"].astype("string")
            if "priority" in featured_df.columns
            else pd.Series(dtype="string")
        )
        is_high_priority = priority_col.isin(["High", "Critical"]).fillna(False).astype(bool)
        is_critical = (priority_col == "Critical").fillna(False).astype(bool)
        priority_level = priority_col.map(self.priority_level_map).astype("Int64")
        featured_df["is_high_priority"] = is_high_priority
        featured_df["is_critical"] = is_critical
        featured_df["priority_level"] = priority_level

        # 3. Response-Time Features
        if "response_time_hrs" in featured_df.columns:
            resp_bucket = pd.cut(
                featured_df["response_time_hrs"],
                bins=self.response_time_bins,
                labels=self.response_time_labels,
                right=False,
            ).astype("string")
            featured_df["response_time_bucket"] = resp_bucket
        else:
            featured_df["response_time_bucket"] = pd.Series(
                pd.NA, index=featured_df.index, dtype="string"
            )

        # 4. Resolution-Time Features
        if "resolution_time_hrs" in featured_df.columns:
            res_bucket = pd.cut(
                featured_df["resolution_time_hrs"],
                bins=self.resolution_time_bins,
                labels=self.resolution_time_labels,
                right=False,
            ).astype("string")
            featured_df["resolution_time_bucket"] = res_bucket
        else:
            featured_df["resolution_time_bucket"] = pd.Series(
                pd.NA, index=featured_df.index, dtype="string"
            )

        # 5. Temporal Age Features
        ref_timestamp = self.get_reference_timestamp(featured_df, custom_reference=reference_time)
        if "created_at" in featured_df.columns and pd.api.types.is_datetime64_any_dtype(
            featured_df["created_at"]
        ):
            created_series = featured_df["created_at"]
            age_seconds = (ref_timestamp - created_series).dt.total_seconds()
            ticket_age_hours = (age_seconds / 3600.0).clip(lower=0.0)
            ticket_age_hours = ticket_age_hours.where(created_series.notna(), other=np.nan)
        else:
            ticket_age_hours = pd.Series(np.nan, index=featured_df.index, dtype="float64")

        featured_df["ticket_age_hours"] = ticket_age_hours.astype("float64")

        # 6. Unresolved Age
        unresolved_age_hours = ticket_age_hours.where(is_unresolved, other=np.nan)
        featured_df["unresolved_age_hours"] = unresolved_age_hours.astype("float64")

        # 7. SLA-Related Features
        is_unresolved_over_24h = (
            is_unresolved & (unresolved_age_hours > 24.0).fillna(False)
        ).astype(bool)
        is_critical_unresolved = (is_critical & is_unresolved).astype(bool)
        featured_df["is_unresolved_over_24h"] = is_unresolved_over_24h
        featured_df["is_critical_unresolved"] = is_critical_unresolved

        # 8. Rating Features
        if "customer_rating" in featured_df.columns:
            rating_series = featured_df["customer_rating"]
            has_customer_rating = rating_series.notna().astype(bool)
            low_customer_rating = (has_customer_rating & (rating_series < 3.0)).astype(bool)
        else:
            has_customer_rating = pd.Series(False, index=featured_df.index, dtype=bool)
            low_customer_rating = pd.Series(False, index=featured_df.index, dtype=bool)

        featured_df["has_customer_rating"] = has_customer_rating
        featured_df["low_customer_rating"] = low_customer_rating

        # 9. Resolution Performance Features
        if "resolution_time_hrs" in featured_df.columns:
            res_series = featured_df["resolution_time_hrs"]
            has_resolution_time = res_series.notna().astype(bool)
            long_resolution = (has_resolution_time & (res_series > 48.0)).astype(bool)
        else:
            has_resolution_time = pd.Series(False, index=featured_df.index, dtype=bool)
            long_resolution = pd.Series(False, index=featured_df.index, dtype=bool)

        featured_df["has_resolution_time"] = has_resolution_time
        featured_df["long_resolution"] = long_resolution

        # 10. Category / Priority Interaction Features
        if "priority" in featured_df.columns and "category" in featured_df.columns:
            priority_str = featured_df["priority"].astype("string")
            category_str = featured_df["category"].astype("string")
            priority_category = (priority_str + "_" + category_str).astype("string")
        else:
            priority_category = pd.Series(pd.NA, index=featured_df.index, dtype="string")

        featured_df["priority_category"] = priority_category

        return featured_df


def engineer_features(
    df: pd.DataFrame,
    reference_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Stage 3 Primary Interface.

    Takes a validated and cleaned DataFrame and generates all engineered features.

    Args:
        df: Cleaned support tickets DataFrame.
        reference_time: Optional reference timestamp. Defaults to `max(created_at)`.

    Returns:
        DataFrame enriched with all engineered features.
    """
    return FeatureEngineer().engineer_features(df, reference_time=reference_time)

"""Unit and integration tests for Stage 3: Feature Engineering."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.pipeline.feature_engineer import (
    FEATURE_METADATA,
    PRIORITY_LEVEL_MAP,
    FeatureEngineer,
    engineer_features,
)
from app.pipeline.parser import parse_csv
from app.pipeline.pipeline import validate_and_clean

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "support_tickets.csv"


@pytest.fixture
def sample_cleaned_df() -> pd.DataFrame:
    """Fixture returning a standard cleaned DataFrame for testing feature engineering."""
    ref_time = pd.Timestamp("2024-03-30 18:00:00")
    return pd.DataFrame(
        {
            "ticket_id": pd.Series(["TKT-001", "TKT-002", "TKT-003", "TKT-004"], dtype="string"),
            "created_at": pd.to_datetime(
                [
                    ref_time - pd.Timedelta(hours=10),   # 10 hours ago
                    ref_time - pd.Timedelta(hours=30),   # 30 hours ago (>24h)
                    ref_time - pd.Timedelta(hours=50),   # 50 hours ago (>48h)
                    ref_time,                            # 0 hours ago (exact ref time)
                ]
            ),
            "category": pd.Series(["General", "Billing", "Technical", "Billing"], dtype="string"),
            "priority": pd.Series(["Low", "Medium", "High", "Critical"], dtype="string"),
            "status": pd.Series(["Resolved", "Open", "Escalated", "Open"], dtype="string"),
            "response_time_hrs": pd.Series([0.5, 2.0, 8.0, 26.0], dtype="float64"),
            "resolution_time_hrs": pd.Series([10.0, np.nan, np.nan, np.nan], dtype="float64"),
            "agent_id": pd.Series(["AGT-01", "AGT-02", "AGT-03", "AGT-04"], dtype="string"),
            "customer_rating": pd.Series([2.0, np.nan, np.nan, np.nan], dtype="float64"),
            "issue_summary": pd.Series(
                ["General question", "Billing dispute", "API timeout", "Critical payment outage"],
                dtype="string",
            ),
        }
    )


# 1. is_resolved
def test_is_resolved_feature(sample_cleaned_df):
    """Test is_resolved returns True only for Resolved status."""
    featured_df = engineer_features(sample_cleaned_df)
    assert featured_df["is_resolved"].tolist() == [True, False, False, False]


# 2. is_unresolved
def test_is_unresolved_feature(sample_cleaned_df):
    """Test is_unresolved returns True for Open and Escalated statuses."""
    featured_df = engineer_features(sample_cleaned_df)
    assert featured_df["is_unresolved"].tolist() == [False, True, True, True]


# 3. Priority level mapping
def test_priority_level_mapping(sample_cleaned_df):
    """Test priority_level maps Low->1, Medium->2, High->3, Critical->4."""
    featured_df = engineer_features(sample_cleaned_df)
    assert featured_df["priority_level"].tolist() == [1, 2, 3, 4]


# 4. is_high_priority
def test_is_high_priority_feature(sample_cleaned_df):
    """Test is_high_priority returns True for High and Critical priorities."""
    featured_df = engineer_features(sample_cleaned_df)
    assert featured_df["is_high_priority"].tolist() == [False, False, True, True]


# 5. is_critical
def test_is_critical_feature(sample_cleaned_df):
    """Test is_critical returns True exclusively for Critical priority."""
    featured_df = engineer_features(sample_cleaned_df)
    assert featured_df["is_critical"].tolist() == [False, False, False, True]


# 6. Ticket age calculation
def test_ticket_age_hours_calculation(sample_cleaned_df):
    """Test ticket_age_hours equals hours between created_at and reference_time and is non-negative."""
    ref_time = pd.Timestamp("2024-03-30 18:00:00")
    featured_df = engineer_features(sample_cleaned_df, reference_time=ref_time)

    expected_ages = [10.0, 30.0, 50.0, 0.0]
    np.testing.assert_allclose(featured_df["ticket_age_hours"], expected_ages)
    assert (featured_df["ticket_age_hours"] >= 0.0).all()


# 7. Unresolved age calculation
def test_unresolved_age_hours_calculation(sample_cleaned_df):
    """Test unresolved_age_hours equals ticket_age_hours for Open/Escalated tickets."""
    ref_time = pd.Timestamp("2024-03-30 18:00:00")
    featured_df = engineer_features(sample_cleaned_df, reference_time=ref_time)

    # Row 1 (Open): 30.0, Row 2 (Escalated): 50.0, Row 3 (Open): 0.0
    assert featured_df.loc[1, "unresolved_age_hours"] == pytest.approx(30.0)
    assert featured_df.loc[2, "unresolved_age_hours"] == pytest.approx(50.0)
    assert featured_df.loc[3, "unresolved_age_hours"] == pytest.approx(0.0)


# 8. Resolved tickets have NULL unresolved age
def test_resolved_tickets_have_null_unresolved_age(sample_cleaned_df):
    """Test that resolved tickets have NaN/NULL for unresolved_age_hours."""
    featured_df = engineer_features(sample_cleaned_df)
    assert pd.isna(featured_df.loc[0, "unresolved_age_hours"])


# 9. is_unresolved_over_24h feature
def test_is_unresolved_over_24h_feature(sample_cleaned_df):
    """Test is_unresolved_over_24h returns True only when unresolved AND age > 24h."""
    ref_time = pd.Timestamp("2024-03-30 18:00:00")
    featured_df = engineer_features(sample_cleaned_df, reference_time=ref_time)

    # Row 0: Resolved (age 10h) -> False
    # Row 1: Open (age 30h) -> True
    # Row 2: Escalated (age 50h) -> True
    # Row 3: Open (age 0h) -> False
    assert featured_df["is_unresolved_over_24h"].tolist() == [False, True, True, False]


# 10. is_critical_unresolved feature
def test_is_critical_unresolved_feature(sample_cleaned_df):
    """Test is_critical_unresolved returns True only for Critical unresolved tickets."""
    featured_df = engineer_features(sample_cleaned_df)
    # Row 3 is Critical and Open -> True
    assert featured_df["is_critical_unresolved"].tolist() == [False, False, False, True]


# 11. Rating features
def test_rating_features(sample_cleaned_df):
    """Test has_customer_rating and low_customer_rating (< 3.0) behavior."""
    # Row 0 has rating 2.0 -> has_customer_rating = True, low_customer_rating = True
    # Rows 1-3 have missing rating -> has_customer_rating = False, low_customer_rating = False
    featured_df = engineer_features(sample_cleaned_df)

    assert featured_df["has_customer_rating"].tolist() == [True, False, False, False]
    assert featured_df["low_customer_rating"].tolist() == [True, False, False, False]

    # Test edge rating = 3.0 (not low) and 5.0 (not low)
    df_ratings = sample_cleaned_df.copy()
    df_ratings["customer_rating"] = [3.0, 4.0, 5.0, 1.0]
    res_df = engineer_features(df_ratings)
    assert res_df["low_customer_rating"].tolist() == [False, False, False, True]


# 12. Response-time bucket edge cases (<1h, 1-4h, 4-12h, 12-24h, >24h)
def test_response_time_buckets_edge_cases():
    """Test response-time bucket categorization across exact boundary values: 0h, 0.99h, 1h, 4h, 12h, 24h, 30h."""
    df = pd.DataFrame(
        {
            "ticket_id": [f"TKT-{i}" for i in range(7)],
            "created_at": pd.date_range("2024-01-01", periods=7, freq="D"),
            "category": ["General"] * 7,
            "priority": ["Low"] * 7,
            "status": ["Resolved"] * 7,
            "response_time_hrs": [0.0, 0.99, 1.0, 4.0, 12.0, 24.0, 30.0],
            "resolution_time_hrs": [5.0] * 7,
            "agent_id": ["AGT-01"] * 7,
            "customer_rating": [4.0] * 7,
            "issue_summary": ["Summary"] * 7,
        }
    )
    featured = engineer_features(df)
    expected_buckets = ["<1h", "<1h", "1-4h", "4-12h", "12-24h", ">24h", ">24h"]
    assert featured["response_time_bucket"].tolist() == expected_buckets


# 13. Resolution-time bucket edge cases (<4h, 4-12h, 12-24h, 24-48h, >48h) and missing values
def test_resolution_time_buckets_edge_cases():
    """Test resolution-time bucket categorization across exact boundary values: 0h, 3.9h, 4h, 12h, 24h, 48h, 72h, and NaN."""
    df = pd.DataFrame(
        {
            "ticket_id": [f"TKT-{i}" for i in range(8)],
            "created_at": pd.date_range("2024-01-01", periods=8, freq="D"),
            "category": ["General"] * 8,
            "priority": ["Low"] * 8,
            "status": ["Resolved"] * 7 + ["Open"],
            "response_time_hrs": [1.0] * 8,
            "resolution_time_hrs": [0.0, 3.9, 4.0, 12.0, 24.0, 48.0, 72.0, np.nan],
            "agent_id": ["AGT-01"] * 8,
            "customer_rating": [4.0] * 7 + [np.nan],
            "issue_summary": ["Summary"] * 8,
        }
    )
    featured = engineer_features(df)
    expected_buckets = ["<4h", "<4h", "4-12h", "12-24h", "24-48h", ">48h", ">48h", pd.NA]
    assert featured["resolution_time_bucket"].tolist()[:7] == expected_buckets[:7]
    assert pd.isna(featured.loc[7, "resolution_time_bucket"])

    # Test long_resolution (> 48h)
    expected_long = [False, False, False, False, False, False, True, False]
    assert featured["long_resolution"].tolist() == expected_long


# 14. Priority category interaction feature
def test_priority_category_interaction(sample_cleaned_df):
    """Test priority_category interaction string formatting."""
    featured_df = engineer_features(sample_cleaned_df)
    expected = ["Low_General", "Medium_Billing", "High_Technical", "Critical_Billing"]
    assert featured_df["priority_category"].tolist() == expected


# 15. Deterministic reference timestamp derivation
def test_deterministic_reference_timestamp():
    """Test that reference timestamp defaults to max(created_at)."""
    df = pd.DataFrame(
        {
            "ticket_id": ["TKT-1", "TKT-2"],
            "created_at": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-05 10:00:00"]),
            "category": ["General", "General"],
            "priority": ["Low", "Low"],
            "status": ["Open", "Open"],
            "response_time_hrs": [1.0, 1.0],
            "resolution_time_hrs": [np.nan, np.nan],
            "agent_id": ["AGT-1", "AGT-1"],
            "customer_rating": [np.nan, np.nan],
            "issue_summary": ["Summary", "Summary"],
        }
    )
    # Default ref time is 2024-01-05 10:00:00
    featured = engineer_features(df)
    # TKT-1 is 4 days = 96 hours older than TKT-2
    assert featured.loc[0, "ticket_age_hours"] == pytest.approx(96.0)
    assert featured.loc[1, "ticket_age_hours"] == pytest.approx(0.0)

    # Custom reference timestamp
    custom_ref = pd.Timestamp("2024-01-06 10:00:00")
    featured_custom = engineer_features(df, reference_time=custom_ref)
    assert featured_custom.loc[0, "ticket_age_hours"] == pytest.approx(120.0)
    assert featured_custom.loc[1, "ticket_age_hours"] == pytest.approx(24.0)


# 16. Feature metadata completeness
def test_feature_metadata_completeness(sample_cleaned_df):
    """Test that all 16 engineered features are fully documented in FEATURE_METADATA."""
    featured_df = engineer_features(sample_cleaned_df)
    original_cols = set(sample_cleaned_df.columns)
    new_cols = set(featured_df.columns) - original_cols

    assert len(new_cols) == 16
    assert set(FEATURE_METADATA.keys()) == new_cols

    for col, meta in FEATURE_METADATA.items():
        assert "type" in meta
        assert "description" in meta
        assert len(meta["description"]) > 5


# 17. Full integration test on actual support_tickets.csv
def test_feature_engineering_actual_dataset():
    """Test complete pipeline from parse -> validate/clean -> feature engineering on actual dataset."""
    raw_df = parse_csv(DATASET_PATH)
    cleaned_df, report = validate_and_clean(raw_df)
    featured_df = engineer_features(cleaned_df)

    # Row count maintained
    assert len(featured_df) == 500
    # Original 10 columns + 16 engineered features = 26 columns
    assert len(featured_df.columns) == 26

    # Verify resolution status counts
    assert featured_df["is_resolved"].sum() == 327
    assert featured_df["is_unresolved"].sum() == 173

    # Verify priority counts
    assert featured_df["is_high_priority"].sum() == 189
    assert featured_df["is_critical"].sum() == 55
    assert (featured_df["priority_level"] >= 1).all() and (featured_df["priority_level"] <= 4).all()

    # Verify missing values in engineered features
    assert featured_df["resolution_time_bucket"].isna().sum() == 173
    assert featured_df["unresolved_age_hours"].isna().sum() == 327

    # Verify boolean SLA flags
    assert featured_df["is_unresolved_over_24h"].dtype == bool
    assert featured_df["is_critical_unresolved"].dtype == bool
    assert featured_df["has_customer_rating"].sum() == 327
    assert featured_df["has_resolution_time"].sum() == 327

"""Unit and integration tests for Stage 5: Anomaly Detection Module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.anomalies import (
    calculate_iqr_thresholds,
    detect_anomalies,
    detect_business_rule_anomalies,
    detect_statistical_anomalies,
)
from app.database import build_database, build_database_from_csv
from app.pipeline import engineer_features, parse_csv, validate_and_clean

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "support_tickets.csv"


@pytest.fixture(scope="module")
def shared_db_path(tmp_path_factory) -> Path:
    """Fixture initializing a SQLite database from the actual dataset for anomaly tests."""
    temp_dir = tmp_path_factory.mktemp("anomaly_db")
    db_path = temp_dir / "anomaly_tickets.db"
    build_database_from_csv(DATASET_PATH, db_path=db_path)
    return db_path


# =============================================================================
# 1. Statistical IQR Outlier Detection Tests
# =============================================================================


def test_calculate_iqr_thresholds_normal_values():
    """Test standard IQR calculation on numeric sequence."""
    values = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
    res = calculate_iqr_thresholds(values)

    assert res["has_sufficient_data"] is True
    assert res["q1"] == pytest.approx(13.5, abs=0.1)
    assert res["q3"] == pytest.approx(20.5, abs=0.1)
    assert res["iqr"] == pytest.approx(7.0, abs=0.1)
    assert res["upper_bound"] == pytest.approx(20.5 + 1.5 * 7.0, abs=0.1)


def test_calculate_iqr_thresholds_edge_cases():
    """Test IQR edge cases: insufficient observations, empty values, all-equal values, and NaN handling."""
    # 1. Empty / fewer than 4 items
    assert calculate_iqr_thresholds([])["has_sufficient_data"] is False
    assert calculate_iqr_thresholds([10.0, 20.0, 30.0])["has_sufficient_data"] is False

    # 2. Sequence with NaNs
    res_nan = calculate_iqr_thresholds([10.0, 20.0, np.nan, 30.0, 40.0])
    assert res_nan["has_sufficient_data"] is True
    assert res_nan["sample_size"] == 4

    # 3. All equal values (IQR = 0)
    res_equal = calculate_iqr_thresholds([5.0, 5.0, 5.0, 5.0])
    assert res_equal["has_sufficient_data"] is True
    assert res_equal["iqr"] == 0.0
    assert res_equal["upper_bound"] == 5.0


def test_detect_statistical_anomalies_actual_dataset(shared_db_path: Path):
    """Test statistical anomaly detection on actual dataset."""
    thresholds, anomalies = detect_statistical_anomalies(shared_db_path)

    assert thresholds["has_sufficient_data"] is True
    assert thresholds["sample_size"] == 327
    assert thresholds["q1"] == pytest.approx(6.15, abs=0.01)
    assert thresholds["q3"] == pytest.approx(22.95, abs=0.01)
    assert thresholds["iqr"] == pytest.approx(16.80, abs=0.01)
    assert thresholds["upper_bound"] == pytest.approx(48.15, abs=0.01)

    assert len(anomalies) == 21
    for a in anomalies:
        assert a["anomaly_type"] == "STATISTICAL_LONG_RESOLUTION"
        assert a["severity"] == "medium"
        assert a["evidence"]["actual_resolution_time_hrs"] > thresholds["upper_bound"]


# =============================================================================
# 2. Business-Rule Anomaly Detection Tests
# =============================================================================


def test_detect_business_rule_anomalies_actual_dataset(shared_db_path: Path):
    """Test business rule anomaly detectors on actual dataset."""
    anomalies = detect_business_rule_anomalies(shared_db_path)

    type_counts: dict[str, int] = {}
    for a in anomalies:
        t = a["anomaly_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    assert type_counts["HIGH_PRIORITY_UNRESOLVED_OVER_24H"] == 80
    assert type_counts["CRITICAL_UNRESOLVED"] == 31
    assert type_counts["LONG_RESOLUTION_TIME"] == 21
    assert len(anomalies) == 132


def test_business_rules_logic_on_synthetic_data(tmp_path: Path):
    """Test edge cases for business rules using controlled synthetic records."""
    ref_time = pd.Timestamp("2024-03-30 18:00:00")
    synthetic_df = pd.DataFrame(
        {
            "ticket_id": ["T-01", "T-02", "T-03", "T-04", "T-05"],
            "created_at": pd.to_datetime(
                [
                    ref_time - pd.Timedelta(hours=10),  # T-01: Low, Open, age 10h
                    ref_time - pd.Timedelta(hours=30),  # T-02: High, Open, age 30h (>24h)
                    ref_time - pd.Timedelta(hours=5),   # T-03: Critical, Open, age 5h (<=24h)
                    ref_time - pd.Timedelta(hours=50),  # T-04: Resolved, age 50h, res_time 10h
                    ref_time - pd.Timedelta(hours=100), # T-05: Resolved, age 100h, res_time 60h (>48h)
                ]
            ),
            "category": ["General", "Billing", "Technical", "General", "Technical"],
            "priority": ["Low", "High", "Critical", "High", "Low"],
            "status": ["Open", "Open", "Open", "Resolved", "Resolved"],
            "response_time_hrs": [1.0, 1.0, 1.0, 1.0, 1.0],
            "resolution_time_hrs": [np.nan, np.nan, np.nan, 10.0, 60.0],
            "agent_id": ["AGT-01", "AGT-01", "AGT-01", "AGT-01", "AGT-01"],
            "customer_rating": [np.nan, np.nan, np.nan, 4.0, 2.0],
            "issue_summary": ["Issue 1", "Issue 2", "Issue 3", "Issue 4", "Issue 5"],
        }
    )
    cleaned, _ = validate_and_clean(synthetic_df)
    featured = engineer_features(cleaned, reference_time=ref_time)

    db_path = tmp_path / "synthetic_anomalies.db"
    build_database(featured, db_path=db_path)

    anomalies = detect_business_rule_anomalies(db_path)
    flagged_map: dict[str, list[str]] = {}
    for a in anomalies:
        flagged_map.setdefault(a["ticket_id"], []).append(a["anomaly_type"])

    # T-01: Low, Open, age 10h -> No anomaly
    assert "T-01" not in flagged_map

    # T-02: High, Open, age 30h -> HIGH_PRIORITY_UNRESOLVED_OVER_24H
    assert flagged_map["T-02"] == ["HIGH_PRIORITY_UNRESOLVED_OVER_24H"]

    # T-03: Critical, Open, age 5h -> CRITICAL_UNRESOLVED (age <=24h does NOT trigger Rule 1)
    assert flagged_map["T-03"] == ["CRITICAL_UNRESOLVED"]

    # T-04: High, Resolved, age 50h, res 10h -> No anomaly (resolved tickets not flagged by unresolved rules)
    assert "T-04" not in flagged_map

    # T-05: Resolved, res 60h -> LONG_RESOLUTION_TIME
    assert flagged_map["T-05"] == ["LONG_RESOLUTION_TIME"]


def test_multi_rule_preservation(shared_db_path: Path):
    """Verify that a ticket satisfying multiple rules retains all applicable anomaly records."""
    # In actual dataset, TKT-361 is Critical, Open, and unresolved_age_hours > 24h
    res = detect_anomalies(shared_db_path)
    anomalies = res["anomalies"]

    crit_tickets = [a for a in anomalies if a["ticket_id"] == "TKT-361"]
    crit_types = {a["anomaly_type"] for a in crit_tickets}

    assert "CRITICAL_UNRESOLVED" in crit_types
    assert "HIGH_PRIORITY_UNRESOLVED_OVER_24H" in crit_types
    assert len(crit_tickets) == 2


# =============================================================================
# 3. High-Level Coordinator Tests
# =============================================================================


def test_detect_anomalies_top_level_coordinator(shared_db_path: Path):
    """Test top-level detect_anomalies function output format and summary metrics."""
    res = detect_anomalies(shared_db_path)

    summary = res["summary"]
    assert summary["total_anomalies"] == 153
    assert summary["critical"] == 31
    assert summary["high"] == 80
    assert summary["medium"] == 42  # 21 Rule 3 + 21 Statistical
    assert summary["unique_tickets_flagged"] == 101

    assert "statistical_thresholds" in res
    assert res["statistical_thresholds"]["upper_bound"] == pytest.approx(48.15, abs=0.01)

    assert len(res["anomalies"]) == 153
    for a in res["anomalies"]:
        assert "ticket_id" in a
        assert "anomaly_type" in a
        assert "severity" in a
        assert "reason" in a
        assert "evidence" in a

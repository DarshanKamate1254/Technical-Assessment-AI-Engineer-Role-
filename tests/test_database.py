"""Unit and integration tests for Stage 4: Store & Query Layer (SQLite)."""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest

from app.database.connection import get_connection, get_db_connection
from app.database.loader import (
    build_database,
    build_database_from_csv,
    prepare_records_for_sqlite,
    store_tickets,
)
from app.database.repository import (
    ALLOWED_COLUMNS,
    ALLOWED_OPERATORS,
    ALLOWED_SORT_COLUMNS,
    TicketRepository,
    get_average_customer_rating_by_agent,
    get_average_resolution_time,
    get_average_resolution_time_by_priority,
    get_average_response_time,
    get_critical_unresolved_count,
    get_high_priority_unresolved_over_24h,
    get_lowest_average_rating_agent,
    get_ticket_by_id,
    get_ticket_count,
    get_ticket_counts_by_category,
    get_ticket_counts_by_priority,
    get_ticket_counts_by_status,
    get_tickets_by_agent,
    get_tickets_by_priority,
    get_tickets_by_status,
    get_unresolved_over_24h_count,
    get_unresolved_ticket_count,
    query_tickets,
)
from app.database.schema import ALL_COLUMNS, TABLE_NAME, initialize_database
from app.pipeline import engineer_features, parse_csv, validate_and_clean

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "support_tickets.csv"


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Fixture returning a path to a temporary SQLite database."""
    return tmp_path / "test_tickets.db"


@pytest.fixture
def populated_db_path(temp_db_path: Path) -> Path:
    """Fixture that initializes and populates a temporary database with actual dataset."""
    raw_df = parse_csv(DATASET_PATH)
    cleaned_df, _ = validate_and_clean(raw_df)
    featured_df = engineer_features(cleaned_df)
    build_database(featured_df, db_path=temp_db_path)
    return temp_db_path


@pytest.fixture
def sample_featured_df() -> pd.DataFrame:
    """Fixture returning a 4-row feature-engineered DataFrame for deterministic testing."""
    ref_time = pd.Timestamp("2024-03-30 18:00:00")
    raw_df = pd.DataFrame(
        {
            "ticket_id": pd.Series(["TKT-001", "TKT-002", "TKT-003", "TKT-004"], dtype="string"),
            "created_at": pd.to_datetime(
                [
                    ref_time - pd.Timedelta(hours=10),
                    ref_time - pd.Timedelta(hours=30),
                    ref_time - pd.Timedelta(hours=50),
                    ref_time,
                ]
            ),
            "category": pd.Series(["General", "Billing", "Technical", "Billing"], dtype="string"),
            "priority": pd.Series(["Low", "Medium", "High", "Critical"], dtype="string"),
            "status": pd.Series(["Resolved", "Open", "Escalated", "Open"], dtype="string"),
            "response_time_hrs": pd.Series([0.5, 2.0, 8.0, 26.0], dtype="float64"),
            "resolution_time_hrs": pd.Series([10.0, np.nan, np.nan, np.nan], dtype="float64"),
            "agent_id": pd.Series(["AGT-01", "AGT-02", "AGT-03", "AGT-02"], dtype="string"),
            "customer_rating": pd.Series([2.0, np.nan, np.nan, np.nan], dtype="float64"),
            "issue_summary": pd.Series(
                ["General question", "Billing dispute", "API timeout", "Critical payment outage"],
                dtype="string",
            ),
        }
    )
    cleaned_df, _ = validate_and_clean(raw_df)
    return engineer_features(cleaned_df, reference_time=ref_time)


# =============================================================================
# 1. Database Schema & Persistence Tests
# =============================================================================


def test_database_initialization(temp_db_path: Path):
    """Test idempotent database initialization and index creation."""
    initialize_database(temp_db_path)
    assert temp_db_path.exists()

    with get_connection(temp_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({TABLE_NAME});")
        columns = [row["name"] for row in cursor.fetchall()]
        assert len(columns) == 26
        assert columns == list(ALL_COLUMNS)

        cursor.execute(f"PRAGMA index_list({TABLE_NAME});")
        indexes = [row["name"] for row in cursor.fetchall()]
        assert "idx_tickets_status" in indexes
        assert "idx_tickets_priority" in indexes
        assert "idx_tickets_agent_id" in indexes

    # Idempotent call
    initialize_database(temp_db_path)


def test_store_tickets_preserves_nulls(temp_db_path: Path, sample_featured_df: pd.DataFrame):
    """Test storing tickets preserves NULL values in SQLite."""
    count = store_tickets(sample_featured_df, db_path=temp_db_path)
    assert count == 4

    with get_connection(temp_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT ticket_id, resolution_time_hrs, customer_rating, unresolved_age_hours "
            f"FROM {TABLE_NAME} WHERE ticket_id = 'TKT-002';"
        )
        row = cursor.fetchone()
        assert row["ticket_id"] == "TKT-002"
        assert row["resolution_time_hrs"] is None
        assert row["customer_rating"] is None
        assert row["unresolved_age_hours"] == 30.0


def test_database_rebuilding(temp_db_path: Path, sample_featured_df: pd.DataFrame):
    """Test that build_database replaces existing data cleanly."""
    build_database(sample_featured_df, db_path=temp_db_path)
    assert TicketRepository(temp_db_path).get_ticket_count() == 4

    # Rebuild with 2-row subset
    build_database(sample_featured_df.iloc[:2], db_path=temp_db_path)
    assert TicketRepository(temp_db_path).get_ticket_count() == 2


# =============================================================================
# 2. Required Core Business Query Tests
# =============================================================================


def test_query_1_critical_unresolved_count(populated_db_path: Path):
    """Test Query 1: How many critical tickets are unresolved?"""
    repo = TicketRepository(populated_db_path)
    count = repo.get_critical_unresolved_count()
    assert count == 31


def test_query_2_lowest_average_rating_agent(populated_db_path: Path):
    """Test Query 2: Which agent has the lowest average customer rating?"""
    repo = TicketRepository(populated_db_path)
    result = repo.get_lowest_average_rating_agent()

    assert result is not None
    assert result["agent_id"] == "AGT-08"
    assert result["average_rating"] == pytest.approx(3.48, abs=0.01)
    assert result["rated_ticket_count"] == 25


def test_query_3_unresolved_over_24h_count(populated_db_path: Path):
    """Test Query 3: How many unresolved tickets are older than 24 hours?"""
    repo = TicketRepository(populated_db_path)
    count = repo.get_unresolved_over_24h_count()
    assert count == 170


def test_query_4_high_priority_unresolved_over_24h(populated_db_path: Path):
    """Test Query 4: Which high-priority unresolved tickets are older than 24 hours?"""
    repo = TicketRepository(populated_db_path)
    tickets = repo.get_high_priority_unresolved_over_24h(limit=100)

    assert len(tickets) == 80
    for t in tickets:
        assert t["priority"] in ["High", "Critical"]
        assert t["status"] in ["Open", "Escalated"]
        assert t["ticket_age_hours"] > 24.0
        assert "ticket_id" in t
        assert "issue_summary" in t


def test_query_5_average_resolution_time_by_priority(populated_db_path: Path):
    """Test Query 5: What is the average resolution time by priority?"""
    repo = TicketRepository(populated_db_path)
    results = repo.get_average_resolution_time_by_priority()

    assert len(results) == 4
    priority_map = {r["priority"]: r for r in results}

    # Low priority takes longest, Critical resolved fastest
    assert priority_map["Low"]["average_resolution_time_hrs"] == pytest.approx(28.47, abs=0.1)
    assert priority_map["Low"]["resolved_ticket_count"] == 101
    assert priority_map["Medium"]["average_resolution_time_hrs"] == pytest.approx(16.94, abs=0.1)
    assert priority_map["Medium"]["resolved_ticket_count"] == 117
    assert priority_map["High"]["average_resolution_time_hrs"] == pytest.approx(13.57, abs=0.1)
    assert priority_map["High"]["resolved_ticket_count"] == 85
    assert priority_map["Critical"]["average_resolution_time_hrs"] == pytest.approx(10.63, abs=0.1)
    assert priority_map["Critical"]["resolved_ticket_count"] == 24


def test_query_6_ticket_counts_by_category(populated_db_path: Path):
    """Test Query 6: How many tickets by category?"""
    repo = TicketRepository(populated_db_path)
    results = repo.get_ticket_counts_by_category()

    category_map = {r["category"]: r["count"] for r in results}
    assert category_map["General"] == 189
    assert category_map["Billing"] == 159
    assert category_map["Technical"] == 152


def test_query_7_ticket_counts_by_status(populated_db_path: Path):
    """Test Query 7: How many tickets by status?"""
    repo = TicketRepository(populated_db_path)
    results = repo.get_ticket_counts_by_status()

    status_map = {r["status"]: r["count"] for r in results}
    assert status_map["Resolved"] == 327
    assert status_map["Open"] == 111
    assert status_map["Escalated"] == 62


def test_query_8_average_customer_rating_by_agent(populated_db_path: Path):
    """Test Query 8: What is the average customer rating by agent?"""
    repo = TicketRepository(populated_db_path)
    results = repo.get_average_customer_rating_by_agent()

    assert len(results) == 12
    # Verify lowest is AGT-08
    assert results[0]["agent_id"] == "AGT-08"
    assert results[0]["average_rating"] == pytest.approx(3.48, abs=0.01)
    assert results[0]["rated_ticket_count"] == 25

    # Total rated tickets sum must equal 327
    total_rated = sum(r["rated_ticket_count"] for r in results)
    assert total_rated == 327


# =============================================================================
# 3. General Query Helpers Tests
# =============================================================================


def test_general_query_helpers(populated_db_path: Path):
    """Test additional general repository helper methods."""
    repo = TicketRepository(populated_db_path)

    assert repo.get_ticket_count() == 500
    assert repo.get_unresolved_ticket_count() == 173

    avg_res = repo.get_average_resolution_time()
    assert avg_res is not None and avg_res > 0

    avg_resp = repo.get_average_response_time()
    assert avg_resp is not None and avg_resp > 0

    # Specific queries
    agent_tickets = repo.get_tickets_by_agent("AGT-01", limit=10)
    assert len(agent_tickets) > 0
    assert all(t["agent_id"] == "AGT-01" for t in agent_tickets)

    tkt = repo.get_ticket_by_id("TKT-001")
    assert tkt is not None
    assert tkt["ticket_id"] == "TKT-001"

    non_existent = repo.get_ticket_by_id("NON_EXISTENT_ID")
    assert non_existent is None


# =============================================================================
# 4. Generic Filtering & Security Tests
# =============================================================================


def test_query_tickets_filtering_and_projection(populated_db_path: Path):
    """Test query_tickets with various filters, column projection, and ordering."""
    repo = TicketRepository(populated_db_path)

    # 1. Equality filter + projection
    results = repo.query_tickets(
        filters={"priority": "Critical", "status": "Open"},
        columns=["ticket_id", "priority", "status", "category"],
        limit=10,
    )
    assert len(results) > 0
    assert len(results) <= 10
    for r in results:
        assert set(r.keys()) == {"ticket_id", "priority", "status", "category"}
        assert r["priority"] == "Critical"
        assert r["status"] == "Open"

    # 2. IN operator
    results_in = repo.query_tickets(
        filters={"priority__in": ["High", "Critical"]},
        limit=200,
    )
    assert len(results_in) == 189
    assert all(r["priority"] in ["High", "Critical"] for r in results_in)

    # 3. Numeric comparison operator (__gt) and sorting
    results_gt = repo.query_tickets(
        filters={"ticket_age_hours__gt": 1000.0},
        order_by="ticket_age_hours",
        descending=True,
        limit=50,
    )
    assert len(results_gt) > 0
    assert all(r["ticket_age_hours"] > 1000.0 for r in results_gt)
    # Check descending order
    ages = [r["ticket_age_hours"] for r in results_gt]
    assert ages == sorted(ages, reverse=True)

    # 4. IS NULL operator (__isnull)
    results_null = repo.query_tickets(
        filters={"resolution_time_hrs__isnull": True},
        limit=200,
    )
    assert len(results_null) == 173
    assert all(r["resolution_time_hrs"] is None for r in results_null)


def test_query_tickets_security_whitelisting(populated_db_path: Path):
    """Test that query_tickets rejects unwhitelisted columns, sort keys, and operators to prevent SQL injection."""
    repo = TicketRepository(populated_db_path)

    # 1. Disallowed column projection
    with pytest.raises(ValueError) as exc_info:
        repo.query_tickets(columns=["ticket_id", "secret_table_drop;--"])
    assert "Disallowed column" in str(exc_info.value)

    # 2. Disallowed filter column
    with pytest.raises(ValueError) as exc_info:
        repo.query_tickets(filters={"malicious_col": "value"})
    assert "Disallowed filter column" in str(exc_info.value)

    # 3. Disallowed operator
    with pytest.raises(ValueError) as exc_info:
        repo.query_tickets(filters={"status__malicious_op": "Open"})
    assert "Disallowed filter operator" in str(exc_info.value)

    # 4. Disallowed sort column
    with pytest.raises(ValueError) as exc_info:
        repo.query_tickets(order_by="1; DROP TABLE support_tickets;--")
    assert "Disallowed sort column" in str(exc_info.value)


def test_query_tickets_sql_injection_value_parameterization(populated_db_path: Path):
    """Test that SQL injection strings inside filter values are safely treated as literal strings."""
    repo = TicketRepository(populated_db_path)

    # Malicious injection attempt inside value
    injection_value = "' OR '1'='1"
    results = repo.query_tickets(filters={"status": injection_value})
    # Must return empty list safely without executing injection
    assert results == []


# =============================================================================
# 5. Data Integrity Check (Stage 3 DataFrame vs SQLite)
# =============================================================================


def test_data_integrity_dataframe_vs_sqlite(populated_db_path: Path):
    """Test complete 1:1 data integrity between Stage 3 DataFrame and SQLite database."""
    raw_df = parse_csv(DATASET_PATH)
    cleaned_df, _ = validate_and_clean(raw_df)
    featured_df = engineer_features(cleaned_df)

    repo = TicketRepository(populated_db_path)

    # 1. Total row count matches
    assert len(featured_df) == repo.get_ticket_count() == 500

    # 2. Distinct ticket IDs match
    with get_connection(populated_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(DISTINCT ticket_id) as unique_ids FROM {TABLE_NAME};")
        db_unique_ids = cursor.fetchone()["unique_ids"]
        assert db_unique_ids == featured_df["ticket_id"].nunique() == 500

    # 3. NULL counts match
    with get_connection(populated_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) as null_res FROM {TABLE_NAME} WHERE resolution_time_hrs IS NULL;")
        assert cursor.fetchone()["null_res"] == featured_df["resolution_time_hrs"].isna().sum() == 173

        cursor.execute(f"SELECT COUNT(*) as null_ratings FROM {TABLE_NAME} WHERE customer_rating IS NULL;")
        assert cursor.fetchone()["null_ratings"] == featured_df["customer_rating"].isna().sum() == 173

        cursor.execute(f"SELECT COUNT(*) as null_unres_age FROM {TABLE_NAME} WHERE unresolved_age_hours IS NULL;")
        assert cursor.fetchone()["null_unres_age"] == featured_df["unresolved_age_hours"].isna().sum() == 327


def test_build_database_from_csv_pipeline(temp_db_path: Path):
    """Test build_database_from_csv end-to-end execution."""
    df, report, rows_stored = build_database_from_csv(DATASET_PATH, db_path=temp_db_path)
    assert len(df) == 500
    assert rows_stored == 500
    assert report["rows_before"] == 500
    assert report["rows_after"] == 500
    assert temp_db_path.exists()

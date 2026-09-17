"""Unit and integration tests for Stage 5: Analytics Module."""

from pathlib import Path
import pytest

from app.analytics import (
    get_agent_performance,
    get_average_customer_rating,
    get_average_rating_by_agent,
    get_average_rating_by_category,
    get_average_rating_by_priority,
    get_average_resolution_time,
    get_average_response_time,
    get_critical_unresolved_count,
    get_critical_unresolved_over_24h,
    get_high_priority_unresolved_count,
    get_high_priority_unresolved_over_24h,
    get_median_resolution_time,
    get_median_response_time,
    get_rating_analysis,
    get_rating_distribution,
    get_resolution_time_analysis,
    get_resolution_time_by_agent,
    get_resolution_time_by_category,
    get_resolution_time_by_priority,
    get_response_time_by_agent,
    get_response_time_by_category,
    get_response_time_by_priority,
    get_ticket_count_by_agent,
    get_ticket_count_by_category,
    get_ticket_count_by_priority,
    get_ticket_count_by_status,
    get_ticket_summary,
    get_total_tickets,
    get_unresolved_by_agent,
    get_unresolved_by_category,
    get_unresolved_by_priority,
    get_unresolved_over_24h,
    get_unresolved_ticket_analysis,
    get_unresolved_ticket_count,
)
from app.database import build_database_from_csv

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "support_tickets.csv"


@pytest.fixture(scope="module")
def shared_db_path(tmp_path_factory) -> Path:
    """Fixture initializing a SQLite database from the actual dataset for analytics tests."""
    temp_dir = tmp_path_factory.mktemp("analytics_db")
    db_path = temp_dir / "analytics_tickets.db"
    build_database_from_csv(DATASET_PATH, db_path=db_path)
    return db_path


# =============================================================================
# 1. Volume & Unresolved Metrics Tests
# =============================================================================


def test_volume_metrics(shared_db_path: Path):
    """Test total ticket volume and categorical breakdowns."""
    assert get_total_tickets(shared_db_path) == 500

    status_counts = {r["status"]: r["count"] for r in get_ticket_count_by_status(shared_db_path)}
    assert status_counts == {"Resolved": 327, "Open": 111, "Escalated": 62}

    priority_counts = {r["priority"]: r["count"] for r in get_ticket_count_by_priority(shared_db_path)}
    assert priority_counts == {"Medium": 169, "Low": 142, "High": 134, "Critical": 55}

    category_counts = {r["category"]: r["count"] for r in get_ticket_count_by_category(shared_db_path)}
    assert category_counts == {"General": 189, "Billing": 159, "Technical": 152}

    agent_counts = get_ticket_count_by_agent(shared_db_path)
    assert len(agent_counts) == 12
    assert sum(r["count"] for r in agent_counts) == 500


def test_unresolved_analytics(shared_db_path: Path):
    """Test unresolved ticket backlog analytics."""
    assert get_unresolved_ticket_count(shared_db_path) == 173
    assert get_critical_unresolved_count(shared_db_path) == 31
    assert get_high_priority_unresolved_count(shared_db_path) == 80  # 49 High + 31 Critical

    unres_prio = {r["priority"]: r["count"] for r in get_unresolved_by_priority(shared_db_path)}
    assert unres_prio["Critical"] == 31
    assert unres_prio["High"] == 49
    assert sum(unres_prio.values()) == 173

    unres_cat = {r["category"]: r["count"] for r in get_unresolved_by_category(shared_db_path)}
    assert sum(unres_cat.values()) == 173

    unres_agent = get_unresolved_by_agent(shared_db_path)
    assert len(unres_agent) == 12
    assert sum(r["count"] for r in unres_agent) == 173


# =============================================================================
# 2. Response-Time Analytics Tests
# =============================================================================


def test_response_time_analytics(shared_db_path: Path):
    """Test average and median response times and grouping dimensions."""
    avg_resp = get_average_response_time(shared_db_path)
    assert avg_resp["average_response_time_hrs"] == pytest.approx(2.62, abs=0.01)
    assert avg_resp["observed_count"] == 500

    med_resp = get_median_response_time(shared_db_path)
    assert med_resp["median_response_time_hrs"] == pytest.approx(2.60, abs=0.05)
    assert med_resp["observed_count"] == 500

    by_prio = get_response_time_by_priority(shared_db_path)
    assert len(by_prio) == 4

    by_cat = get_response_time_by_category(shared_db_path)
    assert len(by_cat) == 3

    by_agent = get_response_time_by_agent(shared_db_path)
    assert len(by_agent) == 12


# =============================================================================
# 3. Resolution-Time Analytics Tests
# =============================================================================


def test_resolution_time_analytics(shared_db_path: Path):
    """Test average and median resolution times, sample counts, and groupings."""
    avg_res = get_average_resolution_time(shared_db_path)
    assert avg_res["average_resolution_time_hrs"] == pytest.approx(19.16, abs=0.05)
    assert avg_res["resolved_ticket_count"] == 327

    med_res = get_median_resolution_time(shared_db_path)
    assert med_res["median_resolution_time_hrs"] == pytest.approx(12.00, abs=0.1)
    assert med_res["resolved_ticket_count"] == 327

    by_prio = {r["priority"]: r for r in get_resolution_time_by_priority(shared_db_path)}
    assert by_prio["Critical"]["average_resolution_time_hrs"] == pytest.approx(10.63, abs=0.1)
    assert by_prio["Low"]["average_resolution_time_hrs"] == pytest.approx(28.47, abs=0.1)

    by_cat = get_resolution_time_by_category(shared_db_path)
    assert len(by_cat) == 3

    by_agent = get_resolution_time_by_agent(shared_db_path)
    assert len(by_agent) == 12


# =============================================================================
# 4. Customer-Rating Analytics Tests
# =============================================================================


def test_customer_rating_analytics(shared_db_path: Path):
    """Test average ratings, star distributions, and agent/priority/category groupings."""
    avg_rating = get_average_customer_rating(shared_db_path)
    assert avg_rating["average_customer_rating"] == pytest.approx(3.7492, abs=0.01)
    assert avg_rating["rated_ticket_count"] == 327

    dist = {r["rating"]: r["count"] for r in get_rating_distribution(shared_db_path)}
    assert set(dist.keys()) == {1, 2, 3, 4, 5}
    assert sum(dist.values()) == 327

    by_agent = get_average_rating_by_agent(shared_db_path)
    assert len(by_agent) == 12
    assert by_agent[0]["agent_id"] == "AGT-08"

    by_prio = get_average_rating_by_priority(shared_db_path)
    assert len(by_prio) == 4

    by_cat = get_average_rating_by_category(shared_db_path)
    assert len(by_cat) == 3


# =============================================================================
# 5. Agent Performance Analytics Tests
# =============================================================================


def test_agent_performance_metrics(shared_db_path: Path):
    """Test structured agent performance table calculations."""
    agent_perf = get_agent_performance(shared_db_path)
    assert len(agent_perf) == 12

    total_tickets_sum = sum(a["total_tickets"] for a in agent_perf)
    resolved_sum = sum(a["resolved_tickets"] for a in agent_perf)
    unresolved_sum = sum(a["unresolved_tickets"] for a in agent_perf)
    rated_sum = sum(a["rated_ticket_count"] for a in agent_perf)

    assert total_tickets_sum == 500
    assert resolved_sum == 327
    assert unresolved_sum == 173
    assert rated_sum == 327

    for a in agent_perf:
        assert a["total_tickets"] == a["resolved_tickets"] + a["unresolved_tickets"]
        assert a["average_response_time"] is not None
        assert a["average_resolution_time"] is not None
        assert a["average_customer_rating"] is not None


# =============================================================================
# 6. SLA & Age Analytics Tests
# =============================================================================


def test_sla_age_analytics(shared_db_path: Path):
    """Test SLA age queries."""
    unres_24h = get_unresolved_over_24h(limit=200, db_path=shared_db_path)
    assert len(unres_24h) == 170

    high_unres_24h = get_high_priority_unresolved_over_24h(limit=200, db_path=shared_db_path)
    assert len(high_unres_24h) == 80

    crit_unres_24h = get_critical_unresolved_over_24h(limit=200, db_path=shared_db_path)
    assert len(crit_unres_24h) == 31


# =============================================================================
# 7. Composite Aggregations Tests
# =============================================================================


def test_composite_aggregations(shared_db_path: Path):
    """Test high-level composite summaries for LangChain tool readiness."""
    summary = get_ticket_summary(shared_db_path)
    assert summary["total_tickets"] == 500
    assert len(summary["by_status"]) == 3
    assert len(summary["by_priority"]) == 4

    unres_analysis = get_unresolved_ticket_analysis(shared_db_path)
    assert unres_analysis["total_unresolved"] == 173
    assert unres_analysis["critical_unresolved"] == 31
    assert unres_analysis["high_priority_unresolved"] == 80

    res_analysis = get_resolution_time_analysis(shared_db_path)
    assert res_analysis["overall_average_resolution_time_hrs"] == pytest.approx(19.16, abs=0.05)
    assert res_analysis["overall_median_resolution_time_hrs"] == pytest.approx(12.00, abs=0.1)

    rating_analysis = get_rating_analysis(shared_db_path)
    assert rating_analysis["average_customer_rating"] == pytest.approx(3.7492, abs=0.01)
    assert rating_analysis["lowest_average_rating_agent"]["agent_id"] == "AGT-08"

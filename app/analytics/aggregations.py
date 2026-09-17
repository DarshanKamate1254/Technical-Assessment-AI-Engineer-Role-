"""Composite Aggregations Module for Customer Support Analytics.

Provides unified, structured summary objects tailored for future LLM / LangChain
tool execution, synthesizing volume, backlog, resolution performance, and ratings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.analytics.metrics import (
    get_average_customer_rating,
    get_average_rating_by_category,
    get_average_rating_by_priority,
    get_average_resolution_time,
    get_average_response_time,
    get_critical_unresolved_count,
    get_high_priority_unresolved_count,
    get_median_resolution_time,
    get_median_response_time,
    get_rating_distribution,
    get_resolution_time_by_category,
    get_resolution_time_by_priority,
    get_ticket_count_by_category,
    get_ticket_count_by_priority,
    get_ticket_count_by_status,
    get_total_tickets,
    get_unresolved_by_category,
    get_unresolved_by_priority,
    get_unresolved_ticket_count,
)
from app.database.repository import (
    get_lowest_average_rating_agent,
    get_unresolved_over_24h_count,
)


def get_ticket_summary(db_path: str | Path | None = None) -> dict[str, Any]:
    """Provide a high-level operational executive summary of the ticket dataset.

    Returns:
        Structured dictionary covering total volume, status distribution, priority distribution,
        category distribution, and overall response/resolution averages.
    """
    total = get_total_tickets(db_path)
    status_counts = get_ticket_count_by_status(db_path)
    priority_counts = get_ticket_count_by_priority(db_path)
    category_counts = get_ticket_count_by_category(db_path)
    avg_resp = get_average_response_time(db_path)
    avg_res = get_average_resolution_time(db_path)
    avg_rating = get_average_customer_rating(db_path)

    return {
        "total_tickets": total,
        "by_status": status_counts,
        "by_priority": priority_counts,
        "by_category": category_counts,
        "response_time": avg_resp,
        "resolution_time": avg_res,
        "customer_rating": avg_rating,
    }


def get_unresolved_ticket_analysis(
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Provide an in-depth operational analysis of the unresolved ticket backlog.

    Returns:
        Structured breakdown of unresolved counts, critical backlog, high-priority SLA breaches,
        and distributions by priority and category.
    """
    total_unresolved = get_unresolved_ticket_count(db_path)
    critical_unresolved = get_critical_unresolved_count(db_path)
    high_priority_unresolved = get_high_priority_unresolved_count(db_path)
    unresolved_over_24h = get_unresolved_over_24h_count(db_path)
    by_priority = get_unresolved_by_priority(db_path)
    by_category = get_unresolved_by_category(db_path)

    return {
        "total_unresolved": total_unresolved,
        "critical_unresolved": critical_unresolved,
        "high_priority_unresolved": high_priority_unresolved,
        "unresolved_older_than_24h": unresolved_over_24h,
        "unresolved_by_priority": by_priority,
        "unresolved_by_category": by_category,
    }


def get_resolution_time_analysis(
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Provide a comprehensive resolution time analysis.

    Returns:
        Overall resolution averages and medians, alongside priority and category breakdowns.
    """
    avg_res = get_average_resolution_time(db_path)
    med_res = get_median_resolution_time(db_path)
    by_priority = get_resolution_time_by_priority(db_path)
    by_category = get_resolution_time_by_category(db_path)

    return {
        "overall_average_resolution_time_hrs": avg_res.get("average_resolution_time_hrs"),
        "overall_median_resolution_time_hrs": med_res.get("median_resolution_time_hrs"),
        "resolved_ticket_count": avg_res.get("resolved_ticket_count"),
        "resolution_time_by_priority": by_priority,
        "resolution_time_by_category": by_category,
    }


def get_rating_analysis(db_path: str | Path | None = None) -> dict[str, Any]:
    """Provide customer rating and satisfaction analytics.

    Returns:
        Overall rating average, star rating distribution, breakdowns by category and priority,
        and the lowest rated agent.
    """
    avg_rating = get_average_customer_rating(db_path)
    dist = get_rating_distribution(db_path)
    by_category = get_average_rating_by_category(db_path)
    by_priority = get_average_rating_by_priority(db_path)
    lowest_agent = get_lowest_average_rating_agent(db_path)

    return {
        "average_customer_rating": avg_rating.get("average_customer_rating"),
        "rated_ticket_count": avg_rating.get("rated_ticket_count"),
        "rating_distribution": dist,
        "rating_by_category": by_category,
        "rating_by_priority": by_priority,
        "lowest_average_rating_agent": lowest_agent,
    }

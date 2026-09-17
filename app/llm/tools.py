"""LangChain Tools Module for Customer Support Analytics.

Exposes deterministic Stage 4 and Stage 5 analytics, queries, and anomaly
detection functions as typed LangChain tools for agent orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.analytics.agents import get_agent_performance as _get_agent_performance
from app.analytics.aggregations import (
    get_rating_analysis as _get_rating_analysis,
    get_resolution_time_analysis as _get_resolution_time_analysis,
    get_ticket_summary as _get_ticket_summary,
    get_unresolved_ticket_analysis as _get_unresolved_ticket_analysis,
)
from app.analytics.metrics import (
    get_average_response_time as _get_average_response_time,
    get_median_response_time as _get_median_response_time,
    get_response_time_by_category as _get_response_time_by_category,
    get_response_time_by_priority as _get_response_time_by_priority,
)
from app.anomalies.detector import detect_anomalies as _detect_anomalies
from app.database.repository import (
    get_critical_unresolved_count as _get_critical_unresolved_count,
    get_high_priority_unresolved_over_24h as _get_high_priority_unresolved_over_24h,
    get_lowest_average_rating_agent as _get_lowest_average_rating_agent,
    get_unresolved_over_24h_count as _get_unresolved_over_24h_count,
)


class LimitInput(BaseModel):
    """Input schema for tools supporting a result limit."""

    limit: int = Field(
        default=50, description="Maximum number of ticket records to return (1-200)."
    )


def create_support_tools(db_path: str | Path | None = None) -> list[BaseTool]:
    """Factory function creating all support analytics LangChain tools bound to a specific database path.

    Args:
        db_path: Optional path to SQLite database.

    Returns:
        List of configured LangChain BaseTool instances.
    """

    @tool
    def get_ticket_summary() -> dict[str, Any]:
        """Return a high-level operational summary of the support ticket dataset,

        including total volume, status distribution, priority distribution,
        category distribution, and overall response/resolution averages.
        """
        return _get_ticket_summary(db_path)

    @tool
    def get_critical_unresolved_count() -> dict[str, Any]:
        """Return the exact count of Critical priority tickets that are currently

        in an unresolved state (Open or Escalated).
        """
        count = _get_critical_unresolved_count(db_path)
        return {"critical_unresolved_count": count}

    @tool
    def get_unresolved_ticket_analysis() -> dict[str, Any]:
        """Return a comprehensive analysis of all unresolved support tickets,

        including total backlog count, critical unresolved count, high-priority SLA breaches,
        unresolved counts older than 24h, and distributions by priority and category.
        """
        return _get_unresolved_ticket_analysis(db_path)

    @tool
    def get_agent_performance() -> list[dict[str, Any]]:
        """Return structured operational performance metrics for all support agents,

        including total tickets handled, resolved tickets, unresolved tickets, average response time,
        average resolution time, average customer rating, and rated ticket count.
        """
        return _get_agent_performance(db_path)

    @tool
    def get_lowest_average_rating_agent() -> dict[str, Any]:
        """Identify the support agent with the lowest average customer rating,

        excluding missing ratings. Returns agent_id, average_rating, and rated_ticket_count.
        """
        result = _get_lowest_average_rating_agent(db_path)
        if result is None:
            return {"message": "No rated agent records found."}
        return result

    @tool(args_schema=LimitInput)
    def get_high_priority_unresolved_tickets(limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve High and Critical priority unresolved tickets that are older than 24 hours.

        Returns ticket details including ticket_id, priority, status, created_at, ticket_age_hours,
        unresolved_age_hours, agent_id, and issue_summary.
        """
        return _get_high_priority_unresolved_over_24h(limit=limit, db_path=db_path)

    @tool
    def get_resolution_time_analysis() -> dict[str, Any]:
        """Return resolution time analytics for resolved tickets, including overall average

        resolution time, overall median resolution time, resolved sample count, and breakdowns
        by priority and category.
        """
        return _get_resolution_time_analysis(db_path)

    @tool
    def get_response_time_analysis() -> dict[str, Any]:
        """Return response time analytics, including overall average response time,

        overall median response time, observed sample count, and response times grouped
        by priority and category.
        """
        avg_resp = _get_average_response_time(db_path)
        med_resp = _get_median_response_time(db_path)
        by_prio = _get_response_time_by_priority(db_path)
        by_cat = _get_response_time_by_category(db_path)
        return {
            "overall_average_response_time_hrs": avg_resp.get("average_response_time_hrs"),
            "overall_median_response_time_hrs": med_resp.get("median_response_time_hrs"),
            "observed_count": avg_resp.get("observed_count"),
            "response_time_by_priority": by_prio,
            "response_time_by_category": by_cat,
        }

    @tool
    def get_rating_analysis() -> dict[str, Any]:
        """Return customer rating and satisfaction analytics, including overall average rating,

        star rating distribution (1 to 5 stars), ratings by category and priority, and the lowest rated agent.
        """
        return _get_rating_analysis(db_path)

    @tool
    def detect_anomalies() -> dict[str, Any]:
        """Run the complete anomaly detection suite on the support ticket dataset.

        Evaluates operational business rules (High/Critical unresolved >24h, Critical unresolved,
        Long resolution >48h) and statistical IQR resolution-time outliers. Returns anomaly counts
        by severity, IQR thresholds, and detailed evidence for each flagged ticket.
        """
        return _detect_anomalies(db_path)

    return [
        get_ticket_summary,
        get_critical_unresolved_count,
        get_unresolved_ticket_analysis,
        get_agent_performance,
        get_lowest_average_rating_agent,
        get_high_priority_unresolved_tickets,
        get_resolution_time_analysis,
        get_response_time_analysis,
        get_rating_analysis,
        detect_anomalies,
    ]


def get_support_tools(db_path: str | Path | None = None) -> list[BaseTool]:
    """Convenience alias for create_support_tools."""
    return create_support_tools(db_path)

"""Pydantic Request and Response Schemas for Support Analytics API."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class MessageItem(BaseModel):
    """Represents a single message in conversation history."""

    role: str = Field(..., description="Role of the sender ('user' or 'assistant')")
    content: str = Field(..., description="Message text content")


class AskRequest(BaseModel):
    """Request schema for POST /ask."""

    question: str = Field(
        ...,
        description="Natural language question about customer support tickets",
        examples=["How many critical tickets are unresolved?"],
    )
    history: list[MessageItem] | None = Field(
        default=None,
        description="Optional conversation history for context",
    )


class ToolCallItem(BaseModel):
    """Represents an executed tool call in response provenance."""

    tool: str = Field(..., description="Name of the executed tool")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool"
    )


class AskResponse(BaseModel):
    """Response schema for POST /ask."""

    question: str
    answer: str
    tool_calls: list[ToolCallItem] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: str | None = None


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str = "ok"


class AnalyticsSummaryResponse(BaseModel):
    """Response schema for GET /analytics/summary."""

    total_tickets: int
    resolved_tickets: int
    unresolved_tickets: int
    critical_unresolved: int
    by_status: list[dict[str, Any]]
    by_priority: list[dict[str, Any]]
    by_category: list[dict[str, Any]]
    response_time: dict[str, Any]
    resolution_time: dict[str, Any]
    customer_rating: dict[str, Any]


class AnomaliesResponse(BaseModel):
    """Response schema for GET /anomalies."""

    summary: dict[str, Any]
    statistical_thresholds: dict[str, Any]
    anomalies: list[dict[str, Any]]

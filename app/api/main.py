"""FastAPI Application for Customer Support Analytics AI System.

Exposes REST endpoints for health checks, deterministic analytics summaries,
anomaly reports, and natural-language Q&A orchestration via LangChain.
"""

from __future__ import annotations

import logging
from typing import Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.analytics.aggregations import get_ticket_summary
from app.analytics.metrics import (
    get_critical_unresolved_count,
    get_unresolved_ticket_count,
)
from app.anomalies.detector import detect_anomalies
from app.api.schemas import (
    AnalyticsSummaryResponse,
    AnomaliesResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    ToolCallItem,
)
from app.llm.service import ask_support_assistant

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Support AI Analytics API",
    description="Deterministic analytics, anomaly detection, and LangChain AI orchestrator API.",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Perform a lightweight system health check (does not call LLM)."""
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse, tags=["Assistant"])
def ask_question(request: AskRequest) -> AskResponse:
    """Process a natural-language question about customer support tickets.

    The question is interpreted by the LangChain agent, which executes the
    appropriate deterministic Stage 4/5 tools and returns a grounded answer.
    """
    clean_question = request.question.strip() if request.question else ""
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string must not be empty.",
        )

    history_payload = (
        [msg.model_dump() for msg in request.history]
        if request.history is not None
        else None
    )

    try:
        result = ask_support_assistant(
            question=clean_question,
            history=history_payload,
        )

        tool_calls_formatted = [
            ToolCallItem(
                tool=tc.get("tool", ""),
                arguments=tc.get("arguments", {}),
            )
            for tc in result.get("tool_calls", [])
        ]

        return AskResponse(
            question=result.get("question", clean_question),
            answer=result.get("answer", "No response generated."),
            tool_calls=tool_calls_formatted,
            evidence=result.get("evidence", {}),
            success=result.get("success", True),
            error=result.get("error"),
        )

    except Exception as err:
        logger.error("Unhandled error in POST /ask: %s", err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process the question at this time.",
        ) from err


@app.get(
    "/analytics/summary",
    response_model=AnalyticsSummaryResponse,
    tags=["Analytics"],
)
def get_summary() -> AnalyticsSummaryResponse:
    """Retrieve high-level operational metrics and categorical breakdowns."""
    try:
        summary_data = get_ticket_summary()
        critical_unresolved = get_critical_unresolved_count()
        unresolved_count = get_unresolved_ticket_count()
        total_count = summary_data.get("total_tickets", 0)
        resolved_count = total_count - unresolved_count

        return AnalyticsSummaryResponse(
            total_tickets=total_count,
            resolved_tickets=resolved_count,
            unresolved_tickets=unresolved_count,
            critical_unresolved=critical_unresolved,
            by_status=summary_data.get("by_status", []),
            by_priority=summary_data.get("by_priority", []),
            by_category=summary_data.get("by_category", []),
            response_time=summary_data.get("response_time", {}),
            resolution_time=summary_data.get("resolution_time", {}),
            customer_rating=summary_data.get("customer_rating", {}),
        )
    except Exception as err:
        logger.error("Error retrieving analytics summary: %s", err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics summary.",
        ) from err


@app.get("/anomalies", response_model=AnomaliesResponse, tags=["Anomalies"])
def get_anomalies() -> AnomaliesResponse:
    """Retrieve operational business-rule and statistical IQR anomaly reports."""
    try:
        anomaly_data = detect_anomalies()
        return AnomaliesResponse(
            summary=anomaly_data.get("summary", {}),
            statistical_thresholds=anomaly_data.get("statistical_thresholds", {}),
            anomalies=anomaly_data.get("anomalies", []),
        )
    except Exception as err:
        logger.error("Error detecting anomalies: %s", err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve anomaly detections.",
        ) from err

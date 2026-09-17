"""Unit and Integration Tests for FastAPI Endpoints and API Client (Stage 7)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import requests

from app.api.main import app
from app.ui.api_client import APIClient


@pytest.fixture
def client() -> TestClient:
    """Fixture providing FastAPI test client."""
    return TestClient(app)


# =============================================================================
# 1. Health Endpoint Tests (GET /health)
# =============================================================================


def test_health_endpoint(client: TestClient) -> None:
    """GET /health must return 200 OK and status 'ok' without calling the LLM."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}


# =============================================================================
# 2. Ask Endpoint Tests (POST /ask)
# =============================================================================


def test_ask_endpoint_valid_question(client: TestClient) -> None:
    """POST /ask with valid question returns grounded response and evidence."""
    mock_result = {
        "question": "How many critical tickets are unresolved?",
        "answer": "There are 31 unresolved critical tickets.",
        "tool_calls": [{"tool": "get_critical_unresolved_count", "arguments": {}}],
        "evidence": {"critical_unresolved_count": 31},
        "success": True,
        "error": None,
    }

    with patch("app.api.main.ask_support_assistant", return_value=mock_result):
        payload = {"question": "How many critical tickets are unresolved?"}
        response = client.post("/ask", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "How many critical tickets are unresolved?"
        assert "31" in data["answer"]
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["tool"] == "get_critical_unresolved_count"
        assert data["evidence"]["critical_unresolved_count"] == 31
        assert data["success"] is True


def test_ask_endpoint_empty_question(client: TestClient) -> None:
    """POST /ask with empty or whitespace string returns HTTP 400."""
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400
    assert "Question string must not be empty" in response.json()["detail"]

    response_ws = client.post("/ask", json={"question": "   "})
    assert response_ws.status_code == 400


def test_ask_endpoint_invalid_body(client: TestClient) -> None:
    """POST /ask with invalid or missing question payload returns HTTP 422."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_endpoint_internal_error(client: TestClient) -> None:
    """POST /ask returns HTTP 500 when backend service raises an unexpected error."""
    with patch(
        "app.api.main.ask_support_assistant",
        side_effect=RuntimeError("Database crashed"),
    ):
        response = client.post(
            "/ask", json={"question": "How many tickets are there?"}
        )
        assert response.status_code == 500
        assert "Unable to process the question" in response.json()["detail"]


# =============================================================================
# 3. Analytics Summary Endpoint Tests (GET /analytics/summary)
# =============================================================================


def test_analytics_summary_endpoint(client: TestClient) -> None:
    """GET /analytics/summary returns structured metrics from existing analytics layer."""
    response = client.get("/analytics/summary")
    assert response.status_code == 200
    data = response.json()

    assert "total_tickets" in data
    assert "resolved_tickets" in data
    assert "unresolved_tickets" in data
    assert "critical_unresolved" in data
    assert data["total_tickets"] == data["resolved_tickets"] + data["unresolved_tickets"]
    assert isinstance(data["by_status"], list)
    assert isinstance(data["by_priority"], list)
    assert isinstance(data["by_category"], list)
    assert "average_response_time_hrs" in data["response_time"]
    assert "average_resolution_time_hrs" in data["resolution_time"]
    assert "average_customer_rating" in data["customer_rating"]


def test_analytics_summary_error_handling(client: TestClient) -> None:
    """GET /analytics/summary handles unexpected failures gracefully."""
    with patch(
        "app.api.main.get_ticket_summary",
        side_effect=Exception("Database lock error"),
    ):
        response = client.get("/analytics/summary")
        assert response.status_code == 500
        assert "Failed to retrieve analytics summary" in response.json()["detail"]


# =============================================================================
# 4. Anomalies Endpoint Tests (GET /anomalies)
# =============================================================================


def test_anomalies_endpoint(client: TestClient) -> None:
    """GET /anomalies returns business-rule and statistical anomaly results."""
    response = client.get("/anomalies")
    assert response.status_code == 200
    data = response.json()

    assert "summary" in data
    assert "total_anomalies" in data["summary"]
    assert "critical" in data["summary"]
    assert "high" in data["summary"]
    assert "medium" in data["summary"]
    assert "statistical_thresholds" in data
    assert isinstance(data["anomalies"], list)


def test_anomalies_error_handling(client: TestClient) -> None:
    """GET /anomalies handles unexpected failures gracefully."""
    with patch(
        "app.api.main.detect_anomalies",
        side_effect=Exception("Detector failure"),
    ):
        response = client.get("/anomalies")
        assert response.status_code == 500
        assert "Failed to retrieve anomaly detections" in response.json()["detail"]


# =============================================================================
# 5. APIClient Unit Tests
# =============================================================================


def test_api_client_get_health_success() -> None:
    """APIClient.get_health returns True when server reports status ok."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok"}

    with patch("requests.get", return_value=mock_resp):
        client = APIClient(base_url="http://mock-api:8000")
        assert client.get_health() is True


def test_api_client_get_health_failure() -> None:
    """APIClient.get_health returns False when server is unreachable."""
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
        client = APIClient(base_url="http://mock-api:8000")
        assert client.get_health() is False


def test_api_client_ask_question_success() -> None:
    """APIClient.ask_question returns parsed JSON from successful API response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "question": "test",
        "answer": "answer",
        "tool_calls": [],
        "evidence": {},
        "success": True,
    }

    with patch("requests.post", return_value=mock_resp):
        client = APIClient(base_url="http://mock-api:8000")
        res = client.ask_question("test")
        assert res["success"] is True
        assert res["answer"] == "answer"


def test_api_client_ask_question_connection_error() -> None:
    """APIClient.ask_question returns graceful error dict on connection failure."""
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
        client = APIClient(base_url="http://mock-api:8000")
        res = client.ask_question("test")
        assert res["success"] is False
        assert "Unable to connect" in res["answer"]


def test_api_client_ask_question_timeout() -> None:
    """APIClient.ask_question returns graceful error dict on request timeout."""
    with patch("requests.post", side_effect=requests.exceptions.Timeout):
        client = APIClient(base_url="http://mock-api:8000")
        res = client.ask_question("test")
        assert res["success"] is False
        assert "timed out" in res["answer"]


def test_api_client_get_summary_and_anomalies_success() -> None:
    """APIClient summary and anomaly getters work correctly on 200 responses."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"key": "val"}

    with patch("requests.get", return_value=mock_resp):
        client = APIClient(base_url="http://mock-api:8000")
        assert client.get_summary() == {"key": "val"}
        assert client.get_anomalies() == {"key": "val"}


def test_api_client_get_summary_and_anomalies_failure() -> None:
    """APIClient summary and anomaly getters return None on request error."""
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
        client = APIClient(base_url="http://mock-api:8000")
        assert client.get_summary() is None
        assert client.get_anomalies() is None

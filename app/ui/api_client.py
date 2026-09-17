"""API Client Module for Streamlit UI to communicate with FastAPI.

Encapsulates all HTTP communications, error handling, timeouts, and structured
response parsing. Ensures the frontend does not access databases or LLMs directly.
"""

from __future__ import annotations

import os
from typing import Any
import requests

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


class APIClient:
    """Client for interacting with the Customer Support Analytics FastAPI backend."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = (base_url or DEFAULT_API_BASE_URL).rstrip("/")
        self.timeout = timeout

    def get_health(self) -> bool:
        """Check if the FastAPI backend is running and healthy."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200 and resp.json().get("status") == "ok"
        except (requests.exceptions.RequestException, ValueError):
            return False

    def ask_question(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        """Send a natural-language question to POST /ask."""
        url = f"{self.base_url}/ask"
        payload = {"question": question, "history": history}

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 400:
                detail = resp.json().get("detail", "Invalid question.")
                return {
                    "question": question,
                    "answer": f"Validation Error: {detail}",
                    "tool_calls": [],
                    "evidence": {},
                    "success": False,
                    "error": detail,
                }
            else:
                detail = resp.json().get("detail", f"Server error ({resp.status_code})")
                return {
                    "question": question,
                    "answer": "Unable to process the request. Please try again later.",
                    "tool_calls": [],
                    "evidence": {},
                    "success": False,
                    "error": str(detail),
                }

        except requests.exceptions.ConnectionError:
            return {
                "question": question,
                "answer": (
                    "Unable to connect to the support analytics API. "
                    f"Please make sure the FastAPI server is running at {self.base_url}."
                ),
                "tool_calls": [],
                "evidence": {},
                "success": False,
                "error": "ConnectionError: FastAPI server unreachable.",
            }
        except requests.exceptions.Timeout:
            return {
                "question": question,
                "answer": "The request timed out while waiting for a response from the analytics API.",
                "tool_calls": [],
                "evidence": {},
                "success": False,
                "error": "TimeoutError: Request exceeded time limit.",
            }
        except Exception as err:
            return {
                "question": question,
                "answer": "An unexpected error occurred while communicating with the API.",
                "tool_calls": [],
                "evidence": {},
                "success": False,
                "error": str(err),
            }

    def get_summary(self) -> dict[str, Any] | None:
        """Fetch executive analytics summary from GET /analytics/summary."""
        url = f"{self.base_url}/analytics/summary"
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            return None
        except requests.exceptions.RequestException:
            return None

    def get_anomalies(self) -> dict[str, Any] | None:
        """Fetch anomaly reports from GET /anomalies."""
        url = f"{self.base_url}/anomalies"
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            return None
        except requests.exceptions.RequestException:
            return None

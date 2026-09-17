"""Unit and integration tests for Stage 6: LLM + LangChain + Groq Orchestration."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
import pytest

from app.database import build_database_from_csv
from app.llm import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_TEMPERATURE,
    MissingApiKeyError,
    SupportAnalyticsAgent,
    ask_support_assistant,
    create_support_tools,
    get_groq_api_key,
    get_llm,
    get_support_tools,
)

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "support_tickets.csv"


@pytest.fixture(scope="module")
def shared_db_path(tmp_path_factory) -> Path:
    """Fixture initializing a SQLite database from the actual dataset for LLM tool tests."""
    temp_dir = tmp_path_factory.mktemp("llm_db")
    db_path = temp_dir / "llm_tickets.db"
    build_database_from_csv(DATASET_PATH, db_path=db_path)
    return db_path


# =============================================================================
# 1. LangChain Tool Unit Tests
# =============================================================================


def test_tool_definitions_and_contracts(shared_db_path: Path):
    """Test that all required tools exist, have descriptions, and execute deterministically."""
    tools = get_support_tools(shared_db_path)
    tool_map = {t.name: t for t in tools}

    expected_tool_names = {
        "get_ticket_summary",
        "get_critical_unresolved_count",
        "get_unresolved_ticket_analysis",
        "get_agent_performance",
        "get_lowest_average_rating_agent",
        "get_high_priority_unresolved_tickets",
        "get_resolution_time_analysis",
        "get_response_time_analysis",
        "get_rating_analysis",
        "detect_anomalies",
    }

    assert set(tool_map.keys()) == expected_tool_names

    for name, tool_obj in tool_map.items():
        assert len(tool_obj.description) > 10, f"Tool {name} description too short"


def test_critical_unresolved_count_tool(shared_db_path: Path):
    """Test get_critical_unresolved_count tool execution."""
    tools = {t.name: t for t in get_support_tools(shared_db_path)}
    result = tools["get_critical_unresolved_count"].invoke({})
    assert result == {"critical_unresolved_count": 31}


def test_lowest_rating_agent_tool(shared_db_path: Path):
    """Test get_lowest_average_rating_agent tool execution."""
    tools = {t.name: t for t in get_support_tools(shared_db_path)}
    result = tools["get_lowest_average_rating_agent"].invoke({})
    assert result["agent_id"] == "AGT-08"
    assert result["average_rating"] == pytest.approx(3.48, abs=0.01)
    assert result["rated_ticket_count"] == 25


def test_high_priority_unresolved_tool(shared_db_path: Path):
    """Test get_high_priority_unresolved_tickets tool execution."""
    tools = {t.name: t for t in get_support_tools(shared_db_path)}
    result = tools["get_high_priority_unresolved_tickets"].invoke({"limit": 100})
    assert len(result) == 80
    assert all(t["priority"] in ["High", "Critical"] for t in result)


def test_detect_anomalies_tool(shared_db_path: Path):
    """Test detect_anomalies tool execution."""
    tools = {t.name: t for t in get_support_tools(shared_db_path)}
    result = tools["detect_anomalies"].invoke({})
    assert result["summary"]["total_anomalies"] == 153
    assert result["summary"]["critical"] == 31
    assert result["summary"]["high"] == 80


# =============================================================================
# 2. Mock Agent Orchestration Tests
# =============================================================================


class MockToolCallingLLM:
    """Mock LLM simulating tool calling and final response generation."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.call_count = 0
        self.tools = []

    def bind_tools(self, tools: list[BaseTool]):
        self.tools = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return AIMessage(content="Default final response.")


def test_agent_orchestration_loop(shared_db_path: Path):
    """Test agent correctly invokes tool and generates natural language answer from tool output."""
    # Step 1: Model requests tool call
    step1_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_critical_unresolved_count",
                "args": {},
                "id": "call_123",
            }
        ],
    )
    # Step 2: Model synthesizes final grounded answer
    step2_msg = AIMessage(
        content="There are currently 31 unresolved critical tickets in the system."
    )

    mock_llm = MockToolCallingLLM([step1_msg, step2_msg])
    agent = SupportAnalyticsAgent(llm=mock_llm, db_path=shared_db_path)

    result = agent.invoke("How many critical tickets are unresolved?")

    assert result["success"] is True
    assert result["answer"] == "There are currently 31 unresolved critical tickets in the system."
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "get_critical_unresolved_count"
    assert result["evidence"]["get_critical_unresolved_count"] == {"critical_unresolved_count": 31}


def test_agent_anomaly_evidence_flow(shared_db_path: Path):
    """Test agent detects anomalies and preserves full evidence."""
    step1_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "detect_anomalies",
                "args": {},
                "id": "call_anom",
            }
        ],
    )
    step2_msg = AIMessage(
        content="A total of 153 anomaly records were detected across 101 unique tickets, including 31 critical unresolved tickets."
    )

    mock_llm = MockToolCallingLLM([step1_msg, step2_msg])
    agent = SupportAnalyticsAgent(llm=mock_llm, db_path=shared_db_path)

    result = agent.invoke("Are there any anomalies in ticket resolutions or response times?")

    assert result["success"] is True
    assert "153 anomaly records" in result["answer"]
    assert "detect_anomalies" in result["evidence"]
    assert result["evidence"]["detect_anomalies"]["summary"]["total_anomalies"] == 153


# =============================================================================
# 3. Grounding & Anti-Hallucination Guardrail Tests
# =============================================================================


def test_out_of_scope_question_handling(shared_db_path: Path):
    """Test agent declines out-of-scope questions without calling data tools."""
    direct_response = AIMessage(
        content="I can only answer questions regarding the customer support ticket dataset, such as ticket status, agent metrics, resolution times, and anomalies."
    )
    mock_llm = MockToolCallingLLM([direct_response])
    agent = SupportAnalyticsAgent(llm=mock_llm, db_path=shared_db_path)

    result = agent.invoke("What was the weather in Bengaluru yesterday?")

    assert result["success"] is True
    assert len(result["tool_calls"]) == 0
    assert "customer support ticket dataset" in result["answer"]


def test_subjective_best_agent_question_handling(shared_db_path: Path):
    """Test agent calls agent performance tool and provides objective metrics instead of subjective ranking."""
    step1_msg = AIMessage(
        content="",
        tool_calls=[{"name": "get_agent_performance", "args": {}, "id": "call_perf"}],
    )
    step2_msg = AIMessage(
        content="Rather than a single subjective 'best' agent, here are the objective metrics: AGT-07 has the highest average customer rating (3.93) and AGT-01 has the fastest average resolution time (13.43h)."
    )

    mock_llm = MockToolCallingLLM([step1_msg, step2_msg])
    agent = SupportAnalyticsAgent(llm=mock_llm, db_path=shared_db_path)

    result = agent.invoke("Which agent is the best?")

    assert result["success"] is True
    assert result["tool_calls"][0]["tool"] == "get_agent_performance"
    assert "objective metrics" in result["answer"]


# =============================================================================
# 4. Error Handling & Configuration Tests
# =============================================================================


def test_missing_api_key_handling():
    """Test that MissingApiKeyError is raised when GROQ_API_KEY is not set."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(MissingApiKeyError):
            get_groq_api_key()


def test_ask_support_assistant_empty_question():
    """Test service handles empty question safely."""
    res = ask_support_assistant("")
    assert res["success"] is False
    assert "valid question" in res["answer"].lower()


def test_ask_support_assistant_missing_api_key():
    """Test service returns user-friendly error when API key is missing."""
    with patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=True):
        res = ask_support_assistant("How many tickets are open?")
        assert res["success"] is False
        assert "groq api key is not configured" in res["answer"].lower()


def test_ask_support_assistant_successful_mock():
    """Test end-to-end service interface with injected mock LLM."""
    step1_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_lowest_average_rating_agent",
                "args": {},
                "id": "call_low",
            }
        ],
    )
    step2_msg = AIMessage(
        content="Agent AGT-08 has the lowest average customer rating at 3.48 across 25 rated tickets."
    )
    mock_llm = MockToolCallingLLM([step1_msg, step2_msg])

    res = ask_support_assistant(
        question="Which agent has the lowest average customer rating?",
        llm=mock_llm,
    )

    assert res["success"] is True
    assert "AGT-08" in res["answer"]
    assert res["evidence"]["get_lowest_average_rating_agent"]["agent_id"] == "AGT-08"

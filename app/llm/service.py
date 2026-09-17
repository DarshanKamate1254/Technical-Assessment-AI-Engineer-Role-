"""Application Service Interface for Customer Support AI Assistant.

Provides the high-level entrypoint `ask_support_assistant` for natural-language
query processing with comprehensive error handling and logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.llm.agent import SupportAnalyticsAgent
from app.llm.config import MissingApiKeyError

logger = logging.getLogger(__name__)


def ask_support_assistant(
    question: str,
    history: list[dict[str, str]] | None = None,
    db_path: str | Path | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Process a natural-language question using the Support Analytics AI orchestrator.

    Args:
        question: User query string.
        history: Optional conversation history formatted as a list of dicts:
                 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        db_path: Optional path to the SQLite database.
        llm: Optional custom LangChain ChatGroq or mock LLM instance.

    Returns:
        Structured response dictionary containing:
            - question: Original user question
            - answer: Synthesized natural-language response
            - tool_calls: List of tools invoked and their arguments
            - evidence: Structured data returned by the tools
            - success: Boolean indicating successful execution
            - error: Error message string if failed, else None
    """
    if not question or not question.strip():
        return {
            "question": question,
            "answer": "Please provide a valid question regarding customer support tickets.",
            "tool_calls": [],
            "evidence": {},
            "success": False,
            "error": "Empty question provided.",
        }

    try:
        agent = SupportAnalyticsAgent(llm=llm, db_path=db_path)
        return agent.invoke(question=question.strip(), history=history)

    except MissingApiKeyError as err:
        logger.error("Configuration error in ask_support_assistant: %s", err)
        return {
            "question": question,
            "answer": (
                "The Groq API key is not configured. Please set the GROQ_API_KEY environment "
                "variable to enable natural language question answering."
            ),
            "tool_calls": [],
            "evidence": {},
            "success": False,
            "error": str(err),
        }

    except Exception as err:
        logger.error("Execution error in ask_support_assistant: %s", err, exc_info=True)
        return {
            "question": question,
            "answer": (
                "An error occurred while processing your request. Please ensure the database "
                "is available and the Groq service is reachable."
            ),
            "tool_calls": [],
            "evidence": {},
            "success": False,
            "error": str(err),
        }

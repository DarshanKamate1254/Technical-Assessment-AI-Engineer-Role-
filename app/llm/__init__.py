"""LLM Orchestration package using LangChain and Groq."""

from app.llm.agent import SupportAnalyticsAgent
from app.llm.config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_TEMPERATURE,
    LLMConfigError,
    MissingApiKeyError,
    get_groq_api_key,
    get_llm,
)
from app.llm.prompts import SYSTEM_PROMPT
from app.llm.service import ask_support_assistant
from app.llm.tools import create_support_tools, get_support_tools

__all__ = [
    "ask_support_assistant",
    "SupportAnalyticsAgent",
    "create_support_tools",
    "get_support_tools",
    "get_llm",
    "get_groq_api_key",
    "SYSTEM_PROMPT",
    "DEFAULT_GROQ_MODEL",
    "DEFAULT_TEMPERATURE",
    "LLMConfigError",
    "MissingApiKeyError",
]

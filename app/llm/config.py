"""Configuration Module for Groq and LangChain LLM Orchestration.

Manages environment variables, default model selection, zero-temperature
configuration, and safe instantiation of the ChatGroq model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env file from project root if present
load_dotenv(override=False)

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.0


class LLMConfigError(Exception):
    """Base exception for LLM configuration errors."""


class MissingApiKeyError(LLMConfigError):
    """Raised when the GROQ_API_KEY environment variable is not set."""


def get_groq_api_key() -> str:
    """Retrieve the Groq API key from environment variables.

    Raises:
        MissingApiKeyError: If GROQ_API_KEY is not set or empty.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise MissingApiKeyError(
            "GROQ_API_KEY is not configured. Please set the GROQ_API_KEY environment "
            "variable in your environment or in a .env file."
        )
    return api_key


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> ChatGroq:
    """Instantiate a ChatGroq LLM with deterministic temperature settings.

    Args:
        model: Groq model name (defaults to GROQ_MODEL env var or llama-3.3-70b-versatile).
        temperature: Temperature setting (defaults to 0 for deterministic outputs).
        api_key: Optional explicit API key (defaults to GROQ_API_KEY env var).
        **kwargs: Additional keyword arguments for ChatGroq.

    Returns:
        Configured ChatGroq instance.
    """
    model_name = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    temp = (
        temperature
        if temperature is not None
        else float(os.getenv("GROQ_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    )
    resolved_key = api_key or get_groq_api_key()

    return ChatGroq(
        model=model_name,
        temperature=temp,
        groq_api_key=resolved_key,
        **kwargs,
    )

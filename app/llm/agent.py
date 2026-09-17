"""LangChain Agent Orchestrator Module for Customer Support Analytics.

Coordinates the Groq LLM tool-calling loop: interpreting user intent, invoking
the appropriate deterministic tools, gathering structured evidence, and generating
accurate, grounded natural language responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from app.llm.config import get_llm
from app.llm.prompts import SYSTEM_PROMPT
from app.llm.tools import get_support_tools


class SupportAnalyticsAgent:
    """Orchestrates tool selection, tool execution, and natural-language synthesis for support analytics."""

    def __init__(
        self,
        llm: Any | None = None,
        tools: Sequence[BaseTool] | None = None,
        db_path: str | Path | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = 5,
    ) -> None:
        self.db_path = db_path
        self.tools = list(tools) if tools is not None else get_support_tools(db_path)
        self.tool_map: dict[str, BaseTool] = {tool.name: tool for tool in self.tools}
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

        # Lazy LLM instantiation or custom passed LLM
        self._llm = llm
        self._bound_llm = None

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    @property
    def bound_llm(self) -> Any:
        if self._bound_llm is None:
            self._bound_llm = self.llm.bind_tools(self.tools)
        return self._bound_llm

    def invoke(
        self,
        question: str,
        history: list[dict[str, str]] | list[BaseMessage] | None = None,
    ) -> dict[str, Any]:
        """Process a natural-language query through the tool-calling orchestration loop.

        Args:
            question: User's natural language question.
            history: Optional conversation history.

        Returns:
            Structured dictionary containing:
                - question (str)
                - answer (str)
                - tool_calls (list of dicts with tool name & arguments)
                - evidence (aggregated tool execution outputs)
                - success (bool)
                - error (str | None)
        """
        messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]

        # Append conversation history
        if history:
            for item in history:
                if isinstance(item, BaseMessage):
                    messages.append(item)
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=question))

        executed_tool_calls: list[dict[str, Any]] = []
        evidence_data: dict[str, Any] = {}

        for _ in range(self.max_iterations):
            ai_message = self.bound_llm.invoke(messages)
            messages.append(ai_message)

            # Check if tool calls were requested
            tool_calls = getattr(ai_message, "tool_calls", None)
            if not tool_calls:
                # LLM finished reasoning and produced the final response
                return {
                    "question": question,
                    "answer": str(ai_message.content).strip(),
                    "tool_calls": executed_tool_calls,
                    "evidence": evidence_data,
                    "success": True,
                    "error": None,
                }

            # Execute each requested tool call
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"call_{tool_name}")

                executed_tool_calls.append(
                    {"tool": tool_name, "arguments": tool_args}
                )

                selected_tool = self.tool_map.get(tool_name)
                if selected_tool:
                    try:
                        tool_result = selected_tool.invoke(tool_args)
                    except Exception as err:
                        tool_result = {"error": f"Tool execution error: {err}"}
                else:
                    tool_result = {"error": f"Tool '{tool_name}' not found."}

                evidence_data[tool_name] = tool_result

                # Return tool result back into context
                tool_msg_content = (
                    json.dumps(tool_result, default=str)
                    if not isinstance(tool_result, str)
                    else tool_result
                )
                messages.append(
                    ToolMessage(content=tool_msg_content, tool_call_id=tool_id)
                )

        # Fallback if max iterations exceeded
        return {
            "question": question,
            "answer": str(messages[-1].content) if messages else "Max iterations reached.",
            "tool_calls": executed_tool_calls,
            "evidence": evidence_data,
            "success": True,
            "error": "Max tool iterations reached.",
        }

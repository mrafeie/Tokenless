"""
OpenAI Agents SDK model adapter backed by a Tokenless endpoint.

Usage
-----
>>> from agents import Agent, Runner, function_tool
>>> with TokenlessLLM(model="gpt-oss:20b") as llm:
...     agent = Agent(
...         name="Assistant",
...         instructions="Use tools when helpful.",
...         model=llm.as_agents_model(),
...         tools=[my_tool],
...     )
...     result = Runner.run_sync(agent, "Use the available tools.")
...     print(result.final_output)
"""

from __future__ import annotations

from typing import Any

try:
    from agents import AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled

    _AGENTS_AVAILABLE = True
except ImportError:
    _AGENTS_AVAILABLE = False
    AsyncOpenAI = None
    OpenAIChatCompletionsModel = object

    def set_tracing_disabled(*_args, **_kwargs):
        return None


class TokenlessAgentsModel(OpenAIChatCompletionsModel):
    """OpenAI Agents SDK chat-completions model pointed at Tokenless."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "kaggle-free",
        disable_tracing: bool = True,
        **kwargs: Any,
    ):
        if not _AGENTS_AVAILABLE:
            raise ImportError(
                "openai-agents is not installed. "
                "Run: pip install openai-agents  (or pip install tokenless[agents])"
            )

        if disable_tracing:
            set_tracing_disabled(disabled=True)

        client = AsyncOpenAI(api_key=api_key, base_url=f"{base_url.rstrip('/')}/v1")
        super().__init__(model=model, openai_client=client, **kwargs)
        self._tokenless_base_url = base_url.rstrip("/")
        self._tokenless_model = model

    def __repr__(self) -> str:
        return (
            "TokenlessAgentsModel("
            f"model={self._tokenless_model!r}, "
            f"url={self._tokenless_base_url!r})"
        )

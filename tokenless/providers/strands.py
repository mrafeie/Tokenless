"""
Strands Agents model provider backed by a Tokenless endpoint.

Usage
-----
>>> with TokenlessLLM(model="gpt-oss:20b") as llm:
...     model = llm.as_strands_model()
...     agent = Agent(model=model, tools=[my_tool])
...     agent("Summarize the latest AI papers")
"""

from __future__ import annotations

from typing import Any

try:
    from strands.models.openai import OpenAIModel

    _STRANDS_AVAILABLE = True
except ImportError:
    _STRANDS_AVAILABLE = False
    OpenAIModel = object


class TokenlessStrandsModel(OpenAIModel):
    """
    Strands-compatible OpenAI model provider pointed at Tokenless.

    Tokenless exposes Ollama through an OpenAI-compatible Chat Completions
    endpoint, so Strands can use its built-in OpenAI provider.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "kaggle-free",
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if not _STRANDS_AVAILABLE:
            raise ImportError(
                "strands-agents is not installed. "
                "Run: pip install strands-agents  (or pip install tokenless[strands])"
            )
        base_url = base_url.rstrip("/")
        super().__init__(
            model_id=model,
            client_args={
                "api_key": api_key,
                "base_url": f"{base_url}/v1",
            },
            params=params or {},
            **kwargs,
        )
        self._tokenless_base_url = base_url
        self._tokenless_model = model

    def __repr__(self) -> str:
        return (
            "TokenlessStrandsModel("
            f"model={self._tokenless_model!r}, "
            f"url={self._tokenless_base_url!r})"
        )

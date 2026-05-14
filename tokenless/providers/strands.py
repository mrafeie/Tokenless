"""
Strands Agents model provider backed by a Kaggle free-GPU endpoint.

Usage
-----
>>> with TokenlessLLM(model="llama3.1-8b") as llm:
...     model = llm.as_strands_model()
...     agent = Agent(model=model, tools=[my_tool])
...     agent("Summarise the latest AI papers")
"""

from __future__ import annotations

from typing import Any

try:
    from strands.models.openai import OpenAIModel
    _STRANDS_AVAILABLE = True
except ImportError:
    _STRANDS_AVAILABLE = False
    OpenAIModel = object  # fallback so the class definition doesn't break at import


class TokenlessStrandsModel(OpenAIModel):
    """
    A Strands-compatible model provider that routes requests to a
    Kaggle notebook running Ollama or vLLM.

    Inherits from OpenAIModel because Kaggle exposes an OpenAI-compatible API.
    """

    def __init__(self, base_url: str, model: str, **kwargs: Any):
        if not _STRANDS_AVAILABLE:
            raise ImportError(
                "strands-agents is not installed. "
                "Run: pip install strands-agents  (or pip install tokenless[strands])"
            )
        super().__init__(
            model_id=model,
            api_key="kaggle-free",        # dummy — no auth needed
            base_url=f"{base_url}/v1",
            **kwargs,
        )
        self._kaggle_base_url = base_url
        self._model_name = model

    def __repr__(self) -> str:
        return (
            f"TokenlessStrandsModel("
            f"model={self._model_name!r}, "
            f"url={self._kaggle_base_url!r})"
        )

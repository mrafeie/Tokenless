"""
LangChain chat model backed by a Tokenless endpoint.

Usage
-----
>>> with TokenlessLLM(model="gpt-oss:20b") as llm:
...     lc_llm = llm.as_langchain_llm()
...     print(lc_llm.invoke("What is RAG?").content)
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_openai import ChatOpenAI

    _LC_AVAILABLE = True
except ImportError:
    _LC_AVAILABLE = False
    ChatOpenAI = object


class TokenlessLangChainLLM(ChatOpenAI):
    """LangChain ChatOpenAI model pointed at Tokenless."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "kaggle-free",
        **kwargs: Any,
    ):
        if not _LC_AVAILABLE:
            raise ImportError(
                "langchain-openai is not installed. "
                "Run: pip install langchain-openai  (or pip install tokenless[langchain])"
            )
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=f"{base_url.rstrip('/')}/v1",
            **kwargs,
        )

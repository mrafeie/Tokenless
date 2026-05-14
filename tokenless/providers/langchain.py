"""
LangChain LLM wrapper backed by a Kaggle free-GPU endpoint.

Usage
-----
>>> with TokenlessLLM(model="llama3.1-8b") as llm:
...     lc_llm = llm.as_langchain_llm()
...     chain = lc_llm | StrOutputParser()
...     print(chain.invoke("What is RAG?"))
"""

from __future__ import annotations

try:
    from langchain_openai import ChatOpenAI
    _LC_AVAILABLE = True
except ImportError:
    _LC_AVAILABLE = False
    ChatOpenAI = object


class TokenlessLangChainLLM(ChatOpenAI):
    """ChatOpenAI pointed at a Kaggle notebook inference endpoint."""

    def __init__(self, base_url: str, model: str, **kwargs):
        if not _LC_AVAILABLE:
            raise ImportError(
                "langchain-openai is not installed. "
                "Run: pip install langchain-openai  (or pip install tokenless[langchain])"
            )
        super().__init__(
            model=model,
            openai_api_key="kaggle-free",
            openai_api_base=f"{base_url}/v1",
            **kwargs,
        )

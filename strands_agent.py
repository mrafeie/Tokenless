"""
Example: Strands agent running on Kaggle's free GPUs.

Install:
    pip install tokenless[strands] strands-agents-tools
"""

from tokenless import TokenlessLLM
from strands import Agent, tool
from strands_tools import http_request


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


@tool
def summarise(text: str) -> str:
    """Return the first 280 characters of text as a short summary."""
    return text[:280] + ("..." if len(text) > 280 else "")


with TokenlessLLM(model="gpt-oss:20b") as llm:
    model = llm.as_strands_model()

    agent = Agent(
        model=model,
        tools=[word_count, summarise, http_request],
        system_prompt=(
            "You are a helpful research assistant. "
            "Use the tools available to answer questions accurately."
        ),
    )

    result = agent(
        "Fetch https://example.com, summarise the content, and count how many words it has."
    )
    print(result)

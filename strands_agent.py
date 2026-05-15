"""
Example: Strands Agents running on Tokenless.

Install:
    pip install -e ".[strands]"
"""

from strands import Agent, tool

from tokenless import TokenlessLLM


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


@tool
def first_chars(text: str, limit: int = 120) -> str:
    """Return the first characters of a piece of text."""
    return text[:limit]


with TokenlessLLM(model="gpt-oss:20b") as llm:
    agent = Agent(
        model=llm.as_strands_model(
            params={
                "temperature": 0.2,
                "max_tokens": 512,
            }
        ),
        tools=[word_count, first_chars],
        system_prompt=(
            "You are a concise assistant. Use tools when they help answer "
            "the user's request."
        ),
    )

    result = agent(
        "Explain asyncio in two sentences, then use the word_count tool on your explanation."
    )
    print(result)

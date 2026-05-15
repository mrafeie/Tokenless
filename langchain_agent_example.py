"""
Example: LangGraph ReAct agent running on Tokenless.

Install:
    pip install -e ".[langchain]"
"""

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from tokenless import TokenlessLLM


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


with TokenlessLLM(model="gpt-oss:20b") as llm:
    chat = llm.as_langchain_llm(
        temperature=0.2,
        max_tokens=512,
    )
    agent = create_react_agent(chat, tools=[word_count])
    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "Explain asyncio in two sentences, then count the words "
                    "in your explanation.",
                )
            ]
        }
    )
    print(result["messages"][-1].content)

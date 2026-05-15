"""
Example: OpenAI Agents SDK running on Tokenless.

Install:
    pip install -e ".[agents]"
"""

from agents import Agent, Runner, function_tool

from tokenless import TokenlessLLM


@function_tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


with TokenlessLLM(model="gpt-oss:20b") as llm:
    agent = Agent(
        name="Tokenless assistant",
        instructions="Use the available tools when they help. Keep answers concise.",
        model=llm.as_agents_model(),
        tools=[word_count],
    )

    result = Runner.run_sync(
        agent,
        "Explain asyncio in two sentences, then count the words in your explanation.",
    )
    print(result.final_output)

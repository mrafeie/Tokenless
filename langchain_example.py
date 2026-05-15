"""
Example: LangChain running on Tokenless.

Install:
    pip install -e ".[langchain]"
"""

from tokenless import TokenlessLLM


with TokenlessLLM(model="gpt-oss:20b") as llm:
    chat = llm.as_langchain_llm(
        temperature=0.2,
        max_tokens=512,
    )
    response = chat.invoke("Explain asyncio in two sentences.")
    print(response.content)

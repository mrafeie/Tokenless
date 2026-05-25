# Tokenless

Run LLMs on Kaggle's free GPU notebooks with an OpenAI-compatible local client.

[![CI](https://github.com/mrafeie/tokenless/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mrafeie/tokenless/actions/workflows/ci.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

## What It Does

`tokenless` starts a private Kaggle GPU script kernel, installs Ollama, pulls the
requested model, opens a temporary Cloudflare tunnel, and lets your local Python
process call the model through the OpenAI-compatible API.

For `gpt-oss:20b`, the model is downloaded once during `llm.start()`. Repeated
`llm.send(...)` calls reuse the running Kaggle kernel until `llm.stop()`.

## Quick Start

Install from PyPI:

```bash
pip install tokenless
```
## Kaggle Credentials

Tokenless needs your Kaggle username and API key to create private Kaggle
kernels. Create an API token from your Kaggle account settings, then use one of
the methods below.

### Python

Pass credentials directly when creating the client:

```python
from tokenless import TokenlessLLM

llm = TokenlessLLM(
    model="gpt-oss:20b",
    kaggle_username="your_username",
    kaggle_key="your_api_key",
)
```

Or set environment variables inside Python before calling `start()`:

```python
import os
from tokenless import TokenlessLLM

os.environ["KAGGLE_USERNAME"] = "your_username"
os.environ["KAGGLE_KEY"] = "your_api_key"

llm = TokenlessLLM(model="gpt-oss:20b")
```
Set your Kaggle credentials, then run your first prompt:

```python
from tokenless import TokenlessLLM

llm = TokenlessLLM(model="gpt-oss:20b")
llm.start()
msg = llm.send("Explain asyncio in two sentences.")
print(msg)
llm.stop()
```

`llm.start()` shows a terminal progress bar while Kaggle starts the GPU kernel,
installs Ollama, downloads the model, and opens the public endpoint. Pass
`show_progress=False` to disable it.

### PDF Input

For `gpt-oss:20b`, `start()` can upload a PDF to Kaggle as a temporary private
dataset and convert it to Markdown inside the Kaggle kernel. This supports both
text PDFs and scanned/image PDFs.

```python
from tokenless import TokenlessLLM

llm = TokenlessLLM(model="gpt-oss:20b")
markdown = llm.start(file_path="paper.pdf")
print(markdown)
```

Pass `kaggle_prompt` as well to ask the model about the converted Markdown:

```python
answer = llm.start(
    file_path="paper.pdf",
    kaggle_prompt="Summarize this PDF in five bullets.",
)
```

## Supported Models

| Model | Backend | Notes |
|-------|---------|-------|
| gpt-oss:20b | Ollama persistent kernel | Downloads once per `start()` |
| llama3.1-8b | Endpoint mode / smoke path | Coming Soon |
| mistral-7b | Endpoint mode / smoke path | Coming Soon |
| gemma-2-9b | Endpoint mode / smoke path | Coming Soon |
| qwen2.5-7b | Endpoint mode / smoke path | Coming Soon |

## Agent Integrations

### OpenAI Agents SDK

```bash
pip install "tokenless[agents]"
```

```python
from agents import Agent, Runner, function_tool
from tokenless import TokenlessLLM

@function_tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())

with TokenlessLLM(model="gpt-oss:20b") as llm:
    agent = Agent(
        name="Tokenless assistant",
        instructions="Use tools when helpful. Keep answers concise.",
        model=llm.as_agents_model(),
        tools=[word_count],
    )
    result = Runner.run_sync(agent, "Explain asyncio, then count the words.")
    print(result.final_output)
```

If you are running in Jupyter, IPython, VS Code interactive, or another
environment that already has an event loop, use the async runner instead:

```python
from agents import Agent, Runner, function_tool
from tokenless import TokenlessLLM

@function_tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())

llm = TokenlessLLM(model="gpt-oss:20b")
llm.start()

agent = Agent(
    name="Tokenless assistant",
    instructions="Use tools when helpful. Keep answers concise.",
    model=llm.as_agents_model(),
    tools=[word_count],
)

result = await Runner.run(agent, "Explain asyncio, then count the words.")
print(result.final_output)

llm.stop()
```

Tokenless uses the Agents SDK chat-completions model adapter because Ollama
exposes an OpenAI-compatible Chat Completions endpoint.

### Strands Agents

```bash
pip install "tokenless[strands]"
```

```python
from tokenless import TokenlessLLM
from strands import Agent, tool

@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())

with TokenlessLLM(model="gpt-oss:20b") as llm:
    agent = Agent(
        model=llm.as_strands_model(params={"temperature": 0.2}),
        tools=[word_count],
        system_prompt="Use tools when helpful. Keep answers concise.",
    )
    result = agent("Explain asyncio, then count the words in your explanation.")
    print(result)
```

### LangChain

```bash
pip install "tokenless[langchain]"
```

```python
from tokenless import TokenlessLLM

with TokenlessLLM(model="gpt-oss:20b") as llm:
    chat = llm.as_langchain_llm(temperature=0.2, max_tokens=512)
    response = chat.invoke("Explain asyncio in two sentences.")
    print(response.content)
```

For PDF question answering, start the Kaggle endpoint with server-side PDF
context. The PDF is uploaded to Kaggle, converted to Markdown there, and kept on
Kaggle. Each LangChain question sends only the question; the Kaggle proxy
selects relevant Markdown chunks before calling the model.

```python
from tokenless import TokenlessLLM

llm = TokenlessLLM(model="gpt-oss:20b")
llm.start(file_path=r"C:\path\to\paper.pdf", pdf_context=True)

chat = llm.as_langchain_llm(temperature=0.2, max_tokens=512)
response = chat.invoke("What are the main findings in this PDF?")
print(response.content)

llm.stop()
```

For a LangGraph ReAct agent:

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tokenless import TokenlessLLM

@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())

with TokenlessLLM(model="gpt-oss:20b") as llm:
    chat = llm.as_langchain_llm(temperature=0.2, max_tokens=512)
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
```

Install all optional integrations with:

```bash
pip install "tokenless[all]"
```

## How It Works

1. `TokenlessLLM.start()` uploads a long-running private Kaggle script kernel.
2. The kernel installs dependencies, starts Ollama, and pulls `gpt-oss:20b`.
3. The kernel starts a Cloudflare tunnel for Ollama's OpenAI-compatible endpoint.
4. The kernel posts the tunnel URL back through a one-time rendezvous topic.
5. `llm.send(...)` sends prompts to the running endpoint.
6. `llm.stop()` deletes the Kaggle kernel.

## Security Notes

- Do not commit Kaggle credentials, `.env` files, or `~/.kaggle/kaggle.json`.
- The Cloudflare tunnel URL is temporary but public while the kernel is running.
- Stop the kernel with `llm.stop()` when you are done.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m py_compile tokenless/client.py tokenless/notebook.py
```

CI runs offline tests only and does not require Kaggle credentials. To run the
live Kaggle smoke test locally:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
python -m pytest --run-live-kaggle -m live_kaggle
```

## License

MIT. See [LICENSE](LICENSE).

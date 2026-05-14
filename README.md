# Tokenless

Run LLMs on Kaggle's free GPU notebooks with an OpenAI-compatible local client.

[![CI](https://github.com/mrafeie/tokenless/actions/workflows/ci.yml/badge.svg)](https://github.com/mrafeie/tokenless/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

## What It Does

`tokenless` starts a private Kaggle GPU script kernel, installs Ollama, pulls the
requested model, opens a temporary Cloudflare tunnel, and lets your local Python
process call the model through the OpenAI-compatible API.

For `gpt-oss:20b`, the model is downloaded once during `llm.start()`. Repeated
`llm.send(...)` calls reuse the running Kaggle kernel until `llm.stop()`.

## Quick Start

Install from source:

```bash
pip install -e .
```

Set your Kaggle credentials:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

On PowerShell:

```powershell
$env:KAGGLE_USERNAME = "your_username"
$env:KAGGLE_KEY = "your_api_key"
```

Run your first prompt:

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

## Supported Models

| Model | Backend | Notes |
|-------|---------|-------|
| gpt-oss:20b | Ollama persistent kernel | Downloads once per `start()` |
| llama3.1-8b | Endpoint mode / smoke path | Experimental |
| llama3.1-70b | Endpoint mode / smoke path | Experimental |
| mistral-7b | Endpoint mode / smoke path | Experimental |
| gemma-2-9b | Endpoint mode / smoke path | Experimental |
| qwen2.5-7b | Endpoint mode / smoke path | Experimental |

## Agent Integrations

### Strands Agents

```python
from tokenless import TokenlessLLM
from strands import Agent

with TokenlessLLM(model="gpt-oss:20b") as llm:
    agent = Agent(model=llm.as_strands_model())
    agent("Summarize how event loops work.")
```

### LangChain

```python
from tokenless import TokenlessLLM

with TokenlessLLM(model="gpt-oss:20b") as llm:
    lc_llm = llm.as_langchain_llm()
    print(lc_llm.invoke("What is RAG?"))
```

Install optional integrations with:

```bash
pip install "tokenless[strands]"
pip install "tokenless[langchain]"
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

## License

MIT. See [LICENSE](LICENSE).

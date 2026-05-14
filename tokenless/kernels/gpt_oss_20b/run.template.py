# Mirrors gpt-oss-20B.ipynb: Ollama + gpt-oss:20b on Kaggle GPU script kernel.
import base64
import os
import subprocess
import sys
import time
from pathlib import Path

PROMPT_B64 = "__TOKENLESS_PROMPT_B64__"
SYSTEM_B64 = "__TOKENLESS_SYSTEM_B64__"
OLLAMA_MODEL = "gpt-oss:20b"

OUT = Path("/kaggle/working/tokenless_model_response.txt")
ERR = Path("/kaggle/working/tokenless_model_error.txt")


def _fail(msg: str) -> None:
    ERR.write_text(msg, encoding="utf-8")
    raise SystemExit(1)


subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openai"])
from openai import OpenAI  # noqa: E402 — pip must run first on Kaggle

prompt = base64.b64decode(PROMPT_B64).decode("utf-8")
system = base64.b64decode(SYSTEM_B64).decode("utf-8")

print("Installing Ollama...")
if os.system("apt-get update -qq && apt-get install -y -qq zstd") != 0:
    _fail("Failed to install zstd, which Ollama needs for extraction.")
if os.system("curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null") != 0:
    _fail("Ollama installer failed.")

print("Starting Ollama server...")
os.system("nohup ollama serve > /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &")
time.sleep(5)

if os.system("ps aux | grep -E 'ollama serve' | grep -v grep > /dev/null 2>&1") != 0:
    _fail("Ollama server failed to start.")

print("Downloading model (this can take a long time)...")
if os.system(f"ollama pull {OLLAMA_MODEL}") != 0:
    _fail("Model download failed.")

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
print("Querying model...")
try:
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    answer = response.choices[0].message.content or ""
except Exception as e:  # noqa: BLE001 — script kernel; surface any failure
    _fail(repr(e))

OUT.write_text(answer, encoding="utf-8")
print("Wrote response to", OUT)

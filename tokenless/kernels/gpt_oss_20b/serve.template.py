# Long-running Kaggle script kernel for gpt-oss:20b via Ollama.
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_MODEL = "gpt-oss:20b"
NTFY_TOPIC = "__TOKENLESS_NTFY_TOPIC__"

OUT = Path("/kaggle/working/tokenless_public_url.txt")
ERR = Path("/kaggle/working/tokenless_server_error.txt")
CLOUDFLARED_LOG = Path("/tmp/tokenless_cloudflared.log")


def _fail(msg: str) -> None:
    ERR.write_text(msg, encoding="utf-8")
    raise SystemExit(1)


def _run(cmd: str, error: str) -> None:
    if os.system(cmd) != 0:
        _fail(error)


def _publish(message: str) -> None:
    if not NTFY_TOPIC or NTFY_TOPIC.startswith("__TOKENLESS_"):
        return
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10).read()
    except (urllib.error.URLError, TimeoutError):
        pass


print("Installing system dependencies...", flush=True)
_publish("Installing system dependencies")
_run(
    "apt-get update -qq && apt-get install -y -qq curl zstd",
    "Failed to install system dependencies.",
)

print("Installing Ollama...", flush=True)
_publish("Installing Ollama")
_run("curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null", "Ollama installer failed.")

print("Starting Ollama server...", flush=True)
_publish("Starting Ollama server")
os.system(
    "OLLAMA_HOST=0.0.0.0:11434 nohup ollama serve "
    "> /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &"
)
time.sleep(5)

if os.system("ps aux | grep -E 'ollama serve' | grep -v grep > /dev/null 2>&1") != 0:
    _fail("Ollama server failed to start.")

print("Downloading model (this can take a long time)...", flush=True)
_publish("Downloading gpt-oss:20b")
_run(f"ollama pull {OLLAMA_MODEL}", "Model download failed.")

print("Installing cloudflared...", flush=True)
_publish("Installing public tunnel")
_run(
    "curl -L --output /tmp/cloudflared.deb "
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64.deb "
    "&& dpkg -i /tmp/cloudflared.deb",
    "cloudflared installation failed.",
)

print("Starting public tunnel...", flush=True)
_publish("Starting public tunnel")
os.system(
    f"nohup cloudflared tunnel --url http://localhost:11434 --no-autoupdate "
    f"> {CLOUDFLARED_LOG} 2>&1 &"
)

url = None
deadline = time.time() + 180
pattern = re.compile(r"https://[-a-zA-Z0-9.]+\.trycloudflare\.com")
while time.time() < deadline:
    if CLOUDFLARED_LOG.exists():
        text = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="replace")
        match = pattern.search(text)
        if match:
            url = match.group(0)
            break
    time.sleep(1)

if not url:
    _fail("Timed out waiting for cloudflared public URL.")

OUT.write_text(url, encoding="utf-8")
print(f"TOKENLESS_PUBLIC_URL={url}", flush=True)
_publish(url)
print("Published TOKENLESS_PUBLIC_URL to rendezvous channel.", flush=True)

print("Tokenless GPT-OSS server is ready. Keeping Kaggle kernel alive...", flush=True)

while True:
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=5).read()
    except Exception as e:  # noqa: BLE001 - keep the kernel log useful
        print(f"Ollama health check failed: {e!r}", flush=True)
    time.sleep(60)

# Long-running Kaggle script kernel for gpt-oss:20b via Ollama.
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OLLAMA_MODEL = "gpt-oss:20b"
NTFY_TOPIC = "__TOKENLESS_NTFY_TOPIC__"
INPUT_DATASET_SLUG_B64 = "__TOKENLESS_INPUT_DATASET_SLUG_B64__"
INPUT_FILENAME_B64 = "__TOKENLESS_INPUT_FILENAME_B64__"

OUT = Path("/kaggle/working/tokenless_public_url.txt")
ERR = Path("/kaggle/working/tokenless_server_error.txt")
CLOUDFLARED_LOG = Path("/tmp/tokenless_cloudflared.log")
PDF_MARKDOWN = Path("/kaggle/working/tokenless_pdf_context.md")


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


def _decode_b64(value: str) -> str:
    if not value:
        return ""
    return base64.b64decode(value).decode("utf-8")


def _install_pdf_tools() -> None:
    print("Installing PDF conversion tools...", flush=True)
    _publish("Installing PDF conversion tools")
    _run(
        "apt-get update -qq && apt-get install -y -qq tesseract-ocr ghostscript qpdf unpaper",
        "Failed to install OCR system dependencies.",
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "ocrmypdf", "pymupdf4llm"]
    )


def _find_pdf_path(dataset_slug: str, filename: str) -> Path:
    dataset_dir = Path("/kaggle/input") / dataset_slug
    expected = dataset_dir / filename
    print("Looking for uploaded PDF at", expected, flush=True)
    if expected.is_file():
        return expected
    candidates = sorted(
        p for p in dataset_dir.glob("**/*") if p.is_file() and p.suffix.lower() == ".pdf"
    )
    if len(candidates) == 1:
        print("Using discovered uploaded PDF at", candidates[0], flush=True)
        return candidates[0]
    available = sorted(str(p) for p in Path("/kaggle/input").glob("**/*"))
    _fail(
        f"Uploaded PDF was not found at {expected}. "
        f"Available Kaggle input paths: {available[:50]}"
    )


def _pdf_to_markdown(pdf_path: Path) -> str:
    import ocrmypdf  # noqa: E402
    import pymupdf4llm  # noqa: E402

    ocr_pdf = Path("/kaggle/working/tokenless_input_ocr.pdf")
    print("Running OCR pass for image and mixed-content PDFs...", flush=True)
    try:
        ocrmypdf.ocr(
            str(pdf_path),
            str(ocr_pdf),
            skip_text=True,
            deskew=True,
            progress_bar=False,
        )
        source = ocr_pdf
    except ocrmypdf.exceptions.PriorOcrFoundError:
        source = pdf_path
    except Exception as e:  # noqa: BLE001 - fall back for already-readable PDFs
        print(f"OCR pass failed; trying direct Markdown conversion: {e!r}", flush=True)
        source = pdf_path

    print("Converting PDF to Markdown...", flush=True)
    markdown = pymupdf4llm.to_markdown(str(source))
    if not markdown.strip():
        _fail("PDF conversion produced empty Markdown.")
    PDF_MARKDOWN.write_text(markdown, encoding="utf-8")
    return markdown


def _prepare_pdf_context() -> None:
    dataset_slug = _decode_b64(INPUT_DATASET_SLUG_B64)
    filename = _decode_b64(INPUT_FILENAME_B64)
    if not dataset_slug or not filename:
        return
    _install_pdf_tools()
    markdown = _pdf_to_markdown(_find_pdf_path(dataset_slug, filename))
    print(f"Stored PDF Markdown context ({len(markdown)} chars).", flush=True)
    _publish("PDF context ready")


def _inject_pdf_context(payload: dict) -> dict:
    if not PDF_MARKDOWN.exists():
        return payload
    markdown = PDF_MARKDOWN.read_text(encoding="utf-8")
    messages = list(payload.get("messages") or [])
    question = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            question = str(message.get("content") or "")
            break
    selected_context = _select_pdf_context(markdown, question)
    context = (
        "You are answering questions about an uploaded PDF. Use the Markdown "
        "context below as the source of truth. If the answer is not in the PDF, "
        "say you cannot find it in the document.\n\n"
        f"<pdf_markdown>\n{selected_context}\n</pdf_markdown>"
    )
    if messages and messages[0].get("role") == "system":
        messages[0] = {**messages[0], "content": f"{messages[0].get('content', '')}\n\n{context}"}
    else:
        messages.insert(0, {"role": "system", "content": context})
    return {**payload, "messages": messages}


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]{3,}", text)}


def _split_markdown(markdown: str, chunk_size: int = 3000, overlap: int = 300) -> list[str]:
    chunks = []
    start = 0
    while start < len(markdown):
        end = min(len(markdown), start + chunk_size)
        chunks.append(markdown[start:end])
        if end == len(markdown):
            break
        start = max(0, end - overlap)
    return chunks


def _select_pdf_context(markdown: str, question: str, max_chars: int = 12000) -> str:
    chunks = _split_markdown(markdown)
    query_terms = _tokenize(question)
    if not query_terms:
        return "\n\n---\n\n".join(chunks[: max(1, max_chars // 3000)])
    scored = []
    for index, chunk in enumerate(chunks):
        score = len(query_terms & _tokenize(chunk))
        if score:
            scored.append((score, index, chunk))
    if not scored:
        return "\n\n---\n\n".join(chunks[: max(1, max_chars // 3000)])
    selected = []
    total = 0
    for _score, _index, chunk in sorted(scored, key=lambda item: (-item[0], item[1])):
        if total + len(chunk) > max_chars and selected:
            break
        selected.append(chunk)
        total += len(chunk)
    return "\n\n---\n\n".join(selected)


class TokenlessProxy(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        target = f"http://localhost:11434{self.path}"
        try:
            with urllib.request.urlopen(target, timeout=30) as response:
                body = response.read()
                self.send_response(response.status)
                self.send_header(
                    "Content-Type",
                    response.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except Exception as e:  # noqa: BLE001
            self._send_json(502, {"error": repr(e)})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            if self.path.rstrip("/") == "/v1/chat/completions":
                payload = _inject_pdf_context(payload)
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"http://localhost:11434{self.path}",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as response:
                response_body = response.read()
                self.send_response(response.status)
                self.send_header(
                    "Content-Type",
                    response.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
        except Exception as e:  # noqa: BLE001
            self._send_json(502, {"error": repr(e)})


def _start_proxy() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8000), TokenlessProxy)
    print("Starting Tokenless PDF context proxy on port 8000...", flush=True)
    server.serve_forever()


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

_prepare_pdf_context()

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
target_port = 8000 if PDF_MARKDOWN.exists() else 11434
if PDF_MARKDOWN.exists():
    import threading

    threading.Thread(target=_start_proxy, daemon=True).start()
    time.sleep(2)
os.system(
    f"nohup cloudflared tunnel --url http://localhost:{target_port} --no-autoupdate "
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

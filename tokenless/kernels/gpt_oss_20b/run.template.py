# Mirrors gpt-oss-20B.ipynb: Ollama + gpt-oss:20b on Kaggle GPU script kernel.
import base64
import os
import re
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)

PROMPT_B64 = "__TOKENLESS_PROMPT_B64__"
SYSTEM_B64 = "__TOKENLESS_SYSTEM_B64__"
INPUT_DATASET_SLUG_B64 = "__TOKENLESS_INPUT_DATASET_SLUG_B64__"
INPUT_FILENAME_B64 = "__TOKENLESS_INPUT_FILENAME_B64__"
OLLAMA_MODEL = "gpt-oss:20b"

OUT = Path("/kaggle/working/tokenless_model_response.txt")
ERR = Path("/kaggle/working/tokenless_model_error.txt")


def _fail(msg: str) -> None:
    ERR.write_text(msg, encoding="utf-8")
    raise SystemExit(1)


def _decode_b64(value: str) -> str:
    if not value:
        return ""
    return base64.b64decode(value).decode("utf-8")


def _clean_source_label(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value.replace("|", "/") if value else "unknown"


def _first_text_line(block: str) -> str:
    for line in block.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _update_source_context(block: str, section: str) -> str:
    first_line = _first_text_line(block)
    heading = re.match(r"^#{1,6}\s+(.+?)\s*#*$", first_line)
    label = heading.group(1).strip() if heading else first_line
    section_match = re.match(
        r"^(?:section|sec\.|§)\s+([A-Za-z0-9IVXLCDMivxlcdm_.-]+)(?:\s*[:.-]\s*(.+))?$",
        label,
        flags=re.IGNORECASE,
    )
    if heading:
        section = label
    if section_match:
        section = label
    return section


def _chunk_page_number(chunk: dict, fallback: int) -> int:
    metadata = chunk.get("metadata") or {}
    for key in ("page", "page_number", "page_index"):
        value = chunk.get(key, metadata.get(key))
        if isinstance(value, int):
            return value + 1 if key == "page_index" else value
    return fallback


def _chunk_text(chunk: dict) -> str:
    for key in ("text", "markdown", "content"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    return ""


def _chunk_section_hint(chunk: dict) -> str:
    toc_items = chunk.get("toc_items") or []
    if not toc_items:
        return ""
    item = sorted(toc_items, key=lambda entry: entry[0] if entry else 0)[-1]
    return str(item[1]).strip() if len(item) > 1 else ""


def _add_pdf_source_markers(page_chunks) -> str:
    if isinstance(page_chunks, str):
        page_chunks = [{"text": page_chunks, "page": 1}]

    output = []
    current_section = "unknown"
    for fallback_page, chunk in enumerate(page_chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        page_number = _chunk_page_number(chunk, fallback_page)
        section_hint = _chunk_section_hint(chunk)
        if section_hint:
            current_section = section_hint
        output.append(f"<!-- tokenless-page: page={page_number} -->")

        text = _chunk_text(chunk).strip()
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if not block:
                continue
            current_section = _update_source_context(
                block,
                current_section,
            )
            output.append(
                "[source: "
                f"page={page_number} | "
                f"section={_clean_source_label(current_section)}"
                "]"
            )
            output.append(block)

    return "\n\n".join(output).strip()


def _install_pdf_tools() -> None:
    print("Installing PDF conversion tools...")
    apt_packages = "tesseract-ocr ghostscript qpdf unpaper"
    if os.system(f"apt-get update -qq && apt-get install -y -qq {apt_packages}") != 0:
        _fail("Failed to install OCR system dependencies.")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "ocrmypdf",
            "pymupdf4llm",
        ]
    )


def _pdf_to_markdown(pdf_path: Path) -> str:
    import ocrmypdf  # noqa: E402
    import pymupdf4llm  # noqa: E402

    ocr_pdf = Path("/kaggle/working/tokenless_input_ocr.pdf")
    print("Running OCR pass for image and mixed-content PDFs...")
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
        print(f"OCR pass failed; trying direct Markdown conversion: {e!r}")
        source = pdf_path

    print("Converting PDF to source-aware Markdown...")
    try:
        page_chunks = pymupdf4llm.to_markdown(str(source), page_chunks=True)
    except TypeError:
        page_chunks = pymupdf4llm.to_markdown(str(source))
    markdown = _add_pdf_source_markers(page_chunks)
    if not markdown.strip():
        _fail("PDF conversion produced empty Markdown.")
    return markdown


def _find_pdf_path(dataset_slug: str, filename: str) -> Path:
    dataset_dir = Path("/kaggle/input") / dataset_slug
    expected = dataset_dir / filename
    print("Looking for uploaded PDF at", expected)
    if expected.is_file():
        return expected

    candidates = sorted(
        p for p in dataset_dir.glob("**/*") if p.is_file() and p.suffix.lower() == ".pdf"
    )
    if len(candidates) == 1:
        print("Using discovered uploaded PDF at", candidates[0])
        return candidates[0]
    if candidates:
        available_pdfs = [str(p) for p in candidates[:50]]
        _fail(f"Expected {expected}, but found multiple PDFs: {available_pdfs}")

    available = sorted(str(p) for p in Path("/kaggle/input").glob("**/*"))
    _fail(
        f"Uploaded PDF was not found at {expected}. "
        f"Available Kaggle input paths: {available[:50]}"
    )


def main() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openai"])
    from openai import OpenAI  # noqa: E402 - pip must run first on Kaggle

    prompt = _decode_b64(PROMPT_B64)
    system = _decode_b64(SYSTEM_B64)
    input_dataset_slug = _decode_b64(INPUT_DATASET_SLUG_B64)
    input_filename = _decode_b64(INPUT_FILENAME_B64)

    pdf_markdown = ""
    if input_dataset_slug and input_filename:
        pdf_path = _find_pdf_path(input_dataset_slug, input_filename)
        _install_pdf_tools()
        pdf_markdown = _pdf_to_markdown(pdf_path)
        if not prompt.strip():
            OUT.write_text(pdf_markdown, encoding="utf-8")
            print("Wrote converted PDF Markdown to", OUT)
            return
        prompt = (
            f"{prompt}\n\n"
            "Use the following source-aware Markdown converted from the uploaded PDF. "
            "Each block has page and section source markers:\n\n"
            f"{pdf_markdown}"
        )

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
    except Exception as e:  # noqa: BLE001 - script kernel; surface any failure
        _fail(repr(e))

    OUT.write_text(answer, encoding="utf-8")
    print("Wrote response to", OUT)


try:
    main()
except SystemExit:
    raise
except Exception:
    ERR.write_text(traceback.format_exc(), encoding="utf-8")
    raise

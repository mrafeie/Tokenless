"""Kaggle notebook lifecycle (GPU kernel hosting the inference server)."""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional

import requests

logger = logging.getLogger(__name__)

SMOKE_OUTPUT_FILENAME = "tokenless_smoke_message.txt"
DEFAULT_SMOKE_KERNEL_SLUG = "tokenless-smoke"
_SMOKE_SCRIPT_NAME = "smoke.py"

GPT_OSS_MODEL_ID = "gpt-oss:20b"

GPT_OSS_RESPONSE_FILENAME = "tokenless_model_response.txt"
GPT_OSS_ERROR_FILENAME = "tokenless_model_error.txt"
DEFAULT_GPT_OSS_KERNEL_SLUG = "tokenless-gpt-oss-20b"
DEFAULT_GPT_OSS_SERVER_KERNEL_SLUG = "tokenless-gpt-oss-20b-server"
_GPT_OSS_SCRIPT_NAME = "run.py"
_GPT_OSS_SERVER_SCRIPT_NAME = "serve.py"
_GPT_OSS_TEMPLATE = (
    Path(__file__).resolve().parent / "kernels" / "gpt_oss_20b" / "run.template.py"
)
_GPT_OSS_SERVER_TEMPLATE = (
    Path(__file__).resolve().parent / "kernels" / "gpt_oss_20b" / "serve.template.py"
)


def _empty_b64() -> str:
    return base64.b64encode(b"").decode("ascii")


def _smoke_script_template() -> Path:
    return Path(__file__).resolve().parent / "kernels" / "smoke" / _SMOKE_SCRIPT_NAME


def _kaggle_api_cls():
    from kaggle.api.kaggle_api_extended import KaggleApi

    return KaggleApi


def _kernel_worker_status_cls():
    from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus

    return KernelWorkerStatus


def run_gpt_oss_20b_prompt_on_kaggle(
    prompt: str,
    *,
    system_prompt: str = "You are a helpful AI assistant.",
    username: Optional[str] = None,
    key: Optional[str] = None,
    kernel_slug: str = DEFAULT_GPT_OSS_KERNEL_SLUG,
    status_timeout: int = 36_000,
    poll_interval: float = 5.0,
    kernel_session_timeout: int = 36_000,
    accelerator: Optional[str] = "NvidiaTeslaT4",
) -> str:
    """
    Push a GPU script kernel (same flow as ``gpt-oss-20B.ipynb``), wait for completion,
    and return the model reply from the kernel output file.

    This performs a **full batch run** on Kaggle (install Ollama, pull ``gpt-oss:20b``,
    one chat completion). Expect long runtimes and Kaggle session limits.
    """
    mgr = KaggleNotebookManager(
        username=username,
        key=key,
        model=GPT_OSS_MODEL_ID,
        kernel_slug=kernel_slug,
    )
    return mgr.run_ollama_gpt_oss_20b_prompt(
        prompt,
        system_prompt=system_prompt,
        kernel_slug=kernel_slug,
        status_timeout=status_timeout,
        poll_interval=poll_interval,
        kernel_session_timeout=kernel_session_timeout,
        accelerator=accelerator,
    )


class KaggleNotebookManager:
    """Manages the lifecycle of the remote Kaggle GPU notebook."""

    _current: ClassVar[Optional["KaggleNotebookManager"]] = None

    def __init__(
        self,
        username: Optional[str] = None,
        key: Optional[str] = None,
        model: str = "",
        backend: str = "ollama",
        kernel_slug: str = DEFAULT_SMOKE_KERNEL_SLUG,
    ):
        self.username = username or os.environ.get("KAGGLE_USERNAME")
        self.key = key or os.environ.get("KAGGLE_KEY")
        self.model = model
        self.backend = backend
        self.kernel_slug = kernel_slug
        self.public_url: Optional[str] = None
        self.smoke_test_message: Optional[str] = None
        self.kernel_ref: Optional[str] = None
        self._temporary_dataset_owner: Optional[str] = None
        self._temporary_dataset_slug: Optional[str] = None

    def launch(
        self,
        *,
        file_path: Optional[str] = None,
        pdf_context: bool = False,
        kaggle_prompt: Optional[str] = None,
        kaggle_system_prompt: Optional[str] = None,
        smoke_status_timeout: int = 1800,
        smoke_poll_interval: float = 3.0,
        smoke_kernel_session_timeout: int = 600,
        gpt_oss_status_timeout: int = 36_000,
        gpt_oss_poll_interval: float = 5.0,
        gpt_oss_kernel_session_timeout: int = 36_000,
        gpt_oss_accelerator: Optional[str] = "NvidiaTeslaT4",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Set ``public_url`` from env, run a smoke script kernel, or prepare the
        ``gpt-oss:20b`` batch runner. Passing ``kaggle_prompt`` or ``file_path``
        keeps the one-shot GPT-OSS behavior unless ``pdf_context`` is true, in
        which case the GPT-OSS server keeps the converted PDF Markdown on Kaggle.
        """
        self.__class__._current = self
        self.smoke_test_message = None
        self.kernel_ref = None
        self._temporary_dataset_owner = None
        self._temporary_dataset_slug = None
        input_path = self._validate_pdf_file_path(file_path) if file_path else None
        if progress_callback:
            progress_callback("Checking endpoint configuration")

        if input_path and self.model != GPT_OSS_MODEL_ID:
            raise ValueError(
                "file_path PDF conversion is currently supported only for gpt-oss:20b."
            )
        if pdf_context and input_path is None:
            raise ValueError("pdf_context=True requires file_path.")

        env_url = os.environ.get("TOKENLESS_PUBLIC_URL")
        if input_path and env_url:
            logger.info(
                "Ignoring TOKENLESS_PUBLIC_URL because file_path requires a Kaggle batch run."
            )
            env_url = None
        if env_url:
            if progress_callback:
                progress_callback("Using TOKENLESS_PUBLIC_URL")
            self.public_url = env_url.rstrip("/")
            logger.info("Using TOKENLESS_PUBLIC_URL for notebook endpoint.")
            return

        if progress_callback:
            progress_callback("Checking Kaggle credentials")
        creds = self._resolve_credentials()
        if not creds:
            raise RuntimeError(
                "Kaggle credentials missing. Set KAGGLE_USERNAME and KAGGLE_KEY "
                "(or ~/.kaggle/kaggle.json), or set TOKENLESS_PUBLIC_URL for a "
                "pre-provisioned endpoint."
            )

        owner, _key = creds
        self.public_url = None

        if self.model == GPT_OSS_MODEL_ID:
            if (kaggle_prompt is None and input_path is None) or pdf_context:
                self.public_url = self._start_ollama_gpt_oss_20b_server(
                    owner,
                    file_path=input_path,
                    kernel_slug=DEFAULT_GPT_OSS_SERVER_KERNEL_SLUG,
                    status_timeout=gpt_oss_status_timeout,
                    poll_interval=gpt_oss_poll_interval,
                    kernel_session_timeout=gpt_oss_kernel_session_timeout,
                    accelerator=gpt_oss_accelerator,
                    progress_callback=progress_callback,
                )
                self.kernel_ref = f"{owner}/{DEFAULT_GPT_OSS_SERVER_KERNEL_SLUG}"
                logger.info("Kaggle GPT-OSS server ready (%s).", self.kernel_ref)
                return

            self.kernel_ref = f"{owner}/{DEFAULT_GPT_OSS_KERNEL_SLUG}"
            sys_p = (
                kaggle_system_prompt
                if kaggle_system_prompt is not None
                else "You are a helpful AI assistant."
            )
            self.smoke_test_message = self.run_ollama_gpt_oss_20b_prompt(
                kaggle_prompt or "",
                file_path=input_path,
                system_prompt=sys_p,
                kernel_slug=DEFAULT_GPT_OSS_KERNEL_SLUG,
                status_timeout=gpt_oss_status_timeout,
                poll_interval=gpt_oss_poll_interval,
                kernel_session_timeout=gpt_oss_kernel_session_timeout,
                accelerator=gpt_oss_accelerator,
            )
            logger.info("Kaggle GPT-OSS script kernel finished (%s).", self.kernel_ref)
            return

        self.smoke_test_message = self._run_smoke_script_kernel(
            owner,
            smoke_status_timeout=smoke_status_timeout,
            smoke_poll_interval=smoke_poll_interval,
            smoke_kernel_session_timeout=smoke_kernel_session_timeout,
        )
        self.kernel_ref = f"{owner}/{self.kernel_slug}"
        logger.info("Kaggle smoke kernel finished (%s).", self.kernel_ref)

    def run_ollama_gpt_oss_20b_prompt(
        self,
        prompt: str,
        *,
        file_path: Optional[Path | str] = None,
        system_prompt: str = "You are a helpful AI assistant.",
        kernel_slug: str = DEFAULT_GPT_OSS_KERNEL_SLUG,
        status_timeout: int = 36_000,
        poll_interval: float = 5.0,
        kernel_session_timeout: int = 36_000,
        accelerator: Optional[str] = "NvidiaTeslaT4",
    ) -> str:
        """Batch one-shot: script kernel like ``gpt-oss-20B.ipynb``; returns assistant text."""
        input_path = self._validate_pdf_file_path(file_path) if file_path else None
        creds = self._resolve_credentials()
        if not creds:
            raise RuntimeError(
                "Kaggle credentials missing. Set KAGGLE_USERNAME and KAGGLE_KEY "
                "(or ~/.kaggle/kaggle.json)."
            )
        owner, _key = creds
        if not _GPT_OSS_TEMPLATE.is_file():
            raise FileNotFoundError(f"Missing GPT-OSS kernel template: {_GPT_OSS_TEMPLATE}")

        pb = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
        sb = base64.b64encode(system_prompt.encode("utf-8")).decode("ascii")
        body = _GPT_OSS_TEMPLATE.read_text(encoding="utf-8")
        body = body.replace("__TOKENLESS_PROMPT_B64__", pb)
        body = body.replace("__TOKENLESS_SYSTEM_B64__", sb)
        kernel_id = f"{owner}/{kernel_slug}"
        dataset_ref: Optional[str] = None
        dataset_slug: Optional[str] = None
        input_filename = ""
        with tempfile.TemporaryDirectory(prefix="tokenless-gpt-oss-") as tmp:
            KaggleApi = _kaggle_api_cls()
            folder = Path(tmp)
            api = self._configure_api()
            if input_path:
                dataset_ref, dataset_slug, input_filename = self._upload_private_input_dataset(
                    api,
                    owner,
                    input_path,
                )
                body = body.replace(
                    "__TOKENLESS_INPUT_DATASET_SLUG_B64__",
                    base64.b64encode(dataset_slug.encode("utf-8")).decode("ascii"),
                )
            else:
                body = body.replace("__TOKENLESS_INPUT_DATASET_SLUG_B64__", _empty_b64())
            body = body.replace(
                "__TOKENLESS_INPUT_FILENAME_B64__",
                base64.b64encode(input_filename.encode("utf-8")).decode("ascii")
                if input_filename
                else _empty_b64(),
            )

            (folder / _GPT_OSS_SCRIPT_NAME).write_text(body, encoding="utf-8")
            meta = {
                "id": kernel_id,
                "title": "Tokenless gpt oss 20b",
                "code_file": _GPT_OSS_SCRIPT_NAME,
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [dataset_ref] if dataset_ref else [],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }
            (folder / KaggleApi.KERNEL_METADATA_FILE).write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )

            logger.info("Pushing GPT-OSS 20B script kernel %s ...", kernel_id)
            try:
                return self._push_poll_fetch_output(
                    api,
                    kernel_id,
                    folder,
                    output_filename=GPT_OSS_RESPONSE_FILENAME,
                    error_filename=GPT_OSS_ERROR_FILENAME,
                    output_file_pattern=r"^tokenless_model_(response|error)\.txt$",
                    status_timeout=status_timeout,
                    poll_interval=poll_interval,
                    kernel_session_timeout=kernel_session_timeout,
                    accelerator=accelerator,
                )
            finally:
                if dataset_slug:
                    try:
                        api.dataset_delete(owner, dataset_slug, no_confirm=True)
                    except Exception as e:  # noqa: BLE001 - best-effort cleanup
                        logger.warning(
                            "Failed to delete temporary Kaggle dataset %s/%s: %s",
                            owner,
                            dataset_slug,
                            e,
                        )

    @staticmethod
    def _validate_pdf_file_path(file_path: Path | str) -> Path:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Input path must be a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported by file_path right now.")
        return path

    def _upload_private_input_dataset(
        self,
        api: Any,
        owner: str,
        file_path: Path,
    ) -> tuple[str, str, str]:
        dataset_slug = f"tokenless-pdf-{uuid.uuid4().hex[:12]}"
        dataset_ref = f"{owner}/{dataset_slug}"
        with tempfile.TemporaryDirectory(prefix="tokenless-input-dataset-") as tmp:
            folder = Path(tmp)
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file_path.name).strip("._")
            if not safe_name:
                safe_name = "input.pdf"
            if not safe_name.lower().endswith(".pdf"):
                safe_name = f"{safe_name}.pdf"
            shutil.copy2(file_path, folder / safe_name)
            meta = {
                "id": dataset_ref,
                "title": f"Tokenless PDF {dataset_slug[-12:]}",
                "licenses": [{"name": "CC0-1.0"}],
            }
            (folder / api.DATASET_METADATA_FILE).write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "Uploading PDF input as temporary private Kaggle dataset %s ...",
                dataset_ref,
            )
            created = api.dataset_create_new(
                str(folder),
                public=False,
                quiet=True,
                convert_to_csv=False,
                dir_mode="skip",
            )
            if created.error:
                raise RuntimeError(f"Kaggle dataset upload failed: {created.error}")
        dataset_ref = self._normalize_dataset_ref(created.ref or dataset_ref)
        _dataset_source_ref, mounted_name = self._wait_for_dataset_available(
            api,
            dataset_ref,
            expected_filename=safe_name,
        )
        return dataset_ref, dataset_slug, mounted_name

    @staticmethod
    def _normalize_dataset_ref(ref: str) -> str:
        ref = ref.strip()
        ref = ref.removeprefix("https://www.kaggle.com/datasets/")
        ref = ref.removeprefix("http://www.kaggle.com/datasets/")
        ref = ref.removeprefix("https://kaggle.com/datasets/")
        ref = ref.removeprefix("http://kaggle.com/datasets/")
        ref = ref.removeprefix("/datasets/")
        ref = ref.strip("/")
        parts = ref.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return ref

    @staticmethod
    def _wait_for_dataset_available(
        api: Any,
        dataset_ref: str,
        *,
        expected_filename: str,
        timeout: int = 300,
        poll_interval: float = 5.0,
    ) -> tuple[str, str]:
        def list_file_names(ref: str) -> list[str]:
            names: list[str] = []
            page_token = None
            while True:
                response = api.dataset_list_files(
                    ref,
                    page_token=page_token,
                    page_size=100,
                )
                files = getattr(response, "dataset_files", None)
                if files is None:
                    files = getattr(response, "files", [])
                names.extend(str(getattr(file, "name", "")) for file in files)
                page_token = getattr(response, "next_page_token", None)
                if not page_token:
                    break
            return [name for name in names if name]

        deadline = time.time() + timeout
        last_status = "unknown"
        last_error = ""
        last_files: list[str] = []
        while time.time() < deadline:
            dataset_source_ref = dataset_ref
            try:
                raw = api.dataset_status(
                    dataset_ref,
                    format="json(status,current_version_number)",
                )
                status = json.loads(raw)
            except Exception as e:  # noqa: BLE001 - dataset may not be visible yet
                last_status = repr(e)
                last_error = repr(e)
                time.sleep(poll_interval)
                continue

            last_status = str(status.get("status") or "unknown")
            version = status.get("current_version_number")
            if version is not None:
                dataset_source_ref = f"{dataset_ref}/{version}"
            if last_status in {"error", "failed", "failure"}:
                raise RuntimeError(
                    f"Kaggle dataset {dataset_ref} reported status {last_status}."
                )
            try:
                last_files = list_file_names(dataset_source_ref)
            except Exception as e:  # noqa: BLE001 - files may lag behind status
                last_error = repr(e)
                time.sleep(poll_interval)
                continue
            if expected_filename in last_files:
                return dataset_source_ref, expected_filename
            pdf_files = [name for name in last_files if name.lower().endswith(".pdf")]
            if len(pdf_files) == 1:
                return dataset_source_ref, pdf_files[0]
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Timed out waiting for Kaggle dataset {dataset_ref} to expose "
            f"{expected_filename!r}. Last status: {last_status}. "
            f"Last files: {last_files}. Last error: {last_error}"
        )

    def stop(self) -> None:
        if self._temporary_dataset_owner and self._temporary_dataset_slug:
            try:
                self._configure_api().dataset_delete(
                    self._temporary_dataset_owner,
                    self._temporary_dataset_slug,
                    no_confirm=True,
                )
                logger.info("Deleted Kaggle dataset %s.", self._temporary_dataset_slug)
            except Exception as e:  # noqa: BLE001 - best-effort cleanup
                logger.warning(
                    "Failed to delete Kaggle dataset %s/%s: %s",
                    self._temporary_dataset_owner,
                    self._temporary_dataset_slug,
                    e,
                )
        if self.kernel_ref and self.model == GPT_OSS_MODEL_ID and self.public_url:
            try:
                self._configure_api().kernels_delete(self.kernel_ref, no_confirm=True)
                logger.info("Deleted Kaggle kernel %s.", self.kernel_ref)
            except Exception as e:  # noqa: BLE001 - best-effort cleanup
                logger.warning("Failed to delete Kaggle kernel %s: %s", self.kernel_ref, e)
        if self.__class__._current is self:
            self.__class__._current = None
        self.public_url = None
        self.smoke_test_message = None
        self.kernel_ref = None
        self._temporary_dataset_owner = None
        self._temporary_dataset_slug = None

    def _start_ollama_gpt_oss_20b_server(
        self,
        owner: str,
        *,
        file_path: Optional[Path] = None,
        kernel_slug: str,
        status_timeout: int,
        poll_interval: float,
        kernel_session_timeout: int,
        accelerator: Optional[str],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Start a long-running Kaggle script kernel and return its public Ollama URL."""
        if not _GPT_OSS_SERVER_TEMPLATE.is_file():
            raise FileNotFoundError(f"Missing GPT-OSS server template: {_GPT_OSS_SERVER_TEMPLATE}")

        if progress_callback:
            progress_callback("Opening startup progress channel")
        kernel_id = f"{owner}/{kernel_slug}"
        rendezvous_topic = f"tokenless-{uuid.uuid4().hex}"
        public_url_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        listener = threading.Thread(
            target=self._listen_for_public_url,
            args=(rendezvous_topic, public_url_queue, status_timeout, progress_callback),
            daemon=True,
        )
        listener.start()

        with tempfile.TemporaryDirectory(prefix="tokenless-gpt-oss-server-") as tmp:
            KaggleApi = _kaggle_api_cls()
            KernelWorkerStatus = _kernel_worker_status_cls()
            folder = Path(tmp)
            body = _GPT_OSS_SERVER_TEMPLATE.read_text(encoding="utf-8")
            body = body.replace("__TOKENLESS_NTFY_TOPIC__", rendezvous_topic)
            dataset_ref: Optional[str] = None
            dataset_slug: Optional[str] = None
            input_filename = ""
            api = self._configure_api()
            if file_path:
                dataset_ref, dataset_slug, input_filename = self._upload_private_input_dataset(
                    api,
                    owner,
                    file_path,
                )
                self._temporary_dataset_owner = owner
                self._temporary_dataset_slug = dataset_slug
                body = body.replace(
                    "__TOKENLESS_INPUT_DATASET_SLUG_B64__",
                    base64.b64encode(dataset_slug.encode("utf-8")).decode("ascii"),
                )
            else:
                body = body.replace("__TOKENLESS_INPUT_DATASET_SLUG_B64__", _empty_b64())
            body = body.replace(
                "__TOKENLESS_INPUT_FILENAME_B64__",
                base64.b64encode(input_filename.encode("utf-8")).decode("ascii")
                if input_filename
                else _empty_b64(),
            )
            (folder / _GPT_OSS_SERVER_SCRIPT_NAME).write_text(body, encoding="utf-8")
            meta = {
                "id": kernel_id,
                "title": "Tokenless gpt oss 20b server",
                "code_file": _GPT_OSS_SERVER_SCRIPT_NAME,
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [dataset_ref] if dataset_ref else [],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }
            (folder / KaggleApi.KERNEL_METADATA_FILE).write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )

            logger.info("Pushing GPT-OSS 20B server kernel %s ...", kernel_id)
            if progress_callback:
                progress_callback("Uploading Kaggle server kernel")
            push = api.kernels_push(
                str(folder),
                timeout=str(kernel_session_timeout),
                acc=accelerator,
            )
            if push.error:
                raise RuntimeError(f"Kaggle kernel push failed: {push.error}")
            if push.url:
                logger.info("Kernel push accepted; progress: %s", push.url)
            if progress_callback:
                progress_callback("Waiting for Kaggle GPU startup")

            deadline = time.time() + status_timeout
            public_url_pattern = re.compile(
                r"TOKENLESS_PUBLIC_URL=(https://[-a-zA-Z0-9.]+\.trycloudflare\.com)"
            )
            while time.time() < deadline:
                try:
                    url = public_url_queue.get_nowait()
                    if progress_callback:
                        progress_callback("Public endpoint received")
                    return url
                except queue.Empty:
                    pass

                status = api.kernels_status(kernel_id)
                st = status.status
                if progress_callback and st != KernelWorkerStatus.RUNNING:
                    progress_callback(f"Kaggle status: {st.name.lower()}")
                if st == KernelWorkerStatus.ERROR:
                    msg = status.failure_message or "unknown error"
                    raise RuntimeError(f"Kaggle kernel run failed: {msg}")
                if st == KernelWorkerStatus.CANCEL_REQUESTED:
                    raise RuntimeError("Kaggle kernel run was cancelled (cancel requested).")
                if st == KernelWorkerStatus.CANCEL_ACKNOWLEDGED:
                    raise RuntimeError("Kaggle kernel run was cancelled.")
                if st == KernelWorkerStatus.COMPLETE:
                    raise RuntimeError(
                        "Kaggle GPT-OSS server kernel completed before exposing a public URL."
                    )

                logs = api.kernels_logs(kernel_id) or ""
                if not isinstance(logs, str):
                    logs = json.dumps(logs)
                match = public_url_pattern.search(logs)
                if match:
                    return match.group(1).rstrip("/")

                time.sleep(poll_interval)

        raise TimeoutError(
            f"Timed out after {status_timeout}s waiting for kernel {kernel_id} public URL."
        )

    @staticmethod
    def _listen_for_public_url(
        topic: str,
        public_url_queue: "queue.Queue[str]",
        timeout: int,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Listen for the one-time URL message published by the Kaggle script."""
        pattern = re.compile(r"https://[-a-zA-Z0-9.]+\.trycloudflare\.com")
        try:
            with requests.get(
                f"https://ntfy.sh/{topic}/json",
                stream=True,
                timeout=(10, timeout),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if public_url_queue.full():
                        return
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") != "message":
                        continue
                    message = str(event.get("message", ""))
                    match = pattern.search(message)
                    if match:
                        public_url_queue.put(match.group(0).rstrip("/"))
                        return
                    if progress_callback and message:
                        progress_callback(message)
        except requests.RequestException as e:
            logger.warning("Public URL rendezvous listener failed: %s", e)

    def _configure_api(self) -> Any:
        KaggleApi = _kaggle_api_cls()
        api = KaggleApi()
        cfg: dict[str, str] = {}
        cfg = api.read_config_environment(cfg)
        cfg = api.read_config_file(cfg, quiet=True)
        if self.username:
            cfg[KaggleApi.CONFIG_NAME_USER] = self.username
        if self.key:
            cfg[KaggleApi.CONFIG_NAME_KEY] = self.key
        user = cfg.get(KaggleApi.CONFIG_NAME_USER)
        secret = cfg.get(KaggleApi.CONFIG_NAME_KEY)
        if not user or not secret:
            raise RuntimeError(
                "Kaggle credentials missing. Set KAGGLE_USERNAME and KAGGLE_KEY "
                "(or ~/.kaggle/kaggle.json)."
            )
        api.config_values = cfg
        return api

    def _resolve_credentials(self) -> Optional[tuple[str, str]]:
        KaggleApi = _kaggle_api_cls()
        api = KaggleApi()
        cfg: dict[str, str] = {}
        cfg = api.read_config_environment(cfg)
        cfg = api.read_config_file(cfg, quiet=True)
        if self.username:
            cfg[KaggleApi.CONFIG_NAME_USER] = self.username
        if self.key:
            cfg[KaggleApi.CONFIG_NAME_KEY] = self.key
        user = cfg.get(KaggleApi.CONFIG_NAME_USER)
        secret = cfg.get(KaggleApi.CONFIG_NAME_KEY)
        if user and secret:
            return user, secret
        return None

    def _push_poll_fetch_output(
        self,
        api: Any,
        kernel_id: str,
        folder: Path,
        *,
        output_filename: str,
        error_filename: Optional[str],
        output_file_pattern: str,
        status_timeout: int,
        poll_interval: float,
        kernel_session_timeout: int,
        accelerator: Optional[str],
    ) -> str:
        def fetch_kernel_output() -> tuple[dict[str, Path], str]:
            out_dir = folder / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            files, token = api.kernels_output(
                kernel_id,
                str(out_dir),
                file_pattern=output_file_pattern,
                force=True,
                quiet=True,
            )
            return {Path(p).name: Path(p) for p in files}, token

        def raise_from_kernel_output(prefix: str) -> None:
            try:
                by_name, _token = fetch_kernel_output()
            except Exception as e:  # noqa: BLE001 - include original kernel status too
                raise RuntimeError(f"{prefix}; failed to fetch kernel output: {e}") from e
            if error_filename and error_filename in by_name:
                err_text = by_name[error_filename].read_text(encoding="utf-8").strip()
                raise RuntimeError(f"{prefix}; kernel reported failure: {err_text}")
            output_names = ", ".join(sorted(by_name)) or "none"
            raise RuntimeError(f"{prefix}; available output files: {output_names}")

        KernelWorkerStatus = _kernel_worker_status_cls()
        push = api.kernels_push(
            str(folder),
            timeout=str(kernel_session_timeout),
            acc=accelerator,
        )
        if push.error:
            raise RuntimeError(f"Kaggle kernel push failed: {push.error}")
        if push.url:
            logger.info("Kernel push accepted; progress: %s", push.url)

        deadline = time.time() + status_timeout
        while time.time() < deadline:
            status = api.kernels_status(kernel_id)
            st = status.status
            if st == KernelWorkerStatus.ERROR:
                msg = status.failure_message or "unknown error"
                raise_from_kernel_output(f"Kaggle kernel run failed: {msg}")
            if st == KernelWorkerStatus.CANCEL_REQUESTED:
                raise RuntimeError("Kaggle kernel run was cancelled (cancel requested).")
            if st == KernelWorkerStatus.CANCEL_ACKNOWLEDGED:
                raise RuntimeError("Kaggle kernel run was cancelled.")
            if st == KernelWorkerStatus.COMPLETE:
                break
            time.sleep(poll_interval)
        else:
            raise TimeoutError(
                f"Timed out after {status_timeout}s waiting for kernel {kernel_id} to complete."
            )

        by_name, _token = fetch_kernel_output()
        if output_filename in by_name:
            return by_name[output_filename].read_text(encoding="utf-8").strip()
        if error_filename and error_filename in by_name:
            err_text = by_name[error_filename].read_text(encoding="utf-8").strip()
            raise RuntimeError(f"Kernel reported failure (see {error_filename}): {err_text}")
        logs = api.kernels_logs(kernel_id) or ""
        if not isinstance(logs, str):
            logs = json.dumps(logs)
        log_tail = "\n".join(logs.splitlines()[-80:])
        output_names = ", ".join(sorted(by_name)) or "none"
        raise RuntimeError(
            f"Expected output file {output_filename!r} not found in kernel output for "
            f"{kernel_id}. Available output files: {output_names}. Recent Kaggle logs:\n{log_tail}"
        )

    def _run_smoke_script_kernel(
        self,
        owner: str,
        *,
        smoke_status_timeout: int,
        smoke_poll_interval: float,
        smoke_kernel_session_timeout: int,
    ) -> str:
        api = self._configure_api()

        template = _smoke_script_template()
        if not template.is_file():
            raise FileNotFoundError(f"Missing smoke kernel script: {template}")

        kernel_id = f"{owner}/{self.kernel_slug}"
        with tempfile.TemporaryDirectory(prefix="tokenless-kernel-") as tmp:
            KaggleApi = _kaggle_api_cls()
            folder = Path(tmp)
            shutil.copy2(template, folder / _SMOKE_SCRIPT_NAME)
            meta = {
                "id": kernel_id,
                "title": "Tokenless smoke",
                "code_file": _SMOKE_SCRIPT_NAME,
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }
            (folder / KaggleApi.KERNEL_METADATA_FILE).write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )

            logger.info("Pushing smoke script kernel %s ...", kernel_id)
            smoke_pat = "^" + re.escape(SMOKE_OUTPUT_FILENAME) + "$"
            return self._push_poll_fetch_output(
                api,
                kernel_id,
                folder,
                output_filename=SMOKE_OUTPUT_FILENAME,
                error_filename=None,
                output_file_pattern=smoke_pat,
                status_timeout=smoke_status_timeout,
                poll_interval=smoke_poll_interval,
                kernel_session_timeout=smoke_kernel_session_timeout,
                accelerator=None,
            )

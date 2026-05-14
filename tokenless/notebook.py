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
from typing import Callable, ClassVar, Optional

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus

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


def _smoke_script_template() -> Path:
    return Path(__file__).resolve().parent / "kernels" / "smoke" / _SMOKE_SCRIPT_NAME


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

    def launch(
        self,
        *,
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
        ``gpt-oss:20b`` batch runner. Passing ``kaggle_prompt`` keeps the legacy
        one-shot GPT-OSS behavior and stores the reply in ``smoke_test_message``.
        """
        self.__class__._current = self
        self.smoke_test_message = None
        self.kernel_ref = None
        if progress_callback:
            progress_callback("Checking endpoint configuration")

        env_url = os.environ.get("TOKENLESS_PUBLIC_URL")
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
            if kaggle_prompt is None:
                self.public_url = self._start_ollama_gpt_oss_20b_server(
                    owner,
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
                kaggle_prompt,
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
        system_prompt: str = "You are a helpful AI assistant.",
        kernel_slug: str = DEFAULT_GPT_OSS_KERNEL_SLUG,
        status_timeout: int = 36_000,
        poll_interval: float = 5.0,
        kernel_session_timeout: int = 36_000,
        accelerator: Optional[str] = "NvidiaTeslaT4",
    ) -> str:
        """Batch one-shot: script kernel like ``gpt-oss-20B.ipynb``; returns assistant text."""
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
        with tempfile.TemporaryDirectory(prefix="tokenless-gpt-oss-") as tmp:
            folder = Path(tmp)
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
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }
            (folder / KaggleApi.KERNEL_METADATA_FILE).write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )

            api = self._configure_api()
            logger.info("Pushing GPT-OSS 20B script kernel %s ...", kernel_id)
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

    def stop(self) -> None:
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

    def _start_ollama_gpt_oss_20b_server(
        self,
        owner: str,
        *,
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
            folder = Path(tmp)
            body = _GPT_OSS_SERVER_TEMPLATE.read_text(encoding="utf-8")
            body = body.replace("__TOKENLESS_NTFY_TOPIC__", rendezvous_topic)
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
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }
            (folder / KaggleApi.KERNEL_METADATA_FILE).write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )

            api = self._configure_api()
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

    def _configure_api(self) -> KaggleApi:
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
        api: KaggleApi,
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
                raise RuntimeError(f"Kaggle kernel run failed: {msg}")
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

        out_dir = folder / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        files, _token = api.kernels_output(
            kernel_id,
            str(out_dir),
            file_pattern=output_file_pattern,
            force=True,
            quiet=True,
        )
        by_name = {Path(p).name: Path(p) for p in files}
        if output_filename in by_name:
            return by_name[output_filename].read_text(encoding="utf-8").strip()
        if error_filename and error_filename in by_name:
            err_text = by_name[error_filename].read_text(encoding="utf-8").strip()
            raise RuntimeError(f"Kernel reported failure (see {error_filename}): {err_text}")
        raise RuntimeError(
            f"Expected output file {output_filename!r} not found in kernel output for {kernel_id}."
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

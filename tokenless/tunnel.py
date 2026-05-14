"""Wait for the public URL of the notebook-side tunnel (e.g. ngrok)."""

from __future__ import annotations

import logging
import time
from typing import Optional

from tokenless.notebook import KaggleNotebookManager

logger = logging.getLogger(__name__)


class TunnelManager:
    def __init__(self):
        self._url: Optional[str] = None

    def get_url(self, timeout: int = 300) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            mgr = KaggleNotebookManager._current
            if mgr and mgr.public_url:
                self._url = mgr.public_url.rstrip("/")
                return self._url
            time.sleep(0.25)
        raise RuntimeError(
            f"Timed out after {timeout}s waiting for a public notebook URL."
        )

    def close(self) -> None:
        self._url = None

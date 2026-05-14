"""Kaggle script kernel: writes a fixed message for tokenless smoke tests."""

from pathlib import Path

MESSAGE = "tokenless:kaggle_smoke_ok"
Path("/kaggle/working/tokenless_smoke_message.txt").write_text(MESSAGE, encoding="utf-8")

import os

import pytest

from tokenless import TokenlessLLM


@pytest.mark.live_kaggle
def test_live_gpt_oss_start_and_send():
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        pytest.skip("KAGGLE_USERNAME and KAGGLE_KEY are required")

    llm = TokenlessLLM(model="gpt-oss:20b")
    try:
        llm.start(show_progress=False)
        reply = llm.send("Say 'tokenless live ok' and nothing else.")
        assert "tokenless" in reply.lower()
    finally:
        llm.stop()

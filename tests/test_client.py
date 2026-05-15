import pytest

from tokenless import GPT_OSS_MODEL_ID, TokenlessLLM


def test_gpt_oss_model_is_supported():
    llm = TokenlessLLM(model=GPT_OSS_MODEL_ID)

    assert llm.model == "gpt-oss:20b"


def test_send_requires_start():
    llm = TokenlessLLM(model=GPT_OSS_MODEL_ID)

    with pytest.raises(RuntimeError, match=r"Call \.start\(\) first"):
        llm.send("hello")


def test_agents_model_requires_running_endpoint():
    llm = TokenlessLLM(model=GPT_OSS_MODEL_ID)

    with pytest.raises(RuntimeError, match="Call .start"):
        llm.as_agents_model()

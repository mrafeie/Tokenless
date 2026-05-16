import pytest

from tokenless import GPT_OSS_MODEL_ID, TokenlessLLM


class DummyNotebook:
    public_url = None
    smoke_test_message = None

    def launch(self, **_kwargs):
        raise AssertionError("unit tests must not launch Kaggle")

    def stop(self):
        return None


def make_llm():
    llm = TokenlessLLM(model=GPT_OSS_MODEL_ID)
    llm._notebook = DummyNotebook()
    return llm


def test_gpt_oss_model_is_supported():
    llm = make_llm()

    assert llm.model == "gpt-oss:20b"


def test_send_requires_start():
    llm = make_llm()

    with pytest.raises(RuntimeError, match=r"Call \.start\(\) first"):
        llm.send("hello")


def test_agents_model_requires_running_endpoint():
    llm = make_llm()

    with pytest.raises(RuntimeError, match="Call .start"):
        llm.as_agents_model()


def test_strands_model_requires_running_endpoint():
    llm = make_llm()

    with pytest.raises(RuntimeError, match="Call .start"):
        llm.as_strands_model()


def test_langchain_model_requires_running_endpoint():
    llm = make_llm()

    with pytest.raises(RuntimeError, match="Call .start"):
        llm.as_langchain_llm()


def test_start_with_public_url_does_not_launch_kaggle():
    llm = make_llm()

    url = llm.start(public_url="https://example.trycloudflare.com", show_progress=False)

    assert url == "https://example.trycloudflare.com"
    assert llm.base_url == "https://example.trycloudflare.com"

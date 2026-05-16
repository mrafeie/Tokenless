import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-live-kaggle",
        action="store_true",
        default=False,
        help="run tests that require real Kaggle credentials and may start GPU kernels",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live-kaggle"):
        return

    skip_live = pytest.mark.skip(reason="requires --run-live-kaggle and Kaggle credentials")
    for item in items:
        if "live_kaggle" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def no_kaggle_credentials(monkeypatch, request):
    """Keep default CI/unit tests independent from the runner's Kaggle config."""
    if request.config.getoption("--run-live-kaggle"):
        return
    if os.environ.get("TOKENLESS_ALLOW_KAGGLE_CREDENTIALS") == "1":
        return

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("TOKENLESS_PUBLIC_URL", raising=False)

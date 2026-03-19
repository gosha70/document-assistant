"""Shared fixtures for Playwright e2e tests.

The app is expected to already be running at BASE_URL before the tests start.
Start it with:
    DOT_CONFIG_PATH=config/local.yaml poetry run uvicorn src.api.main:app --reload
"""

import pytest

BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL

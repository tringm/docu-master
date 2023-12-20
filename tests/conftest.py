from logging import Logger

import pytest
from fastapi.testclient import TestClient
from httpx import Client

from src.app import app
from src.logging import get_logger


@pytest.fixture(scope="session")
def client() -> Client:
    return TestClient(app=app)


@pytest.fixture(scope="session")
def logger() -> Logger:
    return get_logger(name="test")

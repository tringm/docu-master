from logging import Logger

import pytest
from fastapi.testclient import TestClient
from httpx import Client

from src.app import app
from src.logging import get_logger

RUN_EVAL_TESTS_FLAG = "--run-eval"


def pytest_addoption(parser) -> None:  # type: ignore
    parser.addoption(RUN_EVAL_TESTS_FLAG, action="store_true", help="run evaluation tests")


def pytest_collection_modifyitems(session, config, items) -> None:  # type: ignore
    run_eval_tests = config.getoption(RUN_EVAL_TESTS_FLAG)
    skip_eval_tests = pytest.mark.skip(reason=f"require {RUN_EVAL_TESTS_FLAG} to run")

    for item in items:
        if "evaluation" in item.keywords and not run_eval_tests:
            item.add_marker(skip_eval_tests)


@pytest.fixture(scope="session")
def client() -> Client:
    return TestClient(app=app)


@pytest.fixture(scope="session")
def logger() -> Logger:
    return get_logger(name="test")

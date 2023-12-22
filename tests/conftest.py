from logging import Logger

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Client

from src.app import app, get_document_service
from src.docs import DocumentService
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
def document_service() -> DocumentService:
    return DocumentService(chromadb_in_memory=True)


@pytest.fixture(scope="session")
def application(document_service: DocumentService) -> FastAPI:
    app.dependency_overrides = {
        get_document_service: lambda: document_service,
    }
    return app


@pytest.fixture(scope="session")
def client(application: FastAPI) -> Client:
    return TestClient(app=application)


@pytest.fixture(scope="session")
def logger() -> Logger:
    return get_logger(name="test")

import os
from logging import Logger
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Client

from src.app import app, get_document_service, get_llm_service
from src.docs import DocumentService
from src.llm import LLMService
from src.logging import get_logger
from tests import TEST_DIR_PATH

TEST_OUTPUTS_DIR_PATH = TEST_DIR_PATH / "outputs"


class _FLAGS:
    run_evaluation_tests = "--run-eval"
    run_comparison = "--output-diff"


def pytest_addoption(parser) -> None:  # type: ignore
    parser.addoption(_FLAGS.run_evaluation_tests, action="store_true", help="run evaluation tests")
    parser.addoption(_FLAGS.run_comparison, action="store_true", help="show changes of test output file")


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    run_eval_tests = config.getoption(_FLAGS.run_evaluation_tests)

    skip_eval_tests = pytest.mark.skip(reason=f"require {_FLAGS.run_evaluation_tests} to run")

    for item in items:
        if "evaluation" in item.keywords and not run_eval_tests:
            item.add_marker(skip_eval_tests)


@pytest.fixture(scope="session")
def document_service() -> DocumentService:
    return DocumentService(chromadb_in_memory=True)


@pytest.fixture(scope="session")
def llm_service() -> LLMService:
    return LLMService()


@pytest.fixture(scope="session")
def application(document_service: DocumentService, llm_service: LLMService) -> FastAPI:
    app.dependency_overrides = {
        get_document_service: lambda: document_service,
        get_llm_service: lambda: llm_service,
    }
    return app


@pytest.fixture(scope="session")
def client(application: FastAPI) -> Client:
    return TestClient(app=application)


@pytest.fixture(scope="session")
def logger() -> Logger:
    return get_logger(name="test")


@pytest.fixture
def test_case_out_dir(request: pytest.FixtureRequest) -> Path:
    test_case_path = request.path.parent / request.path.stem
    return TEST_OUTPUTS_DIR_PATH.joinpath(test_case_path.relative_to(TEST_DIR_PATH))


@pytest.fixture
def test_case_out_file(request: pytest.FixtureRequest, test_case_out_dir: Path) -> Path:
    test_case_out_dir.mkdir(parents=True, exist_ok=True)
    return test_case_out_dir / f"{request.node.name}_out.txt"


@pytest.fixture(autouse=True)
def run_comparison(
    request: pytest.FixtureRequest,
    test_case_out_file: Path,
) -> None:
    comp_enabled = request.config.getoption(_FLAGS.run_comparison)
    if comp_enabled and test_case_out_file.exists():
        request.addfinalizer(finalizer=lambda: os.system(f"git difftool {test_case_out_file}"))  # noqa: S605

import os
from logging import Logger
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Client

from src.app import app, get_llm_service, get_vector_store
from src.llm import LLMService
from src.logging import get_logger
from src.vector_store import VectorStore
from tests import TEST_DIR_PATH

TEST_OUTPUTS_DIR_PATH = TEST_DIR_PATH / "outputs"


class _FLAGS:
    run_evaluation_tests = "--run-eval"
    run_comparison = "--output-diff"
    integration_tests = "--integration-test"


def pytest_addoption(parser) -> None:  # type: ignore
    parser.addoption(_FLAGS.run_evaluation_tests, action="store_true", help="run evaluation tests")
    parser.addoption(_FLAGS.run_comparison, action="store_true", help="show changes of test output file")
    parser.addoption(_FLAGS.integration_tests, action="store_true", help="run integration tests")


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    run_eval_tests = config.getoption(_FLAGS.run_evaluation_tests)

    skip_eval_tests = pytest.mark.skip(reason=f"require {_FLAGS.run_evaluation_tests} to run")

    for item in items:
        if "evaluation" in item.keywords and not run_eval_tests:
            item.add_marker(skip_eval_tests)


@pytest.fixture(scope="session")
def vector_store(request: pytest.FixtureRequest) -> VectorStore:
    integration_mode_enabled = request.config.getoption(_FLAGS.integration_tests)
    if integration_mode_enabled:
        return VectorStore()
    return VectorStore(chromadb_in_memory=True)


@pytest.fixture(scope="session")
def llm_service() -> LLMService:
    return LLMService()


@pytest.fixture(scope="session")
def application(vector_store: VectorStore, llm_service: LLMService) -> FastAPI:
    app.dependency_overrides = {
        get_vector_store: lambda: vector_store,
        get_llm_service: lambda: llm_service,
    }
    return app


def _get_required_env_var(env_var: str) -> str:
    val = os.getenv(env_var)
    if not val:
        raise KeyError(f"Missing required env var `{env_var}`")
    return val


@pytest.fixture(scope="session")
def client(request: pytest.FixtureRequest, application: FastAPI) -> Client:
    e2e_mode_enabled = request.config.getoption(_FLAGS.integration_tests)
    if e2e_mode_enabled:
        return Client(base_url=_get_required_env_var(env_var="TEST_API_URL"), timeout=60.0)

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

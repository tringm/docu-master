import filecmp
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from httpx import Client, codes
from pytest_mock import MockFixture

from src.app import PATHS
from tests import TEST_DIR_PATH

from .data import EXAMPLE_PDF_FILE


@pytest.fixture
def mock_docs_dir(mocker: MockFixture) -> Iterator[Path]:
    outputs_dir = TEST_DIR_PATH / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    mocker.patch(target="src.app.DOCS_DIR_PATH", new=outputs_dir)
    yield outputs_dir
    shutil.rmtree(outputs_dir)


def test_upload_file(client: Client, mock_docs_dir: Path) -> None:
    response = client.post(url=PATHS.upload_file, files={"file": EXAMPLE_PDF_FILE.open(mode="rb")})
    assert response.status_code == codes.NO_CONTENT
    saved_fp = mock_docs_dir / EXAMPLE_PDF_FILE.name
    assert saved_fp.exists(), f"Expected uploaded file saved to {saved_fp}"
    assert filecmp.cmp(EXAMPLE_PDF_FILE, saved_fp), "Expected saved file to have the same content"

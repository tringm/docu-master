from collections.abc import Iterator
from pathlib import Path

import pytest
from httpx import Client, codes

from src.app import PATHS, UploadFileResponse
from src.errors import DocumentParsingError
from src.vector_store import VectorStore
from tests.test_docs_parsing import EXAMPLE_PDF_FILE, EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT


@pytest.fixture
def upload_example_file(client: Client, vector_store: VectorStore) -> Iterator[str]:
    response = client.post(url=PATHS.upload_file, files={"file": EXAMPLE_PDF_FILE.open(mode="rb")})
    assert response.status_code == codes.OK
    resp = UploadFileResponse.model_validate(response.json())
    doc_id = resp.document_id
    chunks = vector_store.get_chunk_by_document_id(document_id=doc_id)
    assert (
        len(chunks) == EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT
    ), f"Expected {EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT} chunk uploaded to vector store"
    yield doc_id
    vector_store.default_collection.delete(where={"document_id": {"$eq": doc_id}})


def test_upload_invalid_file(client: Client, tmpdir: Path) -> None:
    tmp_txt_file = tmpdir / "text_file.pdf"
    with tmp_txt_file.open(mode="w") as f:
        f.write("Some Content")
    response = client.post(url=PATHS.upload_file, files={"file": tmp_txt_file.open(mode="rb")})
    assert response.status_code == codes.INTERNAL_SERVER_ERROR
    assert DocumentParsingError.__name__ in response.text

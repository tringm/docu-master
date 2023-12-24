import re
from collections.abc import Iterator

import pytest
from httpx import Client, codes

from src.app import IDK_ANSWER, PATHS, QAResponse, UploadFileResponse
from src.docs import DocumentService
from tests.test_doc_service import EXAMPLE_PDF_FILE, EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT


@pytest.fixture
def upload_file(client: Client, document_service: DocumentService) -> Iterator[str]:
    response = client.post(url=PATHS.upload_file, files={"file": EXAMPLE_PDF_FILE.open(mode="rb")})
    assert response.status_code == codes.OK
    resp = UploadFileResponse.model_validate(response.json())
    doc_id = resp.document_id
    chunks = document_service.get_chunk_by_document_id(document_id=doc_id)
    assert (
        len(chunks) == EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT
    ), f"Expected {EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT} chunk uploaded to vector store"
    yield doc_id
    document_service.default_collection.delete(where={"document_id": {"$eq": doc_id}})


def test_qa_endpoint(client: Client, upload_file: Iterator[str]) -> None:
    resp = client.post(url=PATHS.qa, json={"question": "Is Cobra venomous?"})
    assert resp.status_code == codes.OK
    resp = QAResponse.model_validate(resp.json())
    assert resp.answer != IDK_ANSWER, "Expected return an answer"
    assert re.search(r"yes", resp.answer.lower()), "Expected yes in answer"
    assert len(resp.sources), "Expected answer with sources"


@pytest.mark.parametrize(
    "req_json",
    [
        {},
        {"questionnn": "Is Cobra venomous?"},
    ],
)
def test_qa_endpoint_bad_request(client: Client, req_json: dict) -> None:
    resp = client.post(url=PATHS.qa, json=req_json)
    assert resp.status_code == codes.BAD_REQUEST

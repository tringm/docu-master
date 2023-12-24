import re
from collections.abc import Iterator
from uuid import uuid4

import pytest
from httpx import Client, Response, codes

from src.app import IDK_ANSWER, PATHS, QARequest, QAResponse, UploadFileResponse
from src.docs import DocumentChunk, DocumentChunkMetadata, DocumentService
from tests.test_doc_service import EXAMPLE_PDF_FILE, EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT


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


def _assert_correct_cobra_qa_response(resp: Response) -> None:
    assert resp.status_code == codes.OK
    resp = QAResponse.model_validate(resp.json())
    assert resp.answer != IDK_ANSWER, "Expected return an answer"
    assert re.search(r"yes", resp.answer.lower()), "Expected yes in answer"
    assert len(resp.sources) > 0, "Expected answer with sources"


def test_qa_endpoint(client: Client, upload_file: str) -> None:
    resp = client.post(url=PATHS.qa, json=QARequest(question="Is Cobra venomous?").model_dump())
    _assert_correct_cobra_qa_response(resp=resp)


def test_qa_endpoint_filter_document(client: Client, document_service: DocumentService, upload_file: str) -> None:
    new_doc_id = str(uuid4())
    new_doc_chunk = DocumentChunk(
        id=str(uuid4()), text="Random text", metadata=DocumentChunkMetadata(page=0, document_id=new_doc_id)
    )
    document_service.add_multiple_document_chunks(chunks=[new_doc_chunk])

    resp = client.post(
        url=PATHS.qa, json=QARequest(question="Is Cobra venomous?", document_ids=[new_doc_id]).model_dump()
    )
    assert resp.status_code == codes.OK
    resp = QAResponse.model_validate(resp.json())
    assert resp.answer == IDK_ANSWER, "Expected return IDK"
    assert len(resp.sources) == 0, "Expected return no source"

    resp = client.post(
        url=PATHS.qa, json=QARequest(question="Is Cobra venomous?", document_ids=[upload_file]).model_dump()
    )
    _assert_correct_cobra_qa_response(resp=resp)

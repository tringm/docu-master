import re
from uuid import uuid4

import pytest
from httpx import Client, Response, codes

from src.app import IDK_ANSWER, PATHS, QARequest, QAResponse
from src.docs import DocumentChunk, DocumentChunkMetadata
from src.vector_store import VectorStore

# noinspection PyUnresolvedReferences
from .test_upload_endpoint import upload_example_file  # noqa: F401


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


def _assert_correct_cobra_qa_response(resp: Response) -> None:
    assert resp.status_code == codes.OK
    resp = QAResponse.model_validate(resp.json())
    assert resp.answer != IDK_ANSWER, "Expected return an answer"
    assert re.search(r"yes", resp.answer.lower()), "Expected yes in answer"
    assert len(resp.sources) > 0, "Expected answer with sources"


def test_qa_endpoint(client: Client, upload_example_file: str) -> None:  # noqa: F811
    resp = client.post(url=PATHS.qa, json=QARequest(question="Is Cobra venomous?").model_dump())
    _assert_correct_cobra_qa_response(resp=resp)


def test_qa_endpoint_filter_document(
    client: Client, vector_store: VectorStore, upload_example_file: str  # noqa: F811
) -> None:
    new_doc_id = str(uuid4())
    new_doc_chunk = DocumentChunk(
        id=str(uuid4()), text="Random text", metadata=DocumentChunkMetadata(page=0, document_id=new_doc_id)
    )
    vector_store.add_multiple_document_chunks(chunks=[new_doc_chunk])

    resp = client.post(
        url=PATHS.qa, json=QARequest(question="Is Cobra venomous?", document_ids=[new_doc_id]).model_dump()
    )
    assert resp.status_code == codes.OK
    resp = QAResponse.model_validate(resp.json())
    assert resp.answer == IDK_ANSWER, "Expected return IDK"
    assert len(resp.sources) == 0, "Expected return no source"

    resp = client.post(
        url=PATHS.qa, json=QARequest(question="Is Cobra venomous?", document_ids=[upload_example_file]).model_dump()
    )
    _assert_correct_cobra_qa_response(resp=resp)

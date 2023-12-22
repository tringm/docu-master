from collections.abc import Callable, Iterator
from uuid import uuid4

import pytest

from src.docs import DocumentChunk, DocumentService

from .data import RESOURCE_DIR_PATH, load_hotpot_qa_test_cases

EXAMPLE_PDF_FILE = RESOURCE_DIR_PATH / "cobra_wiki.pdf"
EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT = 5


def f1_score(predicted: list[str], gold_standard: list[str]) -> float:
    tp, fp, fn = 0, 0, 0
    for item in predicted:
        if item in gold_standard:
            tp += 1
        else:
            fp += 1
    for item in gold_standard:
        if item not in predicted:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0


@pytest.fixture
def create_collection(document_service: DocumentService) -> Iterator[Callable[[], str]]:
    created_cols = []

    def _create_collection() -> str:
        col_name = str(uuid4())
        created_cols.append(col_name)
        document_service.chromadb_client.create_collection(name=col_name)
        return col_name

    yield _create_collection

    for col in created_cols:
        document_service.chromadb_client.delete_collection(name=col)


@pytest.mark.evaluation
def test_document_retrieval(
    document_service: DocumentService,
    create_collection: Callable[[], str],
) -> None:
    test_cases = load_hotpot_qa_test_cases()

    f1_scores = []

    for case in test_cases:
        collection = create_collection()
        document_service.add_multiple_document_chunks(collection_name=collection, chunks=case.document_chunks)
        retrieval_res = document_service.search(collection_name=collection, query=case.question)

        retrieved_docs = [chunk[0].id for chunk in retrieval_res]
        expected_docs = [chunk.id for chunk in case.source_chunks]

        f1_scores.append(f1_score(predicted=retrieved_docs, gold_standard=expected_docs))

    avg_f1_score = sum(f1_scores) / len(f1_scores)
    min_f1_score = 0.39
    assert avg_f1_score > min_f1_score, f"Expected avg f1 score to be above {min_f1_score}"


def test_parse_pdf(document_service: DocumentService) -> None:
    chunks = document_service.parse_pdf_file(stream=EXAMPLE_PDF_FILE, doc_id=str(uuid4()))
    assert (
        len(chunks) == EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT
    ), f"Expected return {EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT} chunks"
    assert all(isinstance(chunk, DocumentChunk) and chunk.text for chunk in chunks)

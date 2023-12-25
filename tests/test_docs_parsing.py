from uuid import uuid4

from src.docs import DocumentChunk, parse_pdf_file
from tests import RESOURCE_DIR_PATH

EXAMPLE_PDF_FILE = RESOURCE_DIR_PATH / "cobra_wiki.pdf"
EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT = 11


def test_parse_pdf() -> None:
    chunks = parse_pdf_file(stream=EXAMPLE_PDF_FILE, doc_id=str(uuid4()))
    assert (
        len(chunks) == EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT
    ), f"Expected return {EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT} chunks"
    assert all(isinstance(chunk, DocumentChunk) and chunk.text for chunk in chunks)

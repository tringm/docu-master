from httpx import Client, codes

from src.app import PATHS, UploadFileResponse
from src.docs import DocumentService
from tests.test_doc_service import EXAMPLE_PDF_FILE, EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT


def test_upload_file(client: Client, document_service: DocumentService) -> None:
    response = client.post(url=PATHS.upload_file, files={"file": EXAMPLE_PDF_FILE.open(mode="rb")})
    assert response.status_code == codes.OK
    resp = UploadFileResponse.model_validate(response.json())
    chunks = document_service.get_chunk_by_document_id(document_id=resp.document_id)
    assert (
        len(chunks) == EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT
    ), f"Expected {EXAMPLE_PDF_FILE_EXPECTED_CHUNK_COUNT} chunk uploaded to vector store"

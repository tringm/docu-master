from collections.abc import Callable
from typing import Annotated
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from .config import CONFIGS
from .docs import DocumentService
from .logging import logger

app = FastAPI()


class PATHS:
    upload_file = "/upload/"


@app.middleware("http")
async def handling_exception(request: Request, call_next: Callable) -> Response:
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception(str(e))
        return JSONResponse(status_code=500, content={"error": e.__class__.__name__, "messages": e.args})


def get_document_service() -> DocumentService:
    return DocumentService()


@app.post(path=PATHS.upload_file)
async def create_upload_file(
    file: UploadFile, document_service: Annotated[DocumentService, Depends(get_document_service)]
) -> Response:
    doc_id = str(uuid4())
    doc_chunks = document_service.parse_pdf_file(stream=file.file, doc_id=doc_id)
    document_service.add_multiple_document_chunks(chunks=doc_chunks)
    return JSONResponse(status_code=200, content={"document_id": doc_id})


def main() -> None:
    uvicorn.run(app, **CONFIGS.uvicorn.model_dump())


if __name__ == "__main__":
    main()

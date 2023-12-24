from collections.abc import Callable
from typing import Annotated
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .config import CONFIGS
from .docs import DocumentService
from .llm import LLMService
from .logging import logger

app = FastAPI()

IDK_ANSWER = "Unfortunately, I cannot find the answer to the question."


class PATHS:
    upload_file = "/upload/"
    qa = "/qa/"


class UploadFileResponse(BaseModel):
    document_id: str


class QAResponse(BaseModel):
    answer: str
    sources: list[str]


class QARequest(BaseModel):
    question: str
    document_ids: list[str] | None = None


@app.middleware("http")
async def handling_exception(request: Request, call_next: Callable) -> Response:
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception(str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": f"{e.__class__.__name__}: {e.args}"}
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> Response:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": exc.errors()})


def get_document_service() -> DocumentService:
    return DocumentService()


def get_llm_service() -> LLMService:
    return LLMService()


@app.post(path=PATHS.upload_file)
async def create_upload_file(
    file: UploadFile, document_service: Annotated[DocumentService, Depends(get_document_service)]
) -> UploadFileResponse:
    doc_id = str(uuid4())
    doc_chunks = document_service.parse_pdf_file(stream=file.file, doc_id=doc_id)
    document_service.add_multiple_document_chunks(chunks=doc_chunks)
    return UploadFileResponse(document_id=doc_id)


@app.post(path=PATHS.qa)
async def qa(
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    req: QARequest,
) -> QAResponse:
    chunks = document_service.search(query=req.question, document_ids=req.document_ids)
    if not chunks:
        return QAResponse(answer=IDK_ANSWER, sources=[])
    chunks_texts = [chnk.text for chnk in chunks]
    answer = llm_service.answer_question_based_on_sources(
        question=req.question,
        sources=chunks_texts,
    )
    return QAResponse(answer=answer, sources=chunks_texts)


def main() -> None:
    uvicorn.run(app, **CONFIGS.uvicorn.model_dump())


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import IO, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from pypdf import PdfReader
from semantic_text_splitter import CharacterTextSplitter

from .logging import logger


class DocumentChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    page: int
    document_id: str


class DocumentChunk(BaseModel):
    id: str  # noqa: A003
    text: str
    metadata: DocumentChunkMetadata


def parse_pdf_file(
    stream: str | IO[Any] | Path, doc_id: str, chunk_capacity: int | tuple[int, int] = (256, 512)
) -> list[DocumentChunk]:
    logger.info("Parsing PDF stream")
    txt_splitter = CharacterTextSplitter(trim_chunks=True)

    doc_chunks = []

    reader = PdfReader(stream)
    doc_meta = reader.metadata

    for page_idx, page in enumerate(reader.pages):
        page_txt = page.extract_text()
        txt_chunks = txt_splitter.chunks(text=page_txt, chunk_capacity=chunk_capacity)
        doc_chunks += [
            DocumentChunk(
                id=str(uuid4()),
                text=txt,
                metadata=DocumentChunkMetadata(document_id=doc_id, page=page_idx, **doc_meta),
            )
            for txt in txt_chunks
        ]

    return doc_chunks

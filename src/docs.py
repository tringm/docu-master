from pathlib import Path
from typing import IO, Any

from pydantic import BaseModel, ConfigDict
from pypdf import PdfReader
from semantic_text_splitter import CharacterTextSplitter

from .config import CONFIGS
from .logging import logger


class DocumentChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    page: int
    document_id: str


class DocumentChunk(BaseModel):
    id: str  # noqa: A003
    text: str
    metadata: DocumentChunkMetadata


def _text_truncate(text: str, max_length: int = 100) -> str:
    return text if len(text) < max_length else f"{text[:100]}..."


def split_text(text: str) -> list[str]:
    txt_splitter = CharacterTextSplitter(trim_chunks=True)
    chunk_capacity = CONFIGS.docs.chunk_capacity
    try:
        return txt_splitter.chunks(text=text, chunk_capacity=chunk_capacity)  # type: ignore
    except Exception as e:
        logger.exception("Failed to split text %s into chunk(%s): %s", _text_truncate(text=text), chunk_capacity, e)
        raise DocumentParsingError("Failed to split text into chunks") from e


def _generate_chunk_id(doc_id: str, page_idx: int, chunk_idx: int) -> str:
    return f"{doc_id}_p{page_idx}_c{chunk_idx}"


def parse_pdf_file(stream: str | IO[Any] | Path, doc_id: str) -> list[DocumentChunk]:
    logger.info("Parsing PDF stream")

    doc_chunks = []
    try:
        reader = PdfReader(stream)
    except Exception as e:
        logger.exception("Failed to init PdfReader: %s", e)
        raise DocumentParsingError("Failed to parse PDF file") from e
    doc_meta = reader.metadata

    for page_idx, page in enumerate(reader.pages):
        try:
            page_txt = page.extract_text()
        except Exception as e:
            logger.exception("Failed to parse page %s: %s", page_idx, e)
            raise DocumentParsingError(f"Failed to parse page {page_idx}") from e
        doc_chunks += [
            DocumentChunk(
                id=_generate_chunk_id(doc_id=doc_id, page_idx=page_idx, chunk_idx=chunk_idx),
                text=chunk_txt,
                metadata=DocumentChunkMetadata(document_id=doc_id, page=page_idx, **doc_meta),
            )
            for chunk_idx, chunk_txt in enumerate(split_text(text=page_txt))
        ]

    return doc_chunks


def parse_text(text: str, doc_id: str) -> list[DocumentChunk]:
    logger.info("Parsing text")
    page_idx = 1
    return [
        DocumentChunk(
            id=_generate_chunk_id(doc_id=doc_id, page_idx=page_idx, chunk_idx=chunk_idx),
            text=chunk_txt,
            metadata=DocumentChunkMetadata(document_id=doc_id, page=page_idx),
        )
        for chunk_idx, chunk_txt in enumerate(split_text(text=text))
    ]


class DocumentParsingError(Exception):
    pass

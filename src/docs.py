from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any
from uuid import uuid4

from chromadb import Collection, EphemeralClient, HttpClient
from pydantic import BaseModel, ConfigDict
from pypdf import PdfReader
from semantic_text_splitter import CharacterTextSplitter

from .config import CONFIGS
from .logging import get_logger
from .singleton import ThreadUnsafeSingletonMeta


class DocumentChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    page: int
    document_id: str


class DocumentChunk(BaseModel):
    id: str  # noqa: A003
    text: str
    metadata: DocumentChunkMetadata


class DocumentService(metaclass=ThreadUnsafeSingletonMeta):
    DEFAULT_COLLECTION_NAME = "default"

    def __init__(self, chromadb_in_memory: bool = False) -> None:
        self.logger = get_logger(name=self.__class__.__name__)
        if chromadb_in_memory:
            self.chromadb_client = EphemeralClient(database=CONFIGS.chromadb.database)
        else:
            self.chromadb_client = HttpClient(database=CONFIGS.chromadb.database, **CONFIGS.chromadb.client_configs)
        self.distance_score_threshold = CONFIGS.chromadb.distance_score_threshold
        self.default_collection = self.chromadb_client.get_or_create_collection(name=self.DEFAULT_COLLECTION_NAME)

    def get_collection(self, name: str) -> Collection:
        try:
            return self.chromadb_client.get_collection(name=name)
        except Exception:
            self.logger.exception("Failed to retrieve collection %s", name)
            raise

    def add_multiple_document_chunks(
        self, chunks: Iterable[DocumentChunk], collection_name: str = DEFAULT_COLLECTION_NAME
    ) -> None:
        try:
            collection = self.get_collection(name=collection_name)
            ids, metas, docs = zip(
                *((chunk.id, chunk.metadata.model_dump(), chunk.text) for chunk in chunks), strict=True
            )
            collection.add(ids=list(ids), metadatas=list(metas), documents=list(docs))  # chroma check for type list
        except Exception:
            self.logger.exception("Failed to add chunks to collection %s", collection_name)
            raise

    def search(
        self,
        query: str,
        n_results: int = 10,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        document_ids: list[str] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        collection = self.get_collection(name=collection_name)
        where = None
        if document_ids:
            where = {"document_id": {"$in": document_ids}}

        try:
            res = collection.query(
                query_texts=query,
                n_results=n_results,
                where=where,
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            self.logger.exception("Failed to search with query %s", query)
            raise

        return [
            (DocumentChunk(id=c_id, text=text, metadata=DocumentChunkMetadata(**meta)), score)
            for c_id, text, meta, score in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
            )
            if score < self.distance_score_threshold
        ]

    def get_chunk_by_document_id(
        self, document_id: str, collection_name: str = DEFAULT_COLLECTION_NAME, **kwargs: Any
    ) -> list[DocumentChunk]:
        collection = self.get_collection(name=collection_name)

        res = collection.get(where={"document_id": {"$eq": document_id}}, include=["metadatas", "documents"], **kwargs)

        return [
            DocumentChunk(id=c_id, text=text, metadata=DocumentChunkMetadata(**meta))
            for c_id, text, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]

    def parse_pdf_file(
        self, stream: str | IO[Any] | Path, doc_id: str, chunk_capacity: int | tuple[int, int] = (700, 1000)
    ) -> list[DocumentChunk]:
        self.logger.info("Parsing PDF stream")
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

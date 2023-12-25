from collections.abc import Iterable
from typing import Any

from chromadb import Collection, EphemeralClient, HttpClient, Settings

from .config import CONFIGS
from .docs import DocumentChunk, DocumentChunkMetadata
from .logging import get_logger
from .singleton import ThreadUnsafeSingletonMeta


class VectorStore(metaclass=ThreadUnsafeSingletonMeta):
    DEFAULT_COLLECTION_NAME = "default"

    def __init__(self, chromadb_in_memory: bool = False) -> None:
        self.logger = get_logger(name=self.__class__.__name__)
        chromadb_client_settings = Settings(anonymized_telemetry=False)
        if chromadb_in_memory:
            self.chromadb_client = EphemeralClient(
                database=CONFIGS.chromadb.database, settings=chromadb_client_settings
            )
        else:
            self.chromadb_client = HttpClient(
                database=CONFIGS.chromadb.database, settings=chromadb_client_settings, **CONFIGS.chromadb.client_configs
            )
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
        n_results: int = 3,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        document_ids: list[str] | None = None,
    ) -> list[DocumentChunk]:
        collection = self.get_collection(name=collection_name)
        where = None
        if document_ids:
            where = {"document_id": {"$in": document_ids}}

        log_msg = f"Query colection({collection_name}) with '{query}'"
        if where:
            log_msg += f" {where}"
        try:
            res = collection.query(
                query_texts=query,
                n_results=n_results,
                where=where,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as e:
            self.logger.exception("Failed to %s: %s", log_msg, e)
            raise

        ret_chunks = [
            DocumentChunk(id=c_id, text=text, metadata=DocumentChunkMetadata(**meta))
            for c_id, text, meta, score in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
            )
            if score < self.distance_score_threshold
        ]
        self.logger.info("%s got %s results", log_msg, len(ret_chunks))
        return ret_chunks

    def get_chunk_by_document_id(
        self, document_id: str, collection_name: str = DEFAULT_COLLECTION_NAME, **kwargs: Any
    ) -> list[DocumentChunk]:
        collection = self.get_collection(name=collection_name)

        res = collection.get(where={"document_id": {"$eq": document_id}}, include=["metadatas", "documents"], **kwargs)

        return [
            DocumentChunk(id=c_id, text=text, metadata=DocumentChunkMetadata(**meta))
            for c_id, text, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]

import typing
from collections.abc import Iterable

from chromadb import Collection, EphemeralClient, HttpClient
from pydantic import BaseModel, ConfigDict

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


class ChromaDB(metaclass=ThreadUnsafeSingletonMeta):
    def __init__(self, in_memory: bool = False) -> None:
        self.logger = get_logger(name=self.__class__.__name__)
        if in_memory:
            self.client = EphemeralClient(database=CONFIGS.chromadb.database)
        else:
            self.client = HttpClient(database=CONFIGS.chromadb.database, **CONFIGS.chromadb.client_configs)
        self.distance_score_threshold = CONFIGS.chromadb.distance_score_threshold

    def add_multiple_document_chunks(self, collection: Collection, chunks: Iterable[DocumentChunk]) -> None:
        try:
            ids, metas, docs = zip(
                *((chunk.id, chunk.metadata.model_dump(), chunk.text) for chunk in chunks), strict=True
            )
            collection.add(ids=list(ids), metadatas=list(metas), documents=list(docs))  # chroma check for type list
        except Exception:
            self.logger.exception("Failed to add chunks", extra={"collection": collection})
            raise

    def search(
        self,
        collection: Collection,
        query: str,
        n_results: int = 10,
    ) -> list[tuple[DocumentChunk, typing.Any]]:
        res = collection.query(
            query_texts=query,
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )

        return [
            (DocumentChunk(id=c_id, text=text, metadata=DocumentChunkMetadata(**meta)), score)
            for c_id, text, meta, score in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
            )
            if score < self.distance_score_threshold
        ]

import json
from uuid import uuid4

from pydantic import BaseModel, Field

from src.docs import DocumentChunk, DocumentChunkMetadata
from tests import RESOURCE_DIR_PATH

EVALUATION_DATA_DIR_PATH = RESOURCE_DIR_PATH / "evaluation"
HOTPOT_QA_FILE_PATH = EVALUATION_DATA_DIR_PATH / "hotpot_qa.json"


class RetrievalTestCase(BaseModel):
    question: str
    answer: str
    document_chunks: list[DocumentChunk]
    source_chunks: list[DocumentChunk]

    def sources_as_str(self) -> str:
        return "\n".join(f"- {chunk.text}" for chunk in self.source_chunks)


class HotpotQATestCase(BaseModel):
    class Context(BaseModel):
        title: list[str]
        sentences: list[list[str]]

    class SupportingFact(BaseModel):
        title: list[str]
        sentence_id: list[int] = Field(alias="sent_id")

    question: str
    answer: str
    context: Context
    supporting_facts: SupportingFact

    def as_retrieval_test_case(self) -> RetrievalTestCase:
        source_title_sentence_idx = list(
            zip(self.supporting_facts.title, self.supporting_facts.sentence_id, strict=True)
        )
        source_chunks = []
        doc_chunks = []

        for title, sentences in zip(self.context.title, self.context.sentences, strict=True):
            doc_id = str(uuid4())
            for sent_idx, sent in enumerate(sentences):
                chunk_meta = DocumentChunkMetadata(page=1, document_id=doc_id)
                chunk = DocumentChunk(id=str(uuid4()), text=sent, metadata=chunk_meta)
                doc_chunks.append(chunk)

                if (title, sent_idx) in source_title_sentence_idx:
                    source_chunks.append(chunk)

        return RetrievalTestCase(
            question=self.question, answer=self.answer, document_chunks=doc_chunks, source_chunks=source_chunks
        )


def load_hotpot_qa_test_cases() -> list[RetrievalTestCase]:
    with HOTPOT_QA_FILE_PATH.open() as f:
        data = json.load(f)
    return [HotpotQATestCase.model_validate(datum).as_retrieval_test_case() for datum in data]

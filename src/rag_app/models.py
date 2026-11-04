from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: str
    filename: str
    page_number: int
    text: str
    score: float


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]


class EvaluationRequest(BaseModel):
    question: str = Field(min_length=3)
    answer: str = Field(min_length=1)
    document_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class EvaluationResponse(BaseModel):
    retrieval_recall: float
    answer_relevance: float
    citation_coverage: float
    groundedness: float
    notes: list[str]

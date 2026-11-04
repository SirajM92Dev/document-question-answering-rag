from pathlib import Path
from uuid import uuid4

from .chunking import chunk_pages
from .config import Settings
from .embeddings import Embedder
from .llm import LLMError, OpenAIAnswerGenerator
from .models import AnswerResponse, Citation, UploadResponse
from .pdf import extract_pdf_pages
from .vector_store import VectorStore


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._store: VectorStore | None = None
        self._generator: OpenAIAnswerGenerator | None = None

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(
                self.settings.chroma_persist_directory,
                Embedder(self.settings.embedding_model),
            )
        return self._store

    @property
    def generator(self) -> OpenAIAnswerGenerator:
        if self._generator is None:
            self._generator = OpenAIAnswerGenerator(
                self.settings.openai_api_key, self.settings.openai_chat_model
            )
        return self._generator

    def ingest(self, path: Path, original_filename: str) -> UploadResponse:
        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF documents are supported")
        pages = extract_pdf_pages(path)
        if not pages:
            raise ValueError("The PDF contains no extractable text")
        document_id = str(uuid4())
        chunks = chunk_pages(
            pages, document_id, original_filename, self.settings.chunk_size, self.settings.chunk_overlap
        )
        self.store.add(chunks)
        return UploadResponse(
            document_id=document_id,
            filename=original_filename,
            pages=len(pages),
            chunks=len(chunks),
        )

    def ask(self, question: str, document_id: str | None = None, top_k: int | None = None) -> AnswerResponse:
        citations = self.store.search(question, top_k or self.settings.top_k, document_id)
        if not citations:
            return AnswerResponse(answer="No relevant document passages were found.", citations=[])
        try:
            answer = self.generator.answer(question, citations)
        except LLMError:
            raise
        return AnswerResponse(answer=answer, citations=citations)

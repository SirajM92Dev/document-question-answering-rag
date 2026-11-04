from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import get_settings
from .evaluation import evaluate_answer
from .llm import LLMError
from .models import AnswerResponse, EvaluationRequest, EvaluationResponse, QuestionRequest, UploadResponse
from .pipeline import RAGPipeline

app = FastAPI(title="Document QA RAG", version="1.0.0")
pipeline = RAGPipeline(get_settings())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only application/pdf files are supported")
    settings = get_settings()
    size_limit = settings.max_upload_mb * 1024 * 1024
    with NamedTemporaryFile(dir=settings.upload_directory, suffix=".pdf", delete=False) as temporary:
        total = 0
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > size_limit:
                Path(temporary.name).unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="The uploaded file is too large")
            temporary.write(chunk)
        path = Path(temporary.name)
    try:
        return pipeline.ingest(path, file.filename or "document.pdf")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/questions", response_model=AnswerResponse)
def ask_question(request: QuestionRequest) -> AnswerResponse:
    try:
        return pipeline.ask(request.question, request.document_id, request.top_k)
    except (LLMError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    citations = pipeline.store.search(
        request.question, request.top_k or get_settings().top_k, request.document_id
    )
    return evaluate_answer(request.question, request.answer, citations)

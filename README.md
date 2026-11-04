# Document Question Answering Using RAG

A production-oriented Retrieval-Augmented Generation (RAG) application for uploading PDF documents, indexing their contents, retrieving semantically relevant passages, generating grounded LLM answers, and evaluating response quality.

## Architecture

1. **PDF extraction** — `pypdf` extracts text while retaining page numbers.
2. **Chunking** — page text is split into overlapping word windows.
3. **Embeddings** — Sentence Transformers creates normalized semantic vectors.
4. **Vector search** — ChromaDB persists vectors locally and returns cosine-ranked passages.
5. **Generation** — OpenAI generates answers only from retrieved context and includes citations.
6. **Evaluation** — retrieval coverage, lexical relevance, citation coverage, and context overlap are exposed through `/evaluate`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Set OPENAI_API_KEY in .env
uvicorn rag_app.api:app --reload
```

Open `http://localhost:8000/docs` for the interactive API documentation.

## API

- `POST /documents` — multipart PDF upload; returns a document ID.
- `POST /questions` — ask a question with optional `document_id` and `top_k`.
- `POST /evaluate` — evaluate an answer against retrieved passages.
- `GET /health` — service health check.

Example:

```bash
curl -F "file=@report.pdf" http://localhost:8000/documents
curl -X POST http://localhost:8000/questions \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main findings?","document_id":"<id>"}'
```

## CLI

```bash
rag ingest report.pdf
rag ask "What are the main findings?" --document-id <id>
```

## Docker

```bash
cp .env.example .env
docker compose up --build
```

## Development

```bash
ruff check .
pytest --cov=rag_app --cov-report=term-missing
```

The default storage is local ChromaDB under `data/chroma`; uploaded files are processed temporarily and are not retained.
Professional Retrieval-Augmented Generation application for document question answering with PDF extraction, semantic retrieval, LLM answers, and evaluation.

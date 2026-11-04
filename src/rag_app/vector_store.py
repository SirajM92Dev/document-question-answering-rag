from pathlib import Path

import chromadb

from .models import Citation, DocumentChunk


class VectorStore:
    def __init__(self, persist_directory: Path, embedder):
        client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = embedder

    def add(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=self.embedder.encode([chunk.text for chunk in chunks]),
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                }
                for chunk in chunks
            ],
        )

    def search(self, query: str, top_k: int, document_id: str | None = None) -> list[Citation]:
        kwargs = {
            "query_embeddings": self.embedder.encode([query]),
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if document_id:
            kwargs["where"] = {"document_id": document_id}
        result = self.collection.query(**kwargs)
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            Citation(
                chunk_id=chunk_id,
                filename=metadata["filename"],
                page_number=int(metadata["page_number"]),
                text=document,
                score=max(0.0, 1.0 - float(distance)),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

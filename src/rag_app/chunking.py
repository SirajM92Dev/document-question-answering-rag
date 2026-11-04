import re
from uuid import uuid4

from .models import DocumentChunk


def chunk_pages(
    pages: list[tuple[int, str]],
    document_id: str,
    filename: str,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    chunks: list[DocumentChunk] = []
    for page_number, text in pages:
        words = re.findall(r"\S+", text)
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid4()),
                        document_id=document_id,
                        filename=filename,
                        page_number=page_number,
                        text=chunk_text,
                    )
                )
            if end == len(words):
                break
            start = end - overlap
    return chunks

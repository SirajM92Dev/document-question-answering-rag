from pathlib import Path

from pypdf import PdfReader


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Extract non-empty text from a PDF while retaining one-based page numbers."""
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages

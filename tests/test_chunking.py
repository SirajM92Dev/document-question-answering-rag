import pytest

from rag_app.chunking import chunk_pages


def test_chunking_preserves_page_and_overlap():
    chunks = chunk_pages(
        [(2, "one two three four five six")],
        "doc",
        "file.pdf",
        chunk_size=4,
        overlap=1,
    )
    assert len(chunks) == 2
    assert chunks[0].page_number == 2
    assert chunks[1].text.startswith("four")


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_pages([(1, "text")], "doc", "file.pdf", 4, 4)

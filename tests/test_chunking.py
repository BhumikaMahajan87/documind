import pytest

from app.core.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_single_chunk():
    chunks = chunk_text("This is a short document. It has two sentences.", chunk_size=600)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert "short document" in chunks[0].text


def test_long_text_is_split_into_multiple_chunks():
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * 60  # well over the chunk budget
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(c.index == i for i, c in enumerate(chunks))
    # Every chunk should respect the budget within a reasonable margin.
    assert all(len(c.text) <= 200 + 50 for c in chunks)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        chunk_text("hi", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("hi", chunk_size=100, overlap=100)

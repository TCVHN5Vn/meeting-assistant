"""
Tests for the chunker.

Chunking is the one piece of this pipeline that is pure logic: no model, no
network, no database, deterministic output. That makes it the piece that is
actually worth unit testing, and the tests run in milliseconds.

Testing the embedding or the LLM at this level would mean asserting things
about model behaviour, which is a different discipline (evaluation, on a
dataset, with metrics) and does not belong in a unit test suite.
"""

import pytest

from app.rag.chunking import chunk_text


def test_short_text_is_a_single_chunk():
    text = "One short sentence."
    assert chunk_text(text, chunk_size=1000, overlap=200) == [text]


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_long_text_is_split():
    text = "word " * 1000  # ~5000 characters
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) > 1


def test_chunks_respect_the_size_limit():
    text = "The quick brown fox jumps over the lazy dog. " * 200
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    # Every chunk must fit the budget. A chunk that silently overflows is
    # how you end up truncated by the model's context window at runtime.
    assert all(len(c) <= 500 for c in chunks)


def test_consecutive_chunks_overlap():
    """The whole point of overlap: no passage falls between two chunks."""
    text = "".join(f"sentence number {i}. " for i in range(200))
    chunks = chunk_text(text, chunk_size=400, overlap=100)

    # The tail of one chunk should reappear at the head of the next.
    first_tail = chunks[0][-50:]
    assert first_tail in chunks[1], (
        "chunk 1 does not contain the end of chunk 0, so a passage spanning "
        "that boundary exists in neither chunk intact"
    )


def test_full_text_is_covered():
    """No content is dropped between chunks."""
    text = "".join(f"item {i} content here. " for i in range(300))
    chunks = chunk_text(text, chunk_size=600, overlap=120)

    # Strip whitespace differences and confirm every chunk's content is
    # genuinely from the source, and that the last chunk reaches the end.
    assert all(c in text for c in chunks)
    assert text.strip().endswith(chunks[-1][-30:])


def test_overlap_must_be_smaller_than_chunk_size():
    """A guard against an infinite loop, not a style preference.

    If overlap >= chunk_size the window steps backwards or stays still, and
    the function never terminates. Failing loudly at the call is far better
    than a hung ingest with no error message.
    """
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=100, overlap=100)


def test_prefers_breaking_at_sentence_boundaries():
    """Chunks should end at a natural break, not mid-word."""
    text = ("This is a complete sentence. " * 60)
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    for chunk in chunks[:-1]:  # the last one just ends where the text ends
        assert chunk.rstrip().endswith("."), f"chunk cut mid-sentence: ...{chunk[-40:]!r}"

"""
Splitting long text into retrievable pieces.

WHY CHUNK AT ALL?

Two independent reasons, and it is worth being able to state both:

1. Retrieval precision. An embedding is a single fixed-size vector -- 384
   numbers, in our case -- no matter whether you embed a sentence or a
   forty-page handbook. Embedding the whole handbook averages every topic
   in it into one blurry point that is vaguely near everything and
   strongly near nothing. Chunking gives each distinct idea its own vector,
   so a query about holiday policy can actually match the holiday section.

2. Context budget. The retrieved text has to fit in the LLM's context
   window alongside the question and the instructions. Smaller units mean
   you can afford several of them.

WHY OVERLAP?

Because chunk boundaries are arbitrary and will land mid-thought. If a
policy's condition ends up at the tail of chunk 4 and its consequence at
the head of chunk 5, neither chunk answers the question on its own. Having
each chunk repeat the last ~200 characters of the previous one means any
short passage survives intact inside at least one chunk. You pay for it in
storage and in near-duplicate search results.

THE TRADEOFF, IN ONE LINE:
smaller chunks -> sharper retrieval, more risk of losing context;
larger chunks  -> more context per hit, blurrier vectors, fewer hits fit.
1000/200 is a sane default, not a law. Tuning it is a legitimate
experiment, and "how would you choose your chunk size?" is a fair
interview question -- the honest answer is: build an evaluation set of
real questions and measure retrieval, do not guess.
"""

import re

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

# Prefer to cut at a paragraph break; failing that, the end of a sentence;
# failing that, a space. Ordered most to least desirable.
_BOUNDARIES = [re.compile(p) for p in (r"\n\s*\n", r"(?<=[.!?])\s", r"\s")]


def _find_break(text: str, lo: int, hi: int) -> int:
    """Best place to cut inside text[lo:hi], or hi if there is none.

    Searches for the LAST match of the best-available boundary type, so a
    chunk runs as close to full length as it can while still ending
    somewhere natural.
    """
    window = text[lo:hi]
    for pattern in _BOUNDARIES:
        matches = list(pattern.finditer(window))
        if matches:
            return lo + matches[-1].end()
    return hi


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks that end on natural boundaries.

    Sizes are in characters. Tokens would be more accurate -- models think
    in tokens, and the ratio varies by language -- but characters need no
    tokenizer and are close enough at this scale. English averages roughly
    4 characters per token, so ~1000 characters is ~250 tokens.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size, or the "
                         "window never advances and this loops forever")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        hard_end = start + chunk_size

        if hard_end >= len(text):
            end = len(text)
        else:
            # Only look for a boundary in the last quarter of the window.
            # Searching the whole window would happily cut at the first
            # sentence and emit a 40-character chunk.
            end = _find_break(text, start + (chunk_size * 3 // 4), hard_end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Step forward by chunk_size minus the overlap, so the next chunk
        # re-reads the tail of this one. max(..., 1) is a guard: if a
        # boundary search ever returned something pathological, we still
        # advance rather than spinning on the same position.
        start = max(end - overlap, start + 1)

    return chunks

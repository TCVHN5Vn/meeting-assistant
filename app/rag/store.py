"""
The FAISS vector index: the thing that answers "which vectors are nearest
to this one?".

WHY IS THIS A SEPARATE FILE FROM THE DATABASE?

Because SQLite cannot do this. Finding nearest neighbours means comparing a
query vector against every stored vector -- arithmetic over a large matrix
of floats. A B-tree index, which is what a relational database gives you,
sorts values on a line; it has no notion of "close" in 384 dimensions.

So the system ends up with two stores, and one join key between them:

    SQLite                     FAISS
    ------                     -----
    rag_chunks.text       <--  positions 0, 1, 2, ...
    rag_chunks.vector_id

The Postgres+pgvector option in the architecture doc collapses these into
one system, which removes the sync problem below entirely. That is the main
argument for it, and worth saying out loud in an interview: "I used FAISS
plus SQLite and had to keep them consistent manually; pgvector would have
given me transactional consistency for free."

WHY IndexFlatIP AND NOT IndexFlatL2?

IP is inner product (dot product); L2 is Euclidean distance. For vectors
that have been normalised to length 1 -- which ours are, see embeddings.py
-- the dot product IS the cosine of the angle between them: a similarity
in [-1, 1] where 1 means identical direction. So IndexFlatIP over
normalised vectors gives cosine similarity, and cosine is what you want
for text, because it compares direction (what the text is about) and
ignores magnitude (roughly, how emphatic or long it is).

The "Flat" part means exhaustive: it compares the query against every
single vector, so results are exactly correct, with no approximation. That
is O(n) per search, which is completely fine for thousands or even
hundreds of thousands of chunks. Only at millions do you move to an
approximate index (IVF, HNSW) and start trading a little recall for speed.
Reaching for the approximate index first is a classic premature
optimisation.
"""

from pathlib import Path

import numpy as np

from app.config import EMBEDDING_DIM, FAISS_INDEX_PATH, ensure_dirs


def build_index(vectors: np.ndarray):
    """Create a fresh index containing exactly these vectors.

    Row i of `vectors` becomes position i in the index -- that position is
    the `vector_id` we store in SQLite.

    We always rebuild from scratch rather than appending, and that is a
    deliberate simplification. IndexFlat has no real delete: removing a
    vector shifts every later position down by one, so every vector_id
    after the deleted one now points at the wrong text, and searches start
    returning confidently-wrong answers with no error anywhere. Rebuilding
    the whole index whenever the corpus changes makes that class of bug
    impossible. It costs a few seconds at our scale. When rebuilding gets
    too expensive, the real fix is IndexIDMap with stable explicit ids,
    not clever bookkeeping.
    """
    import faiss

    if vectors.shape[0] and vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"expected {EMBEDDING_DIM}-dim vectors, got {vectors.shape[1]}. "
            "Did the embedding model change without EMBEDDING_DIM being updated?"
        )

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    if vectors.shape[0]:
        index.add(vectors)
    return index


def save_index(index, path: Path = FAISS_INDEX_PATH) -> None:
    import faiss

    ensure_dirs()
    faiss.write_index(index, str(path))


def load_index(path: Path = FAISS_INDEX_PATH):
    """Read the index back off disk, or None if it was never built."""
    import faiss

    if not Path(path).exists():
        return None
    return faiss.read_index(str(path))


def search(index, query_vector: np.ndarray, top_k: int = 5):
    """Return (scores, vector_ids) for the top_k nearest vectors.

    FAISS hands back two parallel arrays. `ids` are POSITIONS in the index,
    not anything meaningful on their own -- app/rag/retrieve.py is what
    turns them back into text via the rag_chunks table.

    A vector_id of -1 means "no result in this slot", which happens when
    you ask for more neighbours than the index contains. Callers must skip
    those; treating -1 as a real id would look up a row that does not exist.
    """
    if index is None or index.ntotal == 0:
        return np.zeros((1, 0), dtype="float32"), np.zeros((1, 0), dtype="int64")

    scores, ids = index.search(query_vector, min(top_k, index.ntotal))
    return scores, ids

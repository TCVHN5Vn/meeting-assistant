"""
The R in RAG: question -> the most relevant chunks of text.

This is the half that people skip past on the way to the LLM, and it is the
half that decides whether the answer is any good. The generator can only
work with what retrieval hands it: retrieve the wrong passages and a better
model just writes a more fluent wrong answer.
"""

from dataclasses import dataclass

from app.db import init_db
from app.rag import embeddings, store


@dataclass
class Hit:
    """One retrieved chunk, with where it came from and how well it matched."""
    text: str
    score: float          # cosine similarity, 1.0 = identical direction
    document_title: str
    document_path: str
    chunk_index: int


def retrieve(query: str, top_k: int = 5, min_score: float = 0.0) -> list[Hit]:
    """Find the chunks most semantically similar to `query`.

    min_score exists because vector search ALWAYS returns its top_k, no
    matter how bad the matches are -- ask an HR-handbook corpus about
    submarines and it will still hand back five confident-looking chunks
    with low scores. Passing that to an LLM invites a hallucinated answer
    built out of irrelevant context. A floor lets the pipeline say "I don't
    have anything on that" instead.

    What counts as a good score depends on the embedding model, so calibrate
    it against your own data rather than copying a number. With MiniLM,
    roughly: >0.5 is a solid match, 0.3-0.5 is loosely related, below 0.3 is
    usually noise.
    """
    index = store.load_index()
    if index is None:
        raise RuntimeError(
            "No FAISS index found. Run: python -m scripts.ingest_documents"
        )

    query_vector = embeddings.embed_query(query)
    scores, vector_ids = store.search(index, query_vector, top_k=top_k)

    # scores/ids come back shaped (n_queries, top_k); we sent one query.
    scores, vector_ids = scores[0], vector_ids[0]

    # -1 means "empty slot" -- fewer vectors in the index than we asked for.
    wanted = [
        (int(vid), float(score))
        for vid, score in zip(vector_ids, scores)
        if vid != -1 and score >= min_score
    ]
    if not wanted:
        return []

    conn = init_db()
    try:
        # One query for all the ids rather than one query per id. The
        # placeholders have to be built dynamically because SQLite has no
        # "IN (list)" parameter type -- note they are still bound "?"
        # placeholders, not interpolated values, so this is not an
        # injection risk.
        placeholders = ",".join("?" for _ in wanted)
        rows = conn.execute(
            f"""SELECT dc.vector_id, dc.text, dc.chunk_index,
                       d.title AS document_title, d.path AS document_path
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE dc.vector_id IN ({placeholders})""",
            [vid for vid, _ in wanted],
        ).fetchall()
    finally:
        conn.close()

    # SQL gives no guarantee about row order, and we need FAISS's ranking
    # preserved -- most similar first. So index the rows and walk `wanted`.
    by_vector_id = {row["vector_id"]: row for row in rows}

    hits = []
    for vid, score in wanted:
        row = by_vector_id.get(vid)
        if row is None:
            # FAISS knows about a vector SQLite has no row for: the two
            # stores have drifted apart. Skip rather than crash, but this
            # means it is time to re-run ingest with force=True.
            continue
        hits.append(Hit(
            text=row["text"],
            score=score,
            document_title=row["document_title"],
            document_path=row["document_path"],
            chunk_index=row["chunk_index"],
        ))
    return hits

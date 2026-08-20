"""
The R in RAG: question -> the most relevant chunks, from documents and
meeting transcripts alike.

This is the half people skip past on the way to the LLM, and it is the half
that decides whether the answer is any good. The generator can only work
with what retrieval hands it: retrieve the wrong passages and a better model
just writes a more fluent wrong answer.
"""

from dataclasses import dataclass

from app.db import init_db
from app.rag import embeddings, store
from app.rag.transcripts import format_timestamp

# When a filter is applied we ask FAISS for more neighbours than we need,
# because it cannot filter itself. See the note in retrieve() below.
OVERFETCH_FACTOR = 8
MAX_OVERFETCH = 200


@dataclass
class Hit:
    """One retrieved chunk: the text, how well it matched, and where it came from."""
    text: str
    score: float              # cosine similarity, 1.0 = identical direction
    source_type: str          # 'document' or 'transcript'
    source_id: str
    source_title: str
    chunk_index: int
    start_ts: float | None    # None for documents
    end_ts: float | None

    @property
    def citation(self) -> str:
        """How this chunk should be referred to in an answer.

        The two source types genuinely need different citations. Pointing at
        "chunk 7" of a meeting is useless to a person; pointing at 41:18 of
        the recording lets them go and listen to it and judge for
        themselves. That verifiability is most of the value of citing at
        all -- a citation nobody can follow is decoration.
        """
        if self.source_type == "transcript" and self.start_ts is not None:
            return (f"{self.source_title} @ {format_timestamp(self.start_ts)}"
                    f"-{format_timestamp(self.end_ts)}")
        return self.source_title


def retrieve(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    source_type: str | None = None,
    meeting_id: str | None = None,
) -> list[Hit]:
    """Find the chunks most semantically similar to `query`.

    `min_score` matters because vector search ALWAYS returns its top_k, no
    matter how bad the matches are -- ask this corpus about submarines and
    it will still hand back chunks with low scores. Passing those to an LLM
    invites a hallucinated answer built from irrelevant context. A floor
    lets the pipeline say "I don't have anything on that" instead.

    What counts as a good score depends on the embedding model, so calibrate
    against your own data rather than copying a number. With MiniLM here,
    roughly: >0.5 solid, 0.3-0.5 loosely related, below 0.3 noise.

    FILTERING, AND WHY IT IS AWKWARD

    `source_type` and `meeting_id` restrict the search -- "only the written
    policies", or "only last Tuesday's meeting". FAISS IndexFlat stores
    nothing but raw vectors: it has no metadata and cannot filter. So the
    filter has to be applied AFTER the search, in SQL, which means the
    search can hand back k results that the filter then throws away, leaving
    fewer than k.

    The fix here is to over-fetch -- ask for k * OVERFETCH_FACTOR and filter
    that down. It is a heuristic, not a guarantee: a filter matching 1% of
    the corpus can still starve. When it does, the honest options are a
    separate index per partition, or a vector store with real filtered
    search built in (Qdrant, Weaviate, pgvector with a WHERE clause). This
    is one of the clearest practical arguments for a purpose-built vector
    database over a bare index, and worth being able to state as a limit of
    this design rather than pretending it away.
    """
    index = store.load_index()
    if index is None:
        raise RuntimeError(
            "No index found. Run: python -m scripts.ingest_documents"
        )

    filtered = source_type is not None or meeting_id is not None
    fetch_k = min(top_k * OVERFETCH_FACTOR, MAX_OVERFETCH) if filtered else top_k

    query_vector = embeddings.embed_query(query)
    scores, vector_ids = store.search(index, query_vector, top_k=fetch_k)

    # Shaped (n_queries, k); we sent one query.
    scores, vector_ids = scores[0], vector_ids[0]

    # -1 means "empty slot" -- fewer vectors in the index than we asked for.
    wanted = [
        (int(vid), float(score))
        for vid, score in zip(vector_ids, scores)
        if vid != -1 and score >= min_score
    ]
    if not wanted:
        return []

    conditions = ["vector_id IN ({})".format(",".join("?" for _ in wanted))]
    params: list = [vid for vid, _ in wanted]
    if source_type is not None:
        conditions.append("source_type = ?")
        params.append(source_type)
    if meeting_id is not None:
        conditions.append("source_type = 'transcript' AND source_id = ?")
        params.append(meeting_id)

    conn = init_db()
    try:
        # One query for all ids rather than one per id. The placeholders are
        # built dynamically because SQL has no "IN (list)" parameter type --
        # note they are still bound "?" placeholders, so this is not an
        # injection risk.
        rows = conn.execute(
            f"""SELECT vector_id, text, chunk_index, source_type, source_id,
                       source_title, start_ts, end_ts
                FROM rag_chunks WHERE {' AND '.join(conditions)}""",
            params,
        ).fetchall()
    finally:
        conn.close()

    # SQL guarantees nothing about row order, and FAISS's ranking is exactly
    # what must be preserved -- most similar first. So index the rows and
    # walk `wanted`, which is already in rank order.
    by_vector_id = {row["vector_id"]: row for row in rows}

    hits = []
    for vid, score in wanted:
        row = by_vector_id.get(vid)
        if row is None:
            # Either the filter excluded it, or -- if unfiltered -- FAISS
            # knows about a vector SQLite has no row for, meaning the two
            # stores have drifted. /api/v1/index/stats reports the latter.
            continue
        hits.append(Hit(
            text=row["text"],
            score=score,
            source_type=row["source_type"],
            source_id=row["source_id"],
            source_title=row["source_title"],
            chunk_index=row["chunk_index"],
            start_ts=row["start_ts"],
            end_ts=row["end_ts"],
        ))
        if len(hits) == top_k:
            break
    return hits

"""
Building the FAISS index over everything retrievable.

This is deliberately the ONLY place that writes vector_ids, and it always
rewrites all of them. Documents and transcripts both write their text into
rag_chunks and then call rebuild_index once, at the end.

Why one shared pass rather than each source indexing itself: embedding is
the expensive step, and batching every chunk through the model together is
far faster than several smaller passes. More importantly, vector_ids are
POSITIONS in a single shared index -- they only make sense assigned all at
once, from one ordered list.
"""

from app.config import EMBEDDING_DIM
from app.db import init_db
from app.rag import embeddings, store


def rebuild_index(conn) -> int:
    """Re-embed every chunk in rag_chunks and rebuild the FAISS index.

    Assigning vector_ids here, in one deterministic pass over an explicit
    ORDER BY, is what keeps the two stores consistent: position i in the
    index and the row with vector_id = i are the same chunk by construction,
    because both come from this single ordered list.

    The ORDER BY needs a tiebreaker that is unique (id) -- without one,
    SQLite is free to return rows in a different order between runs, and
    "deterministic" would quietly stop being true.
    """
    rows = conn.execute(
        """SELECT id, text FROM rag_chunks
           ORDER BY source_type, source_id, chunk_index, id"""
    ).fetchall()

    if not rows:
        print("Nothing to index.")
        # Clear any stale vector_ids so the database does not keep claiming
        # to be indexed after its last chunk has been removed.
        conn.execute("UPDATE rag_chunks SET vector_id = NULL")
        conn.commit()
        return 0

    print(f"Embedding {len(rows)} chunks...")
    vectors = embeddings.embed_texts([r["text"] for r in rows])

    print(f"Building FAISS index ({vectors.shape[0]} x {EMBEDDING_DIM})...")
    index = store.build_index(vectors)

    # Clear every vector_id BEFORE assigning the new ones.
    #
    # vector_id is UNIQUE, and reassignment shuffles the values around: a row
    # about to be given id 5 collides with whichever row still holds 5 from
    # the previous build. Updating row by row therefore fails partway through
    # with a UNIQUE constraint error -- but only once the ordering actually
    # changes, so it stays invisible while you are just re-running the same
    # corpus and appears the first time a document is added.
    #
    # NULL is exempt from UNIQUE in SQLite (and in the SQL standard), so
    # blanking them first gives the reassignment a clear field. Both
    # statements are in one transaction, so a crash between them cannot
    # leave half the rows renumbered.
    conn.execute("UPDATE rag_chunks SET vector_id = NULL")
    conn.executemany(
        "UPDATE rag_chunks SET vector_id = ? WHERE id = ?",
        [(i, row["id"]) for i, row in enumerate(rows)],
    )
    conn.commit()

    # Written only after the database is committed, and deliberately in this
    # order. If saving fails now, SQLite holds vector_ids for an index file
    # that is missing or stale -- which load_index reports as a clear error.
    # The reverse order fails silently instead: a new index file paired with
    # old vector_ids returns real text for the wrong vectors, which nothing
    # detects and which looks like the model simply giving poor answers.
    store.save_index(index)

    print(f"Index saved. {index.ntotal} vectors.")
    return len(rows)


def index_stats() -> dict:
    """What is indexed, and do SQLite and FAISS still agree with each other?"""
    conn = init_db()
    try:
        by_source = {
            row["source_type"]: row["n"]
            for row in conn.execute(
                """SELECT source_type, COUNT(*) AS n FROM rag_chunks
                   WHERE vector_id IS NOT NULL GROUP BY source_type"""
            )
        }
        indexed = sum(by_source.values())
        documents = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        meetings = conn.execute(
            """SELECT COUNT(DISTINCT source_id) AS n FROM rag_chunks
               WHERE source_type = 'transcript'"""
        ).fetchone()["n"]

        # rag_chunks.source_id cannot be a foreign key -- it points into
        # either documents or meetings depending on source_type, which SQL
        # cannot express -- so the check the database would normally do for
        # us is done here instead.
        orphans = conn.execute(
            """SELECT COUNT(*) AS n FROM rag_chunks c
               WHERE (c.source_type = 'document'
                      AND NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = c.source_id))
                  OR (c.source_type = 'transcript'
                      AND NOT EXISTS (SELECT 1 FROM meetings m WHERE m.id = c.source_id))"""
        ).fetchone()["n"]
    finally:
        conn.close()

    index = store.load_index()
    vectors = index.ntotal if index is not None else 0

    return {
        "documents": documents,
        "meetings_indexed": meetings,
        "chunks_from_documents": by_source.get("document", 0),
        "chunks_from_transcripts": by_source.get("transcript", 0),
        "indexed_chunks": indexed,
        "vectors_in_faiss": vectors,
        "orphaned_chunks": orphans,
        # If this is ever False the two stores have drifted, and retrieval
        # is returning text that does not correspond to the vector that
        # matched. Fix by re-running ingestion.
        "in_sync": indexed == vectors and orphans == 0,
    }

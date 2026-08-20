"""
The ingestion pipeline: files on disk -> searchable chunks.

    discover files
      -> skip unchanged ones (content hash)
      -> extract text
      -> split into overlapping chunks
      -> embed every chunk
      -> rebuild the FAISS index
      -> record chunk text + vector_id in SQLite

The rebuild step is the one that looks wasteful and is not. Read the long
comment on build_index in app/rag/store.py for why re-indexing everything
is the safe default.
"""

import uuid
from pathlib import Path

import numpy as np

from app.config import EMBEDDING_DIM, SAMPLE_DATA_DIR
from app.db import init_db, utc_now
from app.rag import embeddings, loaders, store
from app.rag.chunking import chunk_text

DEFAULT_DOCS_DIR = SAMPLE_DATA_DIR / "documents"


def ingest_directory(directory: Path = DEFAULT_DOCS_DIR, force: bool = False) -> dict:
    """Ingest every supported document under `directory`.

    force=True re-processes files even if their hash is unchanged -- which
    you need after changing the chunk size or the embedding model, because
    then the file is the same but the chunks derived from it are not.
    """
    conn = init_db()
    try:
        files = loaders.discover(directory)
        if not files:
            print(f"No supported documents found under {directory}")
            print(f"Supported types: {', '.join(sorted(loaders.SUPPORTED_SUFFIXES))}")
            return {"documents": 0, "chunks": 0, "skipped": 0}

        print(f"Found {len(files)} document(s) under {directory}")
        skipped = 0

        for path in files:
            digest = loaders.file_hash(path)
            rel_path = str(path.relative_to(SAMPLE_DATA_DIR.parent))

            existing = conn.execute(
                "SELECT id, content_hash FROM documents WHERE path = ?", (rel_path,)
            ).fetchone()

            if existing and existing["content_hash"] == digest and not force:
                print(f"  skip (unchanged): {path.name}")
                skipped += 1
                continue

            if existing:
                # The file changed, so its old chunks describe text that no
                # longer exists. Delete them before inserting the new ones,
                # or the index ends up holding both versions and retrieval
                # starts quoting stale policy back at people.
                conn.execute(
                    "DELETE FROM document_chunks WHERE document_id = ?",
                    (existing["id"],),
                )
                doc_id = existing["id"]
                conn.execute(
                    "UPDATE documents SET content_hash = ?, created_at = ? WHERE id = ?",
                    (digest, utc_now(), doc_id),
                )
            else:
                doc_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO documents (id, title, path, content_hash, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (doc_id, path.stem, rel_path, digest, utc_now()),
                )

            text = loaders.load_text(path)
            pieces = chunk_text(text)
            print(f"  {path.name}: {len(text):,} chars -> {len(pieces)} chunks")

            for i, piece in enumerate(pieces):
                # vector_id is left NULL here on purpose. It cannot be known
                # until every document has been processed and we know the
                # final ordering of the rebuilt index. It is filled in by
                # _rebuild_index below, in one pass, at the end.
                conn.execute(
                    """INSERT INTO document_chunks
                       (id, document_id, chunk_index, text, vector_id, created_at)
                       VALUES (?, ?, ?, ?, NULL, ?)""",
                    (str(uuid.uuid4()), doc_id, i, piece, utc_now()),
                )

            conn.commit()

        total_chunks = _rebuild_index(conn)
        return {
            "documents": len(files) - skipped,
            "chunks": total_chunks,
            "skipped": skipped,
        }
    finally:
        conn.close()


def _rebuild_index(conn) -> int:
    """Re-embed every chunk in the database and rebuild the FAISS index.

    Assigning vector_ids here, in a single deterministic pass over an
    ORDER BY, is what keeps the two stores consistent: position i in the
    index and the row with vector_id = i are guaranteed to be the same
    chunk, because both come from this one ordered list.
    """
    rows = conn.execute(
        "SELECT id, text FROM document_chunks ORDER BY document_id, chunk_index"
    ).fetchall()

    if not rows:
        print("Nothing to index.")
        return 0

    print(f"Embedding {len(rows)} chunks...")
    vectors = embeddings.embed_texts([r["text"] for r in rows])

    print(f"Building FAISS index ({vectors.shape[0]} x {EMBEDDING_DIM})...")
    index = store.build_index(vectors)
    store.save_index(index)

    conn.executemany(
        "UPDATE document_chunks SET vector_id = ? WHERE id = ?",
        [(i, row["id"]) for i, row in enumerate(rows)],
    )
    conn.commit()

    print(f"Index saved. {index.ntotal} vectors.")
    return len(rows)


def index_stats() -> dict:
    """Quick health check: do SQLite and FAISS agree with each other?"""
    conn = init_db()
    try:
        docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = conn.execute(
            "SELECT COUNT(*) AS n FROM document_chunks WHERE vector_id IS NOT NULL"
        ).fetchone()["n"]
    finally:
        conn.close()

    index = store.load_index()
    vectors = index.ntotal if index is not None else 0

    return {
        "documents": docs,
        "indexed_chunks": chunks,
        "vectors_in_faiss": vectors,
        # If this is ever False the two stores have drifted and retrieval
        # is returning text that does not correspond to the matched vector.
        # Fix by re-running ingest with force=True.
        "in_sync": chunks == vectors,
    }

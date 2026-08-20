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

from app.config import SAMPLE_DATA_DIR
from app.db import init_db, utc_now
from app.rag import loaders
from app.rag.chunking import chunk_text
from app.rag.indexing import index_stats, rebuild_index

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
                    """DELETE FROM rag_chunks
                       WHERE source_type = 'document' AND source_id = ?""",
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
                # start_ts/end_ts are NULL: a document has no time dimension.
                # vector_id is NULL too, on purpose -- it cannot be known
                # until every source has been written and the final index
                # ordering is decided. rebuild_index fills it in at the end.
                conn.execute(
                    """INSERT INTO rag_chunks
                       (id, source_type, source_id, source_title, chunk_index,
                        text, start_ts, end_ts, vector_id, created_at)
                       VALUES (?, 'document', ?, ?, ?, ?, NULL, NULL, NULL, ?)""",
                    (str(uuid.uuid4()), doc_id, path.stem, i, piece, utc_now()),
                )

            conn.commit()

        total_chunks = rebuild_index(conn)
        return {
            "documents": len(files) - skipped,
            "chunks": total_chunks,
            "skipped": skipped,
        }
    finally:
        conn.close()

"""
Everything that touches SQLite lives here.

Two rules this module exists to enforce:
  1. There is ONE definition of the schema. Stage 1 and Stage 2 each had
     their own copy of the same CREATE TABLE statements, which meant any
     schema change had to be made twice, correctly, or the two code paths
     would quietly drift apart.
  2. Nothing outside this module writes raw SQL for the core tables.
"""

import sqlite3
from datetime import datetime, timezone

from app.config import DB_PATH, ensure_dirs

# --- Schema ------------------------------------------------------------
#
# A note on how this differs from the architecture document:
#
# The doc specifies Postgres with a `document_embeddings` table holding a
# `vector(1536)` column (that is the pgvector extension). We are on
# SQLite, which has no vector type, so the vectors live in a FAISS index
# file instead, and `rag_chunks.vector_id` is the join key between
# the two. That column IS the interesting part of this schema -- read the
# comment on it below.

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS transcript_chunks (
    id          TEXT PRIMARY KEY,
    meeting_id  TEXT REFERENCES meetings(id),
    speaker_id  TEXT,
    text        TEXT,
    start_ts    REAL,
    end_ts      REAL,
    confidence  REAL,
    created_at  TEXT
);

-- Without this index, "give me the transcript for meeting X" scans every
-- chunk of every meeting ever recorded. With 767 chunks from one file
-- already in the database, that is worth doing now rather than later.
CREATE INDEX IF NOT EXISTS idx_chunks_meeting
    ON transcript_chunks(meeting_id, start_ts);

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    path          TEXT UNIQUE,
    -- Hash of the file's contents. On re-ingest we compare hashes and
    -- skip files that have not changed, so re-running the ingest script
    -- is cheap and idempotent instead of re-embedding everything.
    content_hash  TEXT,
    created_at    TEXT
);

-- ONE table for everything retrievable, whatever it came from.
--
-- The obvious alternative was a second table and a second FAISS index for
-- transcripts, kept parallel to the document ones. That fails the moment
-- a question needs both -- "does what we agreed in the meeting match the
-- written policy?" -- because two indexes give two separately-ranked lists
-- whose scores cannot be honestly compared: each is normalised against a
-- different corpus. Merging them means inventing a fusion rule.
--
-- One index over one embedding space makes that question a single search,
-- and the ranking is meaningful across sources for free. The cost is that
-- "search only this meeting" is no longer a cheap index lookup -- FAISS
-- IndexFlat carries no metadata and cannot filter. See the over-fetch
-- strategy in app/rag/retrieve.py for how that is paid for.
CREATE TABLE IF NOT EXISTS rag_chunks (
    id           TEXT PRIMARY KEY,

    -- 'document' or 'transcript'. Drives how the chunk is cited: a
    -- document cites a title and position, a transcript cites a meeting
    -- and a timestamp range you can actually seek to in the audio.
    source_type  TEXT NOT NULL,
    -- documents.id or meetings.id, depending on source_type. Deliberately
    -- NOT a foreign key: it points into one of two tables, and SQL cannot
    -- express that. The tradeoff is that the database can no longer stop
    -- an orphan, so app/rag/indexing.py checks for them instead.
    source_id    TEXT NOT NULL,
    -- Copied from the parent rather than joined at read time. This is
    -- denormalisation, done on purpose: retrieval returns chunks from two
    -- different parent tables at once, and a query that has to branch on
    -- source_type to know which table to join is both slower and uglier
    -- than carrying the title along. Titles here never change in place.
    source_title TEXT,

    chunk_index  INTEGER,
    text         TEXT,

    -- Seconds into the meeting. NULL for documents, which have no time
    -- dimension. This is what makes a transcript citation clickable.
    start_ts     REAL,
    end_ts       REAL,

    -- THE JOIN KEY BETWEEN SQLITE AND FAISS.
    --
    -- FAISS does not store text. It stores an array of vectors, and when
    -- you search it, it hands back integer positions -- "your nearest
    -- neighbours are vectors #4, #17, #92" -- plus distances. Those
    -- integers are meaningless on their own. This column is what turns
    -- them back into readable text.
    --
    -- Keeping these two stores in sync is the single most common source
    -- of bugs in a hand-rolled RAG system. Delete a chunk from SQLite
    -- without rebuilding the index and every vector_id after it still
    -- points at the old position -- so search silently returns the wrong
    -- text. We sidestep it by rebuilding the whole index from scratch;
    -- see app/rag/indexing.py.
    vector_id    INTEGER UNIQUE,
    created_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_ragchunks_vector ON rag_chunks(vector_id);
CREATE INDEX IF NOT EXISTS idx_ragchunks_source ON rag_chunks(source_type, source_id);
"""

# Tables that earlier versions of this project created and no longer uses.
# Everything in them is derived from files on disk or from transcript_chunks,
# so dropping is safe -- nothing here is a source of truth.
DEPRECATED_TABLES = ["document_chunks"]


def _drop_deprecated(conn) -> None:
    for table in DEPRECATED_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()


def utc_now() -> str:
    """Timestamp as an ISO-8601 string.

    datetime.utcnow() -- which the Stage 1/2 code used -- is deprecated
    as of Python 3.12 because it returns a naive datetime that merely
    claims to be UTC. This returns an explicitly timezone-aware one.
    """
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    """Open a connection with the settings this project wants everywhere."""
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)

    # Rows behave like dicts: row["text"] instead of row[3]. Positional
    # indexing breaks the moment someone adds a column in the middle.
    conn.row_factory = sqlite3.Row

    # SQLite ships with foreign keys DISABLED for backwards compatibility.
    # Your schema can declare REFERENCES all it likes and SQLite will
    # cheerfully let you insert an orphan row unless you turn this on.
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


def init_db() -> sqlite3.Connection:
    """Create any missing tables and return an open connection."""
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    _drop_deprecated(conn)
    return conn


# --- Meetings ----------------------------------------------------------

def create_meeting(conn: sqlite3.Connection, meeting_id: str, title: str) -> None:
    """Register a meeting. Safe to call again for an existing id.

    INSERT OR IGNORE rather than INSERT: if a client drops its WebSocket
    and reconnects with the same meeting_id, we want to carry on appending
    to that meeting, not crash on a primary key collision.
    """
    conn.execute(
        "INSERT OR IGNORE INTO meetings (id, title, created_at) VALUES (?, ?, ?)",
        (meeting_id, title, utc_now()),
    )
    conn.commit()


def insert_transcript_chunk(
    conn: sqlite3.Connection,
    chunk_id: str,
    meeting_id: str,
    text: str,
    start_ts: float,
    end_ts: float,
    confidence: float,
    speaker_id: str | None = None,
) -> None:
    conn.execute(
        """INSERT INTO transcript_chunks
           (id, meeting_id, speaker_id, text, start_ts, end_ts, confidence, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (chunk_id, meeting_id, speaker_id, text, start_ts, end_ts, confidence, utc_now()),
    )
    conn.commit()


def get_transcript(conn: sqlite3.Connection, meeting_id: str) -> list[sqlite3.Row]:
    """All chunks for one meeting, in the order they were spoken."""
    return conn.execute(
        """SELECT id, speaker_id, text, start_ts, end_ts, confidence
           FROM transcript_chunks
           WHERE meeting_id = ?
           ORDER BY start_ts""",
        (meeting_id,),
    ).fetchall()

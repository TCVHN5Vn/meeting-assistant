"""
Schema-level tests.

These use a throwaway database file per test (tmp_path is a pytest fixture
giving a fresh temporary directory), so they never touch data/ and can be
run in any order without interfering with each other.
"""

import sqlite3
import uuid

import pytest

from app import config, db


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    """A connection to an empty database in a temporary directory."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    connection = db.init_db()
    yield connection
    connection.close()


def test_schema_creates_all_tables(conn):
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"meetings", "transcript_chunks", "documents", "document_chunks"} <= tables


def test_create_meeting_is_idempotent(conn):
    """Reconnecting with the same meeting_id must not raise."""
    meeting_id = str(uuid.uuid4())
    db.create_meeting(conn, meeting_id, "First")
    db.create_meeting(conn, meeting_id, "Second")

    rows = conn.execute("SELECT title FROM meetings WHERE id = ?", (meeting_id,)).fetchall()
    assert len(rows) == 1
    # INSERT OR IGNORE keeps the ORIGINAL row rather than overwriting it.
    assert rows[0]["title"] == "First"


def test_transcript_is_ordered_by_start_time(conn):
    meeting_id = str(uuid.uuid4())
    db.create_meeting(conn, meeting_id, "Ordering")

    # Inserted deliberately out of order.
    for start in (10.0, 0.0, 5.0):
        db.insert_transcript_chunk(
            conn, str(uuid.uuid4()), meeting_id,
            text=f"at {start}", start_ts=start, end_ts=start + 1, confidence=-0.2,
        )

    chunks = db.get_transcript(conn, meeting_id)
    assert [c["start_ts"] for c in chunks] == [0.0, 5.0, 10.0]


def test_foreign_keys_are_enforced(conn):
    """SQLite disables foreign keys by default; get_connection turns them on.

    Without the PRAGMA this insert would silently succeed and leave an
    orphaned chunk pointing at a meeting that does not exist.
    """
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_transcript_chunk(
            conn, str(uuid.uuid4()), "no-such-meeting-id",
            text="orphan", start_ts=0.0, end_ts=1.0, confidence=-0.2,
        )

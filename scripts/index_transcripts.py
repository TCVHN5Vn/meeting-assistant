"""
Index meeting transcripts so they can be searched and asked about.

Usage (from the project root):
    python -m scripts.index_transcripts              # every meeting
    python -m scripts.index_transcripts <meeting_id> # just one
    python -m scripts.index_transcripts --list       # show what is available

In a finished product this would run automatically when a meeting ends,
triggered by the WebSocket disconnecting. It is a manual script here because
being able to re-run indexing on demand -- after changing the window size or
the confidence floor -- is exactly what you need while tuning it.
"""

import sys

from app.db import init_db
from app.rag.indexing import index_stats, rebuild_index
from app.rag.transcripts import index_meeting


def list_meetings(conn) -> None:
    rows = conn.execute(
        """SELECT m.id, m.title, COUNT(c.id) AS segments,
                  MAX(c.end_ts) AS duration,
                  (SELECT COUNT(*) FROM rag_chunks r
                   WHERE r.source_type='transcript' AND r.source_id=m.id) AS indexed
           FROM meetings m LEFT JOIN transcript_chunks c ON c.meeting_id = m.id
           GROUP BY m.id ORDER BY m.created_at"""
    ).fetchall()
    for row in rows:
        duration = f"{(row['duration'] or 0) / 60:.0f}m"
        state = f"{row['indexed']} chunks" if row["indexed"] else "not indexed"
        print(f"  {row['id']}  {row['title'][:20]:20} "
              f"{row['segments']:4d} segments  {duration:>5}  {state}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    conn = init_db()
    try:
        if "--list" in sys.argv:
            list_meetings(conn)
            return

        if args:
            meeting_ids = args
        else:
            # Meetings with no transcript rows would produce zero chunks and
            # just add noise to the output.
            meeting_ids = [
                r["id"] for r in conn.execute(
                    """SELECT DISTINCT m.id FROM meetings m
                       JOIN transcript_chunks c ON c.meeting_id = m.id
                       ORDER BY m.created_at"""
                )
            ]

        if not meeting_ids:
            print("No meetings with transcripts found. "
                  "Record or transcribe one first.")
            return

        total = 0
        for meeting_id in meeting_ids:
            count = index_meeting(conn, meeting_id)
            title = conn.execute(
                "SELECT title FROM meetings WHERE id = ?", (meeting_id,)
            ).fetchone()["title"]
            print(f"  {title[:24]:24} -> {count} windows")
            total += count

        print(f"\nWrote {total} transcript windows. Rebuilding index...")
        rebuild_index(conn)
    finally:
        conn.close()

    stats = index_stats()
    print(f"\nIndex now holds {stats['indexed_chunks']} chunks: "
          f"{stats['chunks_from_documents']} from {stats['documents']} document(s), "
          f"{stats['chunks_from_transcripts']} from "
          f"{stats['meetings_indexed']} meeting(s).")
    if not stats["in_sync"]:
        print("WARNING: SQLite and FAISS disagree. Re-run ingestion.")


if __name__ == "__main__":
    main()

"""
Remove meetings that are not worth keeping.

Development leaves debris: every aborted run, every reconnect, every
five-second smoke test becomes a row in `meetings` with a handful of
transcript segments behind it. Left alone it makes `--list` unreadable and
any screenshot of the data look careless.

Usage (from the project root):
    python -m scripts.prune_meetings                      # dry run
    python -m scripts.prune_meetings --apply
    python -m scripts.prune_meetings --shorter-than 100 --apply

DRY RUN BY DEFAULT

Deleting is the one operation here that cannot be undone from inside the
program, so it takes an explicit flag. The audio files are still on disk, so
anything removed can be transcribed again -- but that is a recovery path,
not a reason to be careless.

WHAT IS NEVER PRUNED

Anything present in the search index. Being indexed is the clearest
available signal that someone deliberately kept a meeting, and a short
meeting that was indexed on purpose is not debris. Enforced in the query
rather than left to the caller to remember.
"""

import sys

from app.db import init_db
from app.rag.indexing import rebuild_index

DEFAULT_MIN_SEGMENTS = 50


def find_prunable(conn, min_segments: int):
    """Meetings below the segment threshold that are not in the index."""
    return conn.execute(
        """SELECT m.id, m.title, COUNT(t.id) AS segments
           FROM meetings m
           LEFT JOIN transcript_chunks t ON t.meeting_id = m.id
           WHERE NOT EXISTS (
               SELECT 1 FROM rag_chunks r
               WHERE r.source_type = 'transcript' AND r.source_id = m.id
           )
           GROUP BY m.id
           HAVING COUNT(t.id) < ?
           ORDER BY segments DESC""",
        (min_segments,),
    ).fetchall()


def main() -> None:
    apply = "--apply" in sys.argv
    min_segments = DEFAULT_MIN_SEGMENTS
    if "--shorter-than" in sys.argv:
        min_segments = int(sys.argv[sys.argv.index("--shorter-than") + 1])

    conn = init_db()
    try:
        doomed = find_prunable(conn, min_segments)
        if not doomed:
            print(f"Nothing to prune (no unindexed meetings under "
                  f"{min_segments} segments).")
            return

        total = sum(row["segments"] for row in doomed)
        print(f"{'Would remove' if not apply else 'Removing'} {len(doomed)} "
              f"meeting(s), {total} transcript segments:\n")
        for row in doomed:
            print(f"  {row['segments']:5d} segments  {row['title'][:44]}")

        if not apply:
            print("\nDry run. Re-run with --apply to delete.")
            return

        ids = [row["id"] for row in doomed]
        placeholders = ",".join("?" for _ in ids)

        # Children first: foreign keys are enforced (see app/db.py), so
        # deleting a meeting that still has transcript rows would fail.
        conn.execute(
            f"DELETE FROM transcript_chunks WHERE meeting_id IN ({placeholders})", ids)
        conn.execute(
            f"""DELETE FROM rag_chunks
                WHERE source_type = 'transcript' AND source_id IN ({placeholders})""",
            ids)
        conn.execute(f"DELETE FROM meetings WHERE id IN ({placeholders})", ids)
        conn.commit()
        print(f"\nRemoved {len(ids)} meeting(s).")

        # Nothing indexed should have been touched, but if the filter ever
        # changes, vector_ids would be stale and retrieval would return the
        # wrong text. Rebuilding is cheap and makes that impossible.
        rebuild_index(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

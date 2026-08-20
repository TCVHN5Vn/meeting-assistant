"""
Extract action items from a meeting transcript.

Usage (from the project root):
    python -m scripts.extract_tasks --list
    python -m scripts.extract_tasks <meeting_id>
    python -m scripts.extract_tasks <meeting_id> --show

--show prints the stored tasks without re-extracting, which is what you want
after a run: extraction costs one model call per window and takes minutes on
a 50-minute meeting, so re-reading should not repeat it.

In a product this would run when the meeting ends. It is a script here
because extraction is the step whose prompt you tune most, and re-running it
on demand is the whole loop.
"""

import sys

from app.db import get_tasks, init_db
from app.rag.transcripts import format_timestamp
from app.tasks import extract_meeting


def show(conn, meeting_id: str) -> None:
    rows = get_tasks(conn, meeting_id)
    if not rows:
        print("No tasks stored for this meeting.")
        return

    print(f"{len(rows)} task(s):\n")
    for row in rows:
        when = format_timestamp(row["start_ts"]) if row["start_ts"] is not None else "?"
        print(f"  [{when}] {row['description']}")
        details = []
        if row["assignee"]:
            details.append(f"assignee: {row['assignee']}")
        if row["due"]:
            details.append(f"due: {row['due']}")
        details.append(f"status: {row['status']}")
        print(f"           {'  |  '.join(details)}")
        # The quote is printed every time, not hidden behind a flag. It is
        # the evidence for the task, and a task list you cannot check
        # against the recording is a list you have to take on faith.
        print(f"           \"{row['quote'][:100]}\"\n")


def list_meetings(conn) -> None:
    for row in conn.execute(
        """SELECT m.id, m.title, COUNT(t.id) AS segments,
                  (SELECT COUNT(*) FROM tasks k WHERE k.meeting_id = m.id) AS tasks
           FROM meetings m LEFT JOIN transcript_chunks t ON t.meeting_id = m.id
           GROUP BY m.id HAVING COUNT(t.id) > 0 ORDER BY segments DESC"""
    ):
        print(f"  {row['id']}  {row['title'][:34]:34} "
              f"{row['segments']:4d} segments  {row['tasks']} tasks")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    conn = init_db()
    try:
        if "--list" in sys.argv or not args:
            list_meetings(conn)
            return

        meeting_id = args[0]
        if "--show" in sys.argv:
            show(conn, meeting_id)
            return

        def progress(done, total):
            print(f"  window {done}/{total}...", flush=True)

        result = extract_meeting(conn, meeting_id, progress=progress)
        print(f"\n{result['windows']} windows -> {result['found']} candidates")
        print(f"  {result['rejected']} rejected (quote not found in transcript)")
        print(f"  {result['duplicates']} duplicates merged")
        print(f"  {result['stored']} stored\n")
        show(conn, meeting_id)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

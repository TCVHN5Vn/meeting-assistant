"""
Making meeting transcripts retrievable.

WHY THIS IS NOT JUST chunk_text() ON THE TRANSCRIPT

The obvious approach -- concatenate every segment into one string and run
the document chunker over it -- throws away the two things that make a
transcript a transcript:

  * TIME. Segments carry start and end timestamps. Flatten them into text
    and a retrieved passage can no longer say "this was said 34 minutes in",
    which is most of what makes a meeting citation useful: you can go and
    listen to it.

  * NATURAL BOUNDARIES. Written text signals structure with blank lines and
    full stops. Speech signals it with SILENCE. A three-second pause is the
    spoken equivalent of a paragraph break, and it is a far better place to
    cut than an arbitrary character count.

The unit of grouping is also different. ASR segments here average about 50
characters -- roughly ten words. Embedding one segment per vector produces
vectors for fragments like "Five to ten." which carry no meaning on their
own and will match queries at random. Segments have to be accumulated into
windows large enough to be *about* something first.

So: same underlying idea as document chunking (bounded size, overlap so
nothing falls between two chunks), different boundary signal and different
metadata carried through.
"""

import uuid
from dataclasses import dataclass

from app.config import (
    MIN_ASR_CONFIDENCE,
    TRANSCRIPT_GAP_SECONDS,
    TRANSCRIPT_MAX_SPAN_SECONDS,
    TRANSCRIPT_OVERLAP_CHARS,
    TRANSCRIPT_WINDOW_CHARS,
)
from app.db import utc_now


@dataclass
class TranscriptWindow:
    """A retrievable stretch of a meeting."""
    text: str
    start_ts: float
    end_ts: float
    segment_count: int


def window_segments(
    segments,
    max_chars: int = TRANSCRIPT_WINDOW_CHARS,
    overlap_chars: int = TRANSCRIPT_OVERLAP_CHARS,
    max_span_seconds: float = TRANSCRIPT_MAX_SPAN_SECONDS,
    gap_seconds: float = TRANSCRIPT_GAP_SECONDS,
    min_confidence: float = MIN_ASR_CONFIDENCE,
) -> list[TranscriptWindow]:
    """Group consecutive ASR segments into windows worth embedding.

    `segments` is any sequence of objects supporting ["text"], ["start_ts"],
    ["end_ts"] and ["confidence"] -- sqlite3.Row satisfies this directly.
    They must already be ordered by start_ts.

    A window is closed when any of three things happens:

      1. Adding the next segment would exceed `max_chars`. The size bound,
         same as for documents.
      2. The window would span more than `max_span_seconds`. A separate
         bound because a quiet stretch of meeting can stay under the
         character limit for half an hour, and "somewhere in this
         thirty-minute window" is not a useful citation.
      3. There is a silence longer than `gap_seconds` before the next
         segment. The topic boundary -- the reason to cut HERE rather than
         two sentences later.

    Segments below `min_confidence` are dropped before any of this. This is
    the payoff for storing avg_logprob raw back in Sprint 1: bad ASR text
    produces a vector that means nothing in particular, and a meaningless
    vector does not sit harmlessly in the index -- it is roughly equidistant
    from everything, so it surfaces against unrelated queries and pushes a
    genuinely relevant chunk out of the top-k. One bad chunk degrades
    answers about topics it has nothing to do with.

    Note the asymmetry: the same low-confidence text is still shown in the
    transcript view, where a human can judge it. Filter at read time, per
    use, not at write time.
    """
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    kept = [s for s in segments if s["confidence"] is None
            or s["confidence"] >= min_confidence]
    if not kept:
        return []

    windows: list[TranscriptWindow] = []
    current: list = []

    def flush() -> None:
        if not current:
            return
        windows.append(TranscriptWindow(
            text=" ".join(s["text"].strip() for s in current),
            start_ts=current[0]["start_ts"],
            end_ts=current[-1]["end_ts"],
            segment_count=len(current),
        ))

    def carry_over() -> list:
        """The tail of the current window, to seed the next one.

        Overlap is measured in characters but applied in whole SEGMENTS.
        Cutting a segment in half would leave a fragment whose timestamps
        no longer describe its text, and the timestamps are the point.
        """
        tail, size = [], 0
        for segment in reversed(current):
            length = len(segment["text"]) + 1
            if size + length > overlap_chars and tail:
                break
            tail.insert(0, segment)
            size += length
        # Never carry the entire window: that would make no forward
        # progress and the loop would emit the same window forever.
        return tail if len(tail) < len(current) else tail[1:]

    for segment in kept:
        if current:
            span = segment["end_ts"] - current[0]["start_ts"]
            chars = sum(len(s["text"]) + 1 for s in current) + len(segment["text"])
            gap = segment["start_ts"] - current[-1]["end_ts"]

            if gap > gap_seconds:
                # Split at a SILENCE: a genuine topic boundary, so no
                # carry-over. Overlap exists to protect passages from being
                # severed by an arbitrary cut -- but this cut is not
                # arbitrary, it is where the speaker stopped. Carrying text
                # across it would put the end of one topic at the head of
                # the next, and would stretch the new window's timestamps
                # back across the silence, so its citation would point at a
                # moment where nothing was said.
                flush()
                current = []
            elif chars > max_chars or span > max_span_seconds:
                # Split for SIZE or SPAN: an arbitrary cut mid-flow, which
                # is exactly the case overlap is for.
                flush()
                current = carry_over()

        current.append(segment)

    flush()
    return windows


def index_meeting(conn, meeting_id: str) -> int:
    """Turn one meeting's transcript into rag_chunks rows. Returns the count.

    Does not build the FAISS index -- app/rag/indexing.py does that once,
    after all sources have been written. Embedding is by far the expensive
    step, so it happens in a single batched pass rather than per meeting.
    """
    meeting = conn.execute(
        "SELECT id, title FROM meetings WHERE id = ?", (meeting_id,)
    ).fetchone()
    if meeting is None:
        raise ValueError(f"no such meeting: {meeting_id}")

    segments = conn.execute(
        """SELECT text, start_ts, end_ts, confidence
           FROM transcript_chunks WHERE meeting_id = ? ORDER BY start_ts""",
        (meeting_id,),
    ).fetchall()

    # Re-indexing a meeting replaces its chunks rather than adding to them,
    # so running this twice does not double the meeting in the index.
    conn.execute(
        "DELETE FROM rag_chunks WHERE source_type = 'transcript' AND source_id = ?",
        (meeting_id,),
    )

    windows = window_segments(segments)
    for i, window in enumerate(windows):
        conn.execute(
            """INSERT INTO rag_chunks
               (id, source_type, source_id, source_title, chunk_index,
                text, start_ts, end_ts, vector_id, created_at)
               VALUES (?, 'transcript', ?, ?, ?, ?, ?, ?, NULL, ?)""",
            (str(uuid.uuid4()), meeting_id, meeting["title"], i,
             window.text, window.start_ts, window.end_ts, utc_now()),
        )
    conn.commit()
    return len(windows)


def format_timestamp(seconds: float) -> str:
    """Seconds into the meeting as h:mm:ss or m:ss, for citations."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"

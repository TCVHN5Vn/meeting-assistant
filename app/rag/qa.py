"""
The full RAG pipeline, end to end:

    question -> retrieve -> build prompt -> generate -> answer + sources

This is the file that makes "RAG" concrete. Everything upstream of it is
plumbing; everything about answer quality is decided here or in retrieval.
"""

from dataclasses import dataclass
from typing import Iterator

from app.config import LIVE_CONTEXT_SECONDS
from app.db import init_db
from app.llm import ollama_client, prompts
from app.rag.retrieve import Hit, retrieve
from app.rag.transcripts import format_timestamp

# Chunks scoring below this are treated as irrelevant. See the note on
# min_score in app/rag/retrieve.py -- calibrate against your own corpus.
DEFAULT_MIN_SCORE = 0.25
DEFAULT_TOP_K = 5


@dataclass
class Answer:
    text: str
    sources: list[Hit]
    used_context: bool  # False when nothing cleared the relevance floor


def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    source_type: str | None = None,
    meeting_id: str | None = None,
) -> Answer:
    hits = retrieve(question, top_k=top_k, min_score=min_score,
                    source_type=source_type, meeting_id=meeting_id)

    # Refuse BEFORE calling the model, not after. If retrieval found
    # nothing relevant there is no answer to be had, and asking the LLM
    # anyway is exactly how you get a fluent hallucination -- it will
    # happily answer from its training data unless it is stopped here.
    # This early return costs nothing and removes a whole class of failure.
    if not hits:
        return Answer(text=prompts.NO_CONTEXT_ANSWER, sources=[], used_context=False)

    text = ollama_client.chat(
        system=prompts.RAG_SYSTEM_PROMPT,
        user=prompts.build_rag_user_prompt(question, hits),
    )
    return Answer(text=text, sources=hits, used_context=True)


def answer_question_stream(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    source_type: str | None = None,
    meeting_id: str | None = None,
) -> tuple[list[Hit], Iterator[str]]:
    """Streaming variant.

    Returns the sources immediately and the answer as a generator, so a UI
    can render "Searching 3 documents..." with the real titles while the
    model is still writing. Retrieval takes milliseconds; generation takes
    seconds. No reason to make the user wait for the fast part.
    """
    hits = retrieve(question, top_k=top_k, min_score=min_score,
                    source_type=source_type, meeting_id=meeting_id)

    if not hits:
        return [], iter([prompts.NO_CONTEXT_ANSWER])

    stream = ollama_client.chat_stream(
        system=prompts.RAG_SYSTEM_PROMPT,
        user=prompts.build_rag_user_prompt(question, hits),
    )
    return hits, stream


# --- Live meeting Q&A ---------------------------------------------------

def recent_transcript(meeting_id: str, seconds: float = LIVE_CONTEXT_SECONDS) -> str:
    """The last `seconds` of this meeting, as plain timestamped text.

    Read straight out of transcript_chunks rather than out of the index,
    because a meeting in progress is not indexed yet -- and because pasting
    a short bounded window in verbatim is better than retrieving over it
    anyway. See build_live_user_prompt for that argument.

    The window is measured back from the LAST timestamp in the meeting, not
    from wall-clock time. Those are different numbers whenever transcription
    has fallen behind the audio, and the transcript's own clock is the one
    that matches the text.
    """
    conn = init_db()
    try:
        latest = conn.execute(
            "SELECT MAX(end_ts) AS t FROM transcript_chunks WHERE meeting_id = ?",
            (meeting_id,),
        ).fetchone()["t"]
        if latest is None:
            return ""

        rows = conn.execute(
            """SELECT text, start_ts FROM transcript_chunks
               WHERE meeting_id = ? AND end_ts >= ?
               ORDER BY start_ts""",
            (meeting_id, latest - seconds),
        ).fetchall()
    finally:
        conn.close()

    return "\n".join(f"[{format_timestamp(r['start_ts'])}] {r['text']}" for r in rows)


def answer_live_stream(
    question: str,
    meeting_id: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> tuple[list[Hit], str, Iterator[str]]:
    """Answer a question asked aloud during a meeting in progress.

    Returns (retrieved hits, recent transcript, token stream).

    Note what does NOT happen here: there is no early refusal when retrieval
    comes back empty. The post-meeting path refuses in that case, and should
    -- with no relevant documents there is nothing to ground an answer in.
    Live is different: "what did we just decide?" is answerable entirely
    from the recent discussion and will legitimately retrieve nothing.
    Refusing on empty retrieval would break exactly the questions a live
    assistant is most useful for.
    """
    hits = retrieve(question, top_k=top_k, min_score=min_score)
    recent = recent_transcript(meeting_id)

    stream = ollama_client.chat_stream(
        system=prompts.LIVE_SYSTEM_PROMPT,
        user=prompts.build_live_user_prompt(question, hits, recent),
    )
    return hits, recent, stream

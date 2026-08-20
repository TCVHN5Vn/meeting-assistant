"""
Pulling action items out of a meeting transcript.

WHAT MAKES THIS HARD IS NOT THE EXTRACTION

Asking a model to list the action items in a passage works on the first
try. The problem is that it works just as readily when there are none.
Meetings are full of language that sounds like commitment and is not --
"we should probably look at that", "someone ought to check" -- and a model
asked to find tasks will find them. A task list with three invented entries
is worse than no task list, because it has to be checked line by line
against the recording, which is the work it was supposed to save.

So the design is built around not trusting the output:

  1. The model must return a QUOTE, copied verbatim from the transcript.
  2. The quote is checked against the source text in code. No match, no task.
  3. The quote is traced back to the segment it came from, giving a
     timestamp somebody can go and listen to.

Point 2 is the load-bearing one. A model that invents a task must also
invent the sentence it came from, and an invented sentence will not be
found in the transcript. It converts an unfalsifiable claim into one the
program can check for itself.

That is the same shape as the relevance floor in app/rag/qa.py: a rule
enforced in code, not a request made in a prompt.
"""

import re
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.config import OLLAMA_MODEL
from app.db import clear_tasks, insert_task
from app.llm import ollama_client
from app.rag.transcripts import window_segments

# Bigger windows than retrieval uses. Retrieval wants a tight passage so its
# vector stays sharp; extraction wants enough context to tell a decision
# from a musing, and to catch the reply that names who is doing it.
EXTRACTION_WINDOW_CHARS = 2500
EXTRACTION_OVERLAP_CHARS = 400
EXTRACTION_MAX_SPAN_SECONDS = 400.0

# Two descriptions sharing this proportion of their words are the same task.
DEDUP_THRESHOLD = 0.6

# Shortest quote that counts as evidence. See verify_quote.
MIN_QUOTE_WORDS = 5

# Length, in words, of the contiguous run a segment must share with a quote
# to count as part of it. See strategy 3 in locate().
MIN_SHARED_RUN = 4

TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "assignee": {"type": ["string", "null"]},
                    "due": {"type": ["string", "null"]},
                    "quote": {"type": "string"},
                },
                "required": ["description", "assignee", "due", "quote"],
            },
        }
    },
    "required": ["tasks"],
}

SYSTEM_PROMPT = """You extract action items from meeting transcripts.

An action item is something a specific person or group committed to DOING. \
Extract it only when the transcript shows a commitment or an instruction.

Do NOT extract:
- Topics discussed, decisions recorded, or facts stated.
- Vague aspirations: "we should look at that", "it would be good to".
- Anything you are inferring rather than reading.

For each action item:
- description: what must be done, as a short imperative.
- assignee: the name or role given in the transcript. If nobody is named, \
use null. Do not write "unassigned", "someone", or "the team" -- use null.
- due: the deadline AS SPOKEN, e.g. "before Friday", "next meeting". If no \
deadline is stated, use null. Do not invent one, and do not write "soon" \
or "TBD" -- use null.
- quote: the sentence from the transcript that the item comes from, copied \
EXACTLY. Do not paraphrase, correct or shorten it.

The transcript is speech recognition output and contains errors. Read for \
meaning, but copy the quote exactly as written, errors included.

If there are no action items, return an empty list. An empty list is a \
correct and useful answer; inventing an item to avoid returning one is not."""


@dataclass
class ExtractedTask:
    description: str
    assignee: str | None
    due: str | None
    quote: str
    start_ts: float
    end_ts: float


# Values models reach for instead of null, despite being told not to.
# Checked case-insensitively after stripping punctuation.
_NULL_ISH = {
    "null", "none", "n/a", "na", "unknown", "unassigned", "tbd", "tba",
    "someone", "somebody", "anyone", "the team", "team", "everyone",
    "not specified", "not stated", "not mentioned", "soon", "asap",
    "no deadline", "none specified", "unspecified",
}


def _clean(value):
    """Normalise the model's stand-ins for null into an actual null.

    The schema declares these fields nullable and the prompt asks for null,
    and the model still returns "unassigned" or "soon" -- because a schema
    constrains SHAPE, not content, and a string is a valid string. Storing
    "soon" as a deadline would put a fabricated value in a column that
    otherwise means something.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.sub(r"[^\w\s]", "", text).strip().lower() in _NULL_ISH:
        return None
    return text


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def verify_quote(quote: str, source: str) -> bool:
    """Is this quote actually in the transcript?

    Compared on normalised text, because the model reliably tidies
    punctuation and capitalisation while copying even when told not to, and
    rejecting a real quote over a comma would throw away good tasks.

    Anything shorter than MIN_QUOTE_WORDS is rejected outright. Word count
    is a proxy for specificity: a quote generic enough to appear in any
    meeting -- "yes", "I will do that" -- verifies against the transcript
    while proving nothing about whether the task was real. Five is a
    judgement call, set so that "I will do that" (four) fails and "Karen
    will book the hall" (five) passes.
    """
    normal_quote = _normalise(quote)
    if len(normal_quote.split()) < MIN_QUOTE_WORDS:
        return False
    return normal_quote in _normalise(source)


def _longest_run(a: list[str], b: list[str]) -> int:
    """Length of the longest run of words appearing in both, in order."""
    return SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b)).size


def locate(quote: str, segments) -> tuple[float, float]:
    """Find when the quoted sentence was said.

    Returns (start, end) covering the segments the quote spans.

    Three strategies, most precise first. The naive version tried only the
    first two and put a task at 0:00 that was said 26 seconds in, because a
    quote can straddle a segment boundary without either side containing the
    other:

        segment: "...nothing is decided yet. One more thing, Naomi, please
                  update the on call"
        segment: "rota before the end of the month. Understood, I will do that."
        quote:   "Naomi, please update the on call rota before the end of
                  the month."

    Each segment holds words the quote does not, and the quote holds words
    from both, so neither containment test fires.
    """
    normal_quote = _normalise(quote)

    # 1. The segment sits inside the quote (quote spans whole segments).
    hits = [s for s in segments if _normalise(s["text"]) in normal_quote]

    # 2. The quote sits inside one long segment.
    if not hits:
        hits = [s for s in segments if normal_quote in _normalise(s["text"])]

    # 3. Partial overlap at both ends: the segment shares a long CONTIGUOUS
    #    run of words with the quote.
    #
    #    A first attempt used the fraction of the segment's words appearing
    #    anywhere in the quote, and failed on the very case it was written
    #    for -- the two segments above scored 0.43 and 0.58 against a 0.6
    #    bar, because each carries a trailing clause the quote does not.
    #    Set overlap counts scattered words, so it is diluted by exactly the
    #    extra text that makes this case hard, and raising the threshold
    #    would have made it worse.
    #
    #    A contiguous run is the right signal: six words in a row in the
    #    same order is a quotation, whereas sharing "the" and "of" six times
    #    over is not. Those two segments share runs of 6 and 7 words.
    if not hits:
        quote_word_list = normal_quote.split()
        hits = [
            segment for segment in segments
            if _longest_run(_normalise(segment["text"]).split(), quote_word_list)
            >= MIN_SHARED_RUN
        ]

    if not hits:
        # A slightly wide timestamp is still useful; dropping a verified
        # task over one would not be.
        return segments[0]["start_ts"], segments[-1]["end_ts"]
    return hits[0]["start_ts"], hits[-1]["end_ts"]


def is_duplicate(description: str, seen: list[str]) -> bool:
    """Has this task already been extracted?

    Windows overlap so that a task is never split across a boundary, which
    means the same sentence is shown to the model twice and the same task
    comes back twice. Tasks are also genuinely restated in meetings --
    raised early, confirmed later.

    Compared by word overlap (Jaccard) rather than exact match, because the
    model phrases the same commitment differently on each pass: "circulate
    the budget" and "send the budget to members" describe one job.
    """
    words = set(_normalise(description).split())
    if not words:
        return True
    for other in seen:
        other_words = set(_normalise(other).split())
        if not other_words:
            continue
        overlap = len(words & other_words) / len(words | other_words)
        if overlap >= DEDUP_THRESHOLD:
            return True
    return False


def extract_from_window(window) -> tuple[list[ExtractedTask], int]:
    """Run the model over one window. Returns (tasks, rejected_count)."""
    result = ollama_client.chat_json(
        system=SYSTEM_PROMPT,
        user=f"Meeting transcript:\n\n{window.text}",
        schema=TASK_SCHEMA,
    )

    tasks, rejected = [], 0
    for raw in result.get("tasks", []):
        description = _clean(raw.get("description"))
        quote = (raw.get("quote") or "").strip()
        if not description or not quote:
            rejected += 1
            continue

        if not verify_quote(quote, window.text):
            # The model produced a task whose supporting sentence is not in
            # the transcript. Either it paraphrased when told not to, or it
            # invented the whole thing. Neither is worth storing, and there
            # is no way to tell which from here.
            rejected += 1
            continue

        start_ts, end_ts = locate(quote, window.segments)
        tasks.append(ExtractedTask(
            description=description,
            assignee=_clean(raw.get("assignee")),
            due=_clean(raw.get("due")),
            quote=quote,
            start_ts=start_ts,
            end_ts=end_ts,
        ))
    return tasks, rejected


def extract_meeting(conn, meeting_id: str, progress=None) -> dict:
    """Extract every action item in a meeting and store them.

    Replaces any tasks already stored for the meeting, so re-running after
    a prompt change gives a clean result instead of a doubled one.
    """
    meeting = conn.execute(
        "SELECT id, title FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
    if meeting is None:
        raise ValueError(f"no such meeting: {meeting_id}")

    segments = conn.execute(
        """SELECT text, start_ts, end_ts, confidence FROM transcript_chunks
           WHERE meeting_id = ? ORDER BY start_ts""",
        (meeting_id,),
    ).fetchall()
    if not segments:
        return {"windows": 0, "found": 0, "rejected": 0, "duplicates": 0, "stored": 0}

    windows = window_segments(
        segments,
        max_chars=EXTRACTION_WINDOW_CHARS,
        overlap_chars=EXTRACTION_OVERLAP_CHARS,
        max_span_seconds=EXTRACTION_MAX_SPAN_SECONDS,
    )

    clear_tasks(conn, meeting_id)
    detected_by = f"{OLLAMA_MODEL} + quote-verified"

    found = rejected = duplicates = stored = 0
    seen: list[str] = []

    for i, window in enumerate(windows):
        if progress:
            progress(i + 1, len(windows))
        tasks, window_rejected = extract_from_window(window)
        found += len(tasks)
        rejected += window_rejected

        for task in tasks:
            if is_duplicate(task.description, seen):
                duplicates += 1
                continue
            seen.append(task.description)
            insert_task(
                conn,
                task_id=str(uuid.uuid4()),
                meeting_id=meeting_id,
                description=task.description,
                assignee=task.assignee,
                due=task.due,
                quote=task.quote,
                start_ts=task.start_ts,
                end_ts=task.end_ts,
                detected_by=detected_by,
            )
            stored += 1
        conn.commit()

    return {"windows": len(windows), "found": found, "rejected": rejected,
            "duplicates": duplicates, "stored": stored}

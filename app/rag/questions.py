"""
Deciding which utterances in a live meeting deserve an answer.

THE PROBLEM THIS EXISTS TO SOLVE

A meeting is full of questions. Real ones, from the transcript in this
repository: "How have those been remembered for the past year?", "What is
the special general meeting?". Every one is a genuine question, and not one
of them is addressed to an assistant. They are people talking to each other.

So "detect questions and answer them" -- the sprint's one-line description --
is not the requirement. Answering every question in a meeting produces an
assistant that interrupts constantly with unwanted answers, and an assistant
that interrupts constantly gets switched off in the first ten minutes.

The asymmetry is the whole design. A missed question costs a person typing
it out by hand. A false positive costs everyone's attention, mid-meeting,
and some credibility besides. Precision matters far more than recall here,
so the default is explicit address: the assistant answers when spoken to.

MATCHING HAS TO TOLERATE MISRECOGNITION

The first version matched the literal string "hey assistant" and did not
fire once on real audio, because Whisper transcribed it as "he assistant".

That is not bad luck, it is structural. A wake phrase is the hardest thing
in the sentence for a recogniser to get right: it is short, it is said
quickly and flatly, and it sits at a phrase boundary where the model has no
surrounding words to constrain its guess. The part of the utterance you are
keying on is the part most likely to come back wrong.

So the name is matched exactly -- "assistant" is long and distinctive enough
to survive -- while the greeting in front of it is matched against a set
that includes the mishearings actually observed. Requiring SOME greeting is
what keeps precision: a bare "assistant" appears in ordinary sentences
("the assistant will circulate the notes") and would fire on them.

DETECTION IS A CHEAP GATE, NOT A MODEL

All of this is string matching, deliberately. It runs on every transcript
chunk, roughly every five seconds, and it has to be free. Asking an LLM
"is this a question for you?" would cost a full generation per chunk on a
machine already running Whisper -- and would make the assistant slower at
the one thing it is supposed to be fast at. Cheap gate first, expensive
model only once the gate passes, is the general shape.
"""

import re
from dataclasses import dataclass

from app.config import ASSISTANT_NAME, AUTO_ANSWER_MODE, WAKE_GREETINGS

# Interrogatives that start a question. Used only as a fallback for when
# speech recognition drops the question mark, which it does often enough to
# matter -- Whisper punctuates from prosody, and a flatly-delivered question
# frequently comes back as a statement.
_INTERROGATIVES = (
    "what", "when", "where", "who", "whom", "whose", "why", "how", "which",
    "can", "could", "should", "would", "will", "does", "do", "did", "is",
    "are", "was", "were", "has", "have", "had", "may", "might", "shall",
)

# Punctuation stripped before matching, so "Hey, assistant --" and
# "hey assistant" are the same thing.
_NORMALISE = re.compile(r"[^\w\s]")

# <greeting> [meeting] assistant  -- the greeting is required, the word
# "meeting" optional, the name exact.
_WAKE_RE = re.compile(
    r"\b(?:" + "|".join(WAKE_GREETINGS) + r")\s+(?:meeting\s+)?"
    + re.escape(ASSISTANT_NAME) + r"\b"
)


@dataclass
class DetectedQuestion:
    """A question the assistant should answer, and why it thinks so."""
    text: str
    trigger: str  # 'wake_phrase' or 'question_form'


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Applied before matching so wake phrases survive whatever punctuation the
    transcriber decided on, and so trailing filler does not prevent a match.
    """
    return " ".join(_NORMALISE.sub(" ", text.lower()).split())


def looks_like_question(text: str) -> bool:
    """Is this utterance question-shaped?

    Two signals, in order of reliability:

    1. It ends in a question mark. Whisper does punctuate, so this is the
       strong signal when it is present.
    2. It opens with an interrogative. The fallback for when punctuation was
       dropped -- but a weak signal on its own, because "how we handled that
       was fine" also opens with one. The length floor filters the worst of
       it: fragments like "how" or "is it" carry no answerable content.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True

    words = normalise(stripped).split()
    if len(words) < 4:
        return False
    return words[0] in _INTERROGATIVES


def strip_wake_phrase(text: str) -> str | None:
    """Return what follows the wake phrase, or None if there is no wake phrase.

    Matching is done on the NORMALISED text but the slice is taken from the
    original, so the returned question keeps its capitalisation and
    punctuation -- both of which the language model reads as signal.

    A wake phrase with nothing after it ("hey assistant") returns None rather
    than an empty question: the speaker started addressing the assistant and
    has not said what they want yet. Answering an empty question would send
    back whatever the retriever happened to consider closest to nothing.
    """
    flat = normalise(text)
    match = _WAKE_RE.search(flat)
    if match is None:
        return None

    # Map the position in the normalised string back to the original by
    # counting WORDS, which survives punctuation removal. Character offsets
    # do not: stripping punctuation shifts every later index.
    words_before = len(flat[:match.start()].split())
    words_in_phrase = len(match.group().split())
    remainder = " ".join(text.split()[words_before + words_in_phrase:])

    return remainder.lstrip(" ,.:;-").strip() or None


def detect(text: str, mode: str = AUTO_ANSWER_MODE) -> DetectedQuestion | None:
    """Should the assistant answer this utterance? If so, with what question?

    Modes:

      'wake'      Answer only when addressed by name. The default, and the
                  right default: precision over recall, for the reason in
                  this module's docstring.
      'questions' Answer anything question-shaped. Noisy in a real meeting
                  -- useful for demonstrating the pipeline, and honest about
                  what it costs.
      'off'       Never auto-answer. Explicit `ask_query` events only.

    Note that 'wake' does NOT additionally require question form. "Hey
    assistant, remind me what the notice period is" is an imperative, not a
    question, and refusing it because it lacks a question mark would be
    pedantry -- being addressed directly is the signal that matters.
    """
    if mode == "off":
        return None

    asked = strip_wake_phrase(text)
    if asked is not None:
        return DetectedQuestion(text=asked, trigger="wake_phrase")

    if mode == "questions" and looks_like_question(text):
        return DetectedQuestion(text=text.strip(), trigger="question_form")

    return None


def detect_in_segments(texts: list[str], mode: str = AUTO_ANSWER_MODE) -> DetectedQuestion | None:
    """Find a question across the sentence segments of one utterance.

    WHY NOT JUST JOIN THEM AND RUN detect() ON THAT

    Because everything after the wake phrase would become the question. A
    real run, one utterance, five segments:

        So the next item on the agenda is the notice period.
        I think we should check what the policy actually says.
        Hey assistant, what is the notice period for a general meeting?
        Let us see what it comes back with.
        And then we can move on to the financial report.

    Joined, the question comes out as "what is the notice period for a
    general meeting? Let us see what it comes back with. And then we can move
    on to the financial report." -- the question plus two sentences that were
    not addressed to anyone.

    Whisper's segments are sentences, so the sentence containing the wake
    phrase is the question, and the ones after it are not. Sprint 3 joined
    them because the wake phrase and its question routinely landed in
    different five-second chunks; with audio cut at pauses instead, an
    utterance holds whole sentences and the join is no longer needed.

    One exception is kept: a wake phrase with nothing after it in its own
    segment ("Hey assistant.") takes the NEXT segment as the question, since
    the speaker addressed the assistant and then asked.
    """
    if mode == "off":
        return None

    for i, text in enumerate(texts):
        asked = strip_wake_phrase(text)
        if asked:
            return DetectedQuestion(text=asked, trigger="wake_phrase")

        # Wake phrase present but nothing after it in this sentence.
        if asked is None and strip_wake_phrase(f"{text} placeholder") is not None:
            following = texts[i + 1] if i + 1 < len(texts) else ""
            if following.strip():
                return DetectedQuestion(text=following.strip(), trigger="wake_phrase")
            return None

    if mode == "questions":
        for text in texts:
            if looks_like_question(text):
                return DetectedQuestion(text=text.strip(), trigger="question_form")

    return None

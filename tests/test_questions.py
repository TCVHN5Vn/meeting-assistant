"""
Tests for deciding which utterances deserve an answer.

Precision matters more than recall here, so most of these tests are about
what must NOT fire. An assistant that answers questions people were asking
each other interrupts a meeting constantly and gets switched off; one that
occasionally misses being addressed costs someone typing their question.

The strings below are taken from, or modelled on, real transcript output --
including the recognition errors.
"""

import pytest

from app.rag.questions import (
    detect,
    looks_like_question,
    normalise,
    strip_wake_phrase,
)


# --- wake phrase matching -----------------------------------------------

def test_plain_wake_phrase():
    assert strip_wake_phrase("Hey assistant, what is the leave policy?") \
        == "what is the leave policy?"


def test_misrecognised_greeting_still_matches():
    """The failure that made the first version never fire on real audio.

    Whisper transcribed "hey assistant" as "he assistant". The wake phrase is
    short, fast and sits at a phrase boundary, so it is the part of the
    sentence most likely to come back wrong -- matching it strictly means
    matching the least reliable words in the utterance.
    """
    assert strip_wake_phrase("he assistant, what is the notice period?") \
        == "what is the notice period?"


def test_misrecognised_NAME_still_matches():
    """The second half of the same lesson, found in real use.

    The first fix made the greeting tolerant and kept the name exact,
    reasoning that "assistant" was long enough to survive. A real recording
    returned "Hey, Assessent, what's the notes period..." -- the name is no
    safer than the greeting.
    """
    assert strip_wake_phrase("Hey, Assessent, what's the notes period?") \
        == "what's the notes period?"
    assert strip_wake_phrase("hey assistance what did we decide") \
        == "what did we decide"
    assert strip_wake_phrase("hi assistent can you summarise") \
        == "can you summarise"


def test_similar_words_after_a_greeting_do_not_fire():
    """A bare similarity score cannot separate these from real mishearings.

        assessent 0.67   insistent 0.67   consistent 0.63

    Requiring the first two letters to match is what does it: recognisers
    rarely mangle the opening of a stressed word, and these confusions all
    happen in the middle and the end.
    """
    assert strip_wake_phrase("hey everyone, please be consistent about this") is None
    assert strip_wake_phrase("hi, what is the attendance for tonight") is None
    assert strip_wake_phrase("ok, the president will open the meeting") is None


def test_wake_phrase_mid_utterance():
    """Chunks rarely start neatly at the wake phrase."""
    text = "So, um, what the policy says about this. Hey assistant, what changed?"
    assert strip_wake_phrase(text) == "what changed?"


def test_optional_meeting_in_the_name():
    assert strip_wake_phrase("okay meeting assistant, summarise that") \
        == "summarise that"


def test_bare_name_does_not_match():
    """A greeting is required, and that is what keeps precision.

    "assistant" appears in ordinary meeting sentences. Matching it alone
    would fire on them.
    """
    assert strip_wake_phrase("the assistant will circulate the notes") is None
    assert strip_wake_phrase("she is the assistant treasurer") is None


def test_wake_phrase_with_nothing_after_it():
    """Someone started addressing the assistant and has not asked yet.

    Returning an empty question would send whatever the retriever considered
    closest to nothing.
    """
    assert strip_wake_phrase("hey assistant") is None
    assert strip_wake_phrase("Hey assistant.") is None


def test_punctuation_does_not_block_matching():
    assert strip_wake_phrase("Hey, assistant -- what is the quorum?") \
        == "what is the quorum?"


def test_capitalisation_is_preserved_in_the_question():
    """Matching happens on normalised text; the slice comes from the original.

    Case and punctuation are signal for the language model, so they must
    survive detection.
    """
    assert strip_wake_phrase("Hey assistant, what did Karen say about SAML?") \
        == "what did Karen say about SAML?"


# --- question shape ------------------------------------------------------

def test_question_mark_is_enough():
    assert looks_like_question("What is the quorum?")


def test_interrogative_opening_without_punctuation():
    assert looks_like_question("how many days notice do we need")


def test_short_fragments_are_not_questions():
    """"how" or "is it" carry no answerable content."""
    assert not looks_like_question("how")
    assert not looks_like_question("is it")
    assert not looks_like_question("")


def test_statement_opening_with_an_interrogative():
    """The weakness of the fallback signal, made explicit."""
    assert not looks_like_question("The board reviewed the reserves last month.")


# --- mode behaviour ------------------------------------------------------

def test_wake_mode_ignores_questions_between_people():
    """The core precision requirement, using a real line from the transcript."""
    assert detect("How have those been remembered for the past year?",
                  mode="wake") is None


def test_wake_mode_answers_when_addressed():
    result = detect("Hey assistant, what is the quorum?", mode="wake")
    assert result is not None
    assert result.text == "what is the quorum?"
    assert result.trigger == "wake_phrase"


def test_wake_mode_accepts_an_imperative():
    """Being addressed is the signal, not question form.

    "Hey assistant, remind me what the notice period is" is a request, not a
    question. Refusing it for lacking a question mark would be pedantry.
    """
    result = detect("Hey assistant, remind me what the notice period is",
                    mode="wake")
    assert result is not None
    assert result.text == "remind me what the notice period is"


def test_questions_mode_fires_on_any_question():
    result = detect("How have those been remembered for the past year?",
                    mode="questions")
    assert result is not None
    assert result.trigger == "question_form"


def test_questions_mode_still_prefers_the_wake_phrase():
    """Addressed directly, the question is the part AFTER the wake phrase."""
    result = detect("Hey assistant, what is the quorum?", mode="questions")
    assert result.trigger == "wake_phrase"
    assert result.text == "what is the quorum?"


def test_off_mode_never_fires():
    assert detect("Hey assistant, what is the quorum?", mode="off") is None
    assert detect("What is the quorum?", mode="off") is None


@pytest.mark.parametrize("text,expected", [
    ("Hey, Assistant!", "hey assistant"),
    ("  MULTIPLE   spaces  ", "multiple spaces"),
    ("punctuation... removed?", "punctuation removed"),
])
def test_normalise(text, expected):
    assert normalise(text) == expected

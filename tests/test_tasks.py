"""
Tests for action-item verification.

The extraction itself is a model call and is not unit-tested. What IS tested
is everything the system does to avoid trusting it: checking that a quoted
sentence really appears in the transcript, turning the model's stand-ins for
null into actual nulls, locating a quote in time, and merging the duplicates
that overlapping windows produce.

That split is the point. The model is allowed to be wrong; the code around
it is not allowed to pass a wrong answer through unchecked.
"""

import pytest

from app.tasks import _clean, is_duplicate, locate, verify_quote

TRANSCRIPT = (
    "Karen, can you circulate the revised budget to the members before Friday? "
    "Yes I will do that. And we still need someone to book the hall for the AGM. "
    "I can take that on. Let us move on to the next item."
)


def seg(text, start, end):
    return {"text": text, "start_ts": start, "end_ts": end, "confidence": -0.3}


# --- quote verification: the load-bearing guard ---------------------------

def test_real_quote_is_accepted():
    assert verify_quote(
        "can you circulate the revised budget to the members before Friday",
        TRANSCRIPT)


def test_invented_quote_is_rejected():
    """The whole reason quotes are required.

    A model that invents a task must also invent the sentence it came from,
    and an invented sentence is not in the transcript. This turns an
    unfalsifiable claim into one the program can check.
    """
    assert not verify_quote(
        "John agreed to update the website by the end of the month", TRANSCRIPT)


def test_punctuation_and_case_differences_are_tolerated():
    """Models tidy quotes while copying, even when told not to.

    Rejecting a genuine quote over a comma would throw away real tasks, so
    the comparison is on normalised text.
    """
    assert verify_quote(
        "Karen can you circulate the revised budget to the members before Friday!!",
        TRANSCRIPT)
    assert verify_quote(
        "YES I WILL DO THAT and we still need someone to book the hall",
        TRANSCRIPT)


def test_paraphrase_is_rejected():
    """"Copy it exactly" is an instruction; this is the enforcement."""
    assert not verify_quote(
        "Karen will send out the updated budget to everyone by Friday", TRANSCRIPT)


def test_generic_quotes_are_rejected():
    """A quote can be present in the transcript and still prove nothing.

    "I will do that" verifies perfectly and appears in almost every meeting.
    Word count stands in for specificity: too short, and matching the
    transcript is not evidence that the task was real.
    """
    assert not verify_quote("Yes", TRANSCRIPT)
    assert not verify_quote("I will do that", TRANSCRIPT)   # four words
    assert not verify_quote("", TRANSCRIPT)


def test_short_but_specific_quotes_are_kept():
    """The floor must not reject genuinely short action items."""
    assert verify_quote("we still need someone to book the hall", TRANSCRIPT)


# --- null-ish values ------------------------------------------------------

@pytest.mark.parametrize("value", [
    None, "", "  ", "null", "None", "N/A", "unassigned", "Unassigned",
    "TBD", "someone", "the team", "not specified", "soon", "ASAP",
])
def test_placeholder_values_become_null(value):
    """A schema constrains shape, not content.

    The field is declared nullable and the prompt asks for null, and the
    model still returns "soon" for a deadline nobody stated -- because a
    string is a valid string. Storing that would put a fabricated value in
    a column that otherwise means something.
    """
    assert _clean(value) is None


@pytest.mark.parametrize("value,expected", [
    ("Karen", "Karen"),
    ("  the treasurer  ", "the treasurer"),
    ("before Friday", "before Friday"),
    ("next meeting", "next meeting"),
])
def test_real_values_are_kept(value, expected):
    assert _clean(value) == expected


def test_null_ish_matching_is_not_overeager():
    """"Unassigned" is a placeholder; "Naomi" is a person."""
    assert _clean("Naomi") == "Naomi"
    assert _clean("Team Lead Sarah") == "Team Lead Sarah"


# --- locating a quote in time ---------------------------------------------

def test_quote_is_traced_to_its_segment():
    segments = [
        seg("Let us move on to the budget.", 10.0, 13.0),
        seg("Karen can you circulate the revised budget before Friday?", 13.0, 18.0),
        seg("Yes I will do that.", 18.0, 20.0),
    ]
    start, end = locate("Karen can you circulate the revised budget before Friday?", segments)
    assert (start, end) == (13.0, 18.0)


def test_quote_spanning_two_segments():
    segments = [
        seg("We still need someone to book the hall", 30.0, 33.0),
        seg("for the AGM next month.", 33.0, 35.0),
    ]
    start, end = locate(
        "We still need someone to book the hall for the AGM next month.", segments)
    assert (start, end) == (30.0, 35.0)


def test_quote_straddling_a_boundary_without_containment():
    """The case that put a task at 0:00 which was said 26 seconds in.

    The quote runs across two segments, and each segment holds words the
    quote does not, so neither containment test fires.
    """
    segments = [
        seg("Right, let us go through the actions from the search rewrite.", 0.0, 3.3),
        seg("but nothing is decided yet. One more thing, Naomi, please "
            "update the on call", 26.4, 31.4),
        seg("wrote-a before the end of the month. Understood, I will do that.",
            31.4, 35.0),
    ]
    start, end = locate(
        "Naomi, please update the on call wrote-a before the end of the month.",
        segments)
    assert (start, end) == (26.4, 35.0), "must not fall back to the whole window"


def test_partial_overlap_does_not_match_on_common_words_alone():
    """Strategy 3 must not drag in a segment that merely shares 'the'."""
    segments = [
        seg("The budget for the year is on the intranet.", 5.0, 9.0),
        seg("Karen will book the hall for the AGM in March.", 9.0, 13.0),
    ]
    start, end = locate("Karen will book the hall for the AGM in March.", segments)
    assert start == 9.0 and end == 13.0


def test_unlocatable_quote_falls_back_to_the_window():
    """A slightly wide timestamp beats dropping a verified task."""
    segments = [seg("something else entirely", 5.0, 9.0)]
    assert locate("a quote that is not in here at all", segments) == (5.0, 9.0)


# --- deduplication --------------------------------------------------------

def test_identical_descriptions_are_duplicates():
    assert is_duplicate("Circulate the budget", ["Circulate the budget"])


def test_rephrased_descriptions_are_duplicates():
    """Overlapping windows show the model the same sentence twice, and it
    phrases the same commitment differently on each pass."""
    assert is_duplicate(
        "Circulate the revised budget to members",
        ["Circulate the revised budget to the members"])


def test_different_tasks_are_not_duplicates():
    assert not is_duplicate(
        "Book the hall for the AGM",
        ["Circulate the revised budget to the members"])


def test_empty_description_counts_as_duplicate():
    """Nothing useful can be stored, and it must not become a `seen` entry
    that then swallows every later task by matching nothing."""
    assert is_duplicate("", ["Book the hall"])
    assert is_duplicate("...", ["Book the hall"])


def test_first_task_is_never_a_duplicate():
    assert not is_duplicate("Book the hall for the AGM", [])

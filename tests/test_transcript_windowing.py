"""
Tests for grouping ASR segments into retrievable windows.

Like the chunker, this is pure deterministic logic with no model or network
involved, which is what makes it worth unit testing. The rules it enforces
(size bound, time bound, silence boundary, confidence floor) are each easy
to break by accident and hard to notice once broken -- a silently oversized
window just quietly degrades retrieval rather than failing.
"""

import pytest

from app.rag.transcripts import format_timestamp, window_segments


def seg(text, start, end, confidence=-0.3):
    """Build a segment. dict works because window_segments only uses ["key"]."""
    return {"text": text, "start_ts": start, "end_ts": end, "confidence": confidence}


def test_empty_input():
    assert window_segments([]) == []


def test_short_segments_are_merged_into_one_window():
    """The core purpose: 50-character segments are too small to embed alone."""
    segments = [seg(f"short sentence {i}.", i * 2.0, i * 2.0 + 2.0) for i in range(5)]
    windows = window_segments(segments)

    assert len(windows) == 1
    assert windows[0].segment_count == 5
    assert windows[0].start_ts == 0.0
    assert windows[0].end_ts == 10.0
    assert "short sentence 0." in windows[0].text
    assert "short sentence 4." in windows[0].text


def test_window_respects_the_character_limit():
    segments = [seg("x" * 100, i * 1.0, i * 1.0 + 1.0) for i in range(40)]
    windows = window_segments(segments, max_chars=500, overlap_chars=100,
                              max_span_seconds=10_000)
    assert len(windows) > 1
    # A little slack: a window is closed when the NEXT segment would exceed
    # the budget, so the last segment added can carry it slightly over.
    assert all(len(w.text) <= 700 for w in windows)


def test_long_silence_starts_a_new_window():
    """A pause is the transcript's paragraph break."""
    segments = [
        seg("first topic.", 0.0, 2.0),
        seg("still first topic.", 2.0, 4.0),
        # 30-second silence
        seg("completely different topic.", 34.0, 36.0),
    ]
    windows = window_segments(segments, gap_seconds=3.0)

    assert len(windows) == 2
    assert "first topic." in windows[0].text
    assert windows[1].text.startswith("completely different")


def test_small_gaps_do_not_split():
    segments = [
        seg("one.", 0.0, 2.0),
        seg("two.", 2.5, 4.0),   # 0.5s gap: normal speech
        seg("three.", 4.2, 6.0),
    ]
    assert len(window_segments(segments, gap_seconds=3.0)) == 1


def test_time_span_bound_applies_even_when_text_is_short():
    """A quiet stretch can stay under the character limit for a long time.

    "Somewhere in this half hour" is not a useful citation, so the span
    bound closes the window independently of how much was said.
    """
    # One brief remark every 30 seconds for 20 minutes: tiny text, huge span.
    segments = [seg("mm hmm.", i * 30.0, i * 30.0 + 1.0) for i in range(40)]
    windows = window_segments(segments, max_chars=10_000,
                              max_span_seconds=180.0, gap_seconds=1_000.0)

    assert len(windows) > 1
    assert all(w.end_ts - w.start_ts <= 400.0 for w in windows)


def test_low_confidence_segments_are_dropped():
    """Garbage ASR must not reach the index. See the note in transcripts.py."""
    segments = [
        seg("this is clear speech.", 0.0, 2.0, confidence=-0.2),
        seg("grbl mnh zzzt.", 2.0, 4.0, confidence=-2.5),
        seg("also clear speech.", 4.0, 6.0, confidence=-0.3),
    ]
    windows = window_segments(segments, min_confidence=-1.0)

    assert len(windows) == 1
    assert "grbl" not in windows[0].text
    assert windows[0].segment_count == 2


def test_all_segments_below_threshold_yields_nothing():
    segments = [seg("garbage", 0.0, 2.0, confidence=-3.0)]
    assert window_segments(segments, min_confidence=-1.0) == []


def test_missing_confidence_is_kept():
    """None means "unknown", which is not the same as "known to be bad"."""
    segments = [seg("text with no score.", 0.0, 2.0, confidence=None)]
    assert len(window_segments(segments, min_confidence=-1.0)) == 1


def test_consecutive_windows_overlap():
    segments = [seg(f"sentence {i} here.", i * 2.0, i * 2.0 + 2.0) for i in range(60)]
    windows = window_segments(segments, max_chars=300, overlap_chars=80,
                              max_span_seconds=10_000)

    assert len(windows) > 2
    # Overlap is applied in whole segments, so the time ranges of adjacent
    # windows must actually intersect.
    assert windows[1].start_ts < windows[0].end_ts


def test_overlap_never_prevents_progress():
    """Guard against the loop that emits the same window forever.

    If the carry-over were ever the entire window, the next window would
    start where the last one did and never advance.
    """
    segments = [seg("x" * 200, i * 1.0, i * 1.0 + 1.0) for i in range(20)]
    windows = window_segments(segments, max_chars=250, overlap_chars=240,
                              max_span_seconds=10_000)

    starts = [w.start_ts for w in windows]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts), "a window repeated its start time"


def test_overlap_must_be_smaller_than_the_window():
    with pytest.raises(ValueError):
        window_segments([seg("a", 0.0, 1.0)], max_chars=100, overlap_chars=100)


@pytest.mark.parametrize("seconds,expected", [
    (0, "0:00"),
    (9, "0:09"),
    (61, "1:01"),
    (599, "9:59"),
    (3600, "1:00:00"),
    (3661, "1:01:01"),
])
def test_timestamp_formatting(seconds, expected):
    assert format_timestamp(seconds) == expected

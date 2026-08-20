"""
Tests for cutting the audio stream into utterances.

The detector and the segmenter are tested separately on purpose. Whether a
particular 32ms window contains speech is a question about a neural model —
answering it belongs in evaluation against labelled audio, not a unit test.
WHERE to cut, given a sequence of speech/silence decisions, is ordinary
deterministic logic, and that is what these cover.

The buffer takes its detector as an argument, which is what makes that split
possible. Here the fake one calls any loud window speech, so the tests can
write "400ms of speech, then a second of silence" directly.
"""

import numpy as np
import pytest

from app.config import SAMPLE_RATE, VAD_WINDOW_SAMPLES
from app.vad import MIN_SPEECH_WINDOWS, SILENCE_WINDOWS, UtteranceBuffer

WINDOW_MS = VAD_WINDOW_SAMPLES / SAMPLE_RATE * 1000  # 32ms


def audio(*spans) -> np.ndarray:
    """Build a waveform from (is_speech, milliseconds) spans.

    Speech is a loud constant, silence is zeros. Crude, and exactly enough:
    the fake detector below only has to tell them apart.
    """
    pieces = []
    for is_speech, ms in spans:
        n = int(SAMPLE_RATE * ms / 1000)
        pieces.append(np.full(n, 0.9 if is_speech else 0.0, dtype=np.float32))
    return np.concatenate(pieces)


def loud_is_speech(window: np.ndarray) -> float:
    return 1.0 if np.abs(window).max() > 0.5 else 0.0


def buffer() -> UtteranceBuffer:
    return UtteranceBuffer(probability=loud_is_speech)


def feed(buf: UtteranceBuffer, samples: np.ndarray, frame_ms: int = 1000):
    """Push audio through in frames, as the network would."""
    step = int(SAMPLE_RATE * frame_ms / 1000)
    out = []
    for i in range(0, len(samples), step):
        out += buf.add(samples[i:i + step])
    return out


# --- nothing to emit -----------------------------------------------------

def test_pure_silence_emits_nothing():
    assert feed(buffer(), audio((False, 3000))) == []


def test_silence_is_not_buffered_forever():
    """Silence before speech must not accumulate.

    An hour of an empty room would otherwise grow the buffer until the
    process died, and then hand Whisper an hour of nothing.
    """
    buf = buffer()
    feed(buf, audio((False, 30_000)))
    assert len(buf._preroll) <= buf._preroll.maxlen


def test_brief_noise_is_discarded():
    """A cough or a chair is not an utterance.

    Transcribing a fragment of noise makes Whisper invent words for it, and
    the invented text then pollutes both the transcript and the index.
    """
    blip_ms = (MIN_SPEECH_WINDOWS - 4) * WINDOW_MS
    result = feed(buffer(), audio((False, 500), (True, blip_ms), (False, 1500)))
    assert result == []


# --- the ordinary case ---------------------------------------------------

def test_speech_then_pause_emits_one_utterance():
    result = feed(buffer(), audio((False, 300), (True, 2000), (False, 1500)))

    assert len(result) == 1
    assert result[0].reason == "silence"
    assert result[0].is_complete


def test_two_utterances_separated_by_a_pause():
    result = feed(buffer(), audio(
        (True, 1500), (False, 1200),   # pause well over the threshold
        (True, 1500), (False, 1200),
    ))
    assert len(result) == 2
    assert all(u.reason == "silence" for u in result)
    assert result[1].start > result[0].end


def test_short_gaps_do_not_split_an_utterance():
    """The gap between words is not the end of a sentence."""
    gap_ms = (SILENCE_WINDOWS - 8) * WINDOW_MS
    result = feed(buffer(), audio(
        (True, 800), (False, gap_ms), (True, 800), (False, 1500),
    ))
    assert len(result) == 1


def test_preroll_keeps_the_onset():
    """Speech is detected a beat after it starts, so keep what came before.

    Without this the first consonant is clipped and the recogniser guesses
    at the word.
    """
    result = feed(buffer(), audio((False, 1000), (True, 1000), (False, 1500)))

    assert len(result) == 1
    # The utterance must begin before the point speech was detected.
    assert result[0].start < 1.0


def test_trailing_silence_is_trimmed_but_not_all_of_it():
    """Keep a short tail so the last word does not sound cut off."""
    result = feed(buffer(), audio((True, 1000), (False, 3000)))

    assert len(result) == 1
    # 1s of speech plus a small tail -- not the full 3s of silence.
    assert 1.0 <= result[0].duration < 1.6


# --- boundaries and clocks ------------------------------------------------

def test_timestamps_track_the_sample_count():
    """The meeting clock comes from samples consumed, so it cannot drift."""
    result = feed(buffer(), audio(
        (False, 2000), (True, 1000), (False, 1500),
        (True, 1000), (False, 1500),
    ))
    assert len(result) == 2
    # Second utterance starts around 2s + 1s + 1.5s = 4.5s.
    assert 4.0 < result[1].start < 5.0


def test_frame_size_does_not_change_the_result():
    """Network framing must not affect where the audio is cut.

    The client chooses a frame size; the server decides the boundaries. If
    those were coupled, a client that changed its buffering would silently
    change transcription quality.
    """
    samples = audio((True, 1200), (False, 1200), (True, 1200), (False, 1200))
    a = [(round(u.start, 2), round(u.end, 2)) for u in feed(buffer(), samples, frame_ms=1000)]
    b = [(round(u.start, 2), round(u.end, 2)) for u in feed(buffer(), samples, frame_ms=137)]
    assert a == b and len(a) == 2


def test_long_speech_is_cut_by_the_cap_and_marked_incomplete():
    """A monologue still produces transcript rather than growing forever."""
    buf = buffer()
    # Patch the cap down so the test does not need 20 seconds of audio.
    import app.vad as vad
    original = vad.MAX_WINDOWS
    vad.MAX_WINDOWS = 40  # ~1.3s
    try:
        result = feed(buf, audio((True, 2000), (False, 1500)))
    finally:
        vad.MAX_WINDOWS = original

    assert len(result) >= 2
    assert result[0].reason == "length_cap"
    assert not result[0].is_complete       # the sentence probably continues
    # The final piece ends at the real pause, not at another cap: the cap is
    # only checked on speech windows, so it cannot pre-empt a boundary the
    # speaker was already arriving at.
    assert result[-1].reason == "silence"
    assert result[-1].is_complete
    # No gap: the speaker never stopped, so the next utterance resumes where
    # the last was cut. Treating the words after a cap as pre-roll would
    # lose them.
    assert result[1].start == pytest.approx(result[0].end, abs=0.05)


# --- end of stream --------------------------------------------------------

def test_flush_emits_speech_that_never_got_its_pause():
    """Whoever was talking when the audio stopped must not be dropped."""
    buf = buffer()
    assert feed(buf, audio((True, 2000))) == []   # no closing silence yet

    final = buf.flush()
    assert final is not None
    assert final.reason == "stream_end"
    # Complete: nothing more is coming, so there is nothing to wait for.
    assert final.is_complete


def test_flush_with_nothing_buffered():
    buf = buffer()
    feed(buf, audio((False, 1000)))
    assert buf.flush() is None

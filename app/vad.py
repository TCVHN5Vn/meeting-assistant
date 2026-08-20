"""
Cutting the audio stream into utterances.

THE PROBLEM

Audio arrives continuously and has to be cut somewhere before Whisper sees
it. The first version cut every five seconds on a stopwatch, which severs
words mid-syllable and hands the recogniser a fragment with no beginning.
Real output from that version, one cut landing mid-question:

    [ 5.0s] ...he assistant, what is the notice period?
    [10.0s] for a general meeting, let us see what it comes back with

That single design choice caused, directly, three separate problems
elsewhere: garbled words at every seam, a wake phrase misheard as "he", and
a question split across two chunks that then needed a timer to reassemble.
Two of those were patched where they surfaced. This module removes the cause.

THE FIX

Cut where the speaker stopped. Silero VAD gives a probability of speech for
each 32ms window; an utterance ends after enough consecutive silent windows.
The cut then lands in a gap rather than through a word.

WHY A NEURAL VAD AND NOT AN ENERGY THRESHOLD

Root-mean-square energy is the obvious cheap approach and it cannot tell
loud noise from speech. Measured on this machine, Silero returns p=0.0049 on
random noise at the same amplitude as speech -- an energy detector scores
that as loudly voiced. Meetings have chairs, doors, typing and coughs, all
of which are energetic and none of which are worth transcribing.

The cost of the better detector is close to nothing: 0.1ms per window,
roughly 322x faster than realtime on CPU.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from app.config import (
    SAMPLE_RATE,
    VAD_MAX_UTTERANCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_PREROLL_MS,
    VAD_SILENCE_MS,
    VAD_TAIL_MS,
    VAD_THRESHOLD,
    VAD_WINDOW_SAMPLES,
)


def _ms_to_windows(ms: float) -> int:
    """Convert milliseconds to whole VAD windows, at least one."""
    return max(1, round(ms * SAMPLE_RATE / 1000 / VAD_WINDOW_SAMPLES))


SILENCE_WINDOWS = _ms_to_windows(VAD_SILENCE_MS)
PREROLL_WINDOWS = _ms_to_windows(VAD_PREROLL_MS)
TAIL_WINDOWS = _ms_to_windows(VAD_TAIL_MS)
MIN_SPEECH_WINDOWS = _ms_to_windows(VAD_MIN_SPEECH_MS)
MAX_WINDOWS = _ms_to_windows(VAD_MAX_UTTERANCE_MS)


def _silero_detector():
    """Load Silero and return a window -> p(speech) function.

    Imported here rather than at module import: it pulls in torch, which
    takes seconds to load, and nothing should pay that just to read a
    constant out of this module.

    Silero is RECURRENT -- its hidden state carries across windows and must
    not be reset mid-stream. That state is what lets it distinguish a pause
    between words from the end of a sentence, which an isolated per-window
    classifier cannot do. Hence the state lives in the closure, one per
    connection.
    """
    import torch
    from silero_vad import load_silero_vad

    model = load_silero_vad()
    model.reset_states()

    def probability(window):
        return model(torch.from_numpy(window), SAMPLE_RATE).item()

    return probability


@dataclass
class Utterance:
    """One continuous stretch of speech, cut at the pauses around it."""
    audio: np.ndarray      # float32, mono, SAMPLE_RATE
    start: float           # seconds since the meeting began
    end: float

    # WHY the audio was cut here. This is why VAD is worth more than better
    # transcription alone -- it tells the layer above whether a sentence is
    # finished:
    #
    #   'silence'     the speaker stopped. The sentence is complete.
    #   'length_cap'  still talking when the cap hit. It probably continues.
    #   'stream_end'  the audio ran out. Nothing more is coming either way.
    #
    # Sprint 3 could not distinguish these and had to assume every question
    # might continue, so it waited five seconds for every one.
    reason: str

    @property
    def is_complete(self) -> bool:
        """True when nothing more will be added to this utterance."""
        return self.reason in ("silence", "stream_end")

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


class UtteranceBuffer:
    """Streams audio in, emits complete utterances.

    One per connection, because the detector carries state across the stream.
    """

    def __init__(self, probability=None):
        """`probability` maps one window of audio to p(speech).

        Injectable so the segmentation state machine can be tested without
        loading a model: the interesting logic here is when to cut, not how
        speech is detected, and those deserve to be tested separately.
        """
        self._probability = probability or _silero_detector()

        # Samples received but not yet a whole window.
        self._leftover = np.zeros(0, dtype=np.float32)

        # Windows kept from before speech began, so the first consonant is
        # not clipped. A bounded deque discards the older ones for free.
        self._preroll: deque = deque(maxlen=PREROLL_WINDOWS)

        self._current: list[np.ndarray] = []
        self._silence_run = 0
        self._in_speech = False

        # Windows of ACTUAL SPEECH in the current utterance. Counted apart
        # from len(self._current), which also holds pre-roll and trailing
        # silence -- so a 250ms cough plus 300ms of pre-roll would otherwise
        # clear a 400ms minimum and be sent off for transcription.
        self._speech_windows = 0

        # Total samples consumed on this connection. THE MEETING CLOCK.
        # Deriving time from the sample count is exact, and it replaces the
        # running offset the old code accumulated from chunk durations --
        # which could drift, and did.
        self._consumed = 0
        self._start_sample = 0

    # ---- public API -----------------------------------------------------

    def add(self, pcm: np.ndarray) -> list[Utterance]:
        """Feed audio in. Returns any utterances that completed."""
        samples = np.concatenate([self._leftover, pcm]) if self._leftover.size else pcm
        window_count = len(samples) // VAD_WINDOW_SAMPLES
        self._leftover = samples[window_count * VAD_WINDOW_SAMPLES:]

        finished: list[Utterance] = []
        for i in range(window_count):
            window = samples[i * VAD_WINDOW_SAMPLES:(i + 1) * VAD_WINDOW_SAMPLES]
            utterance = self._consume(window, self._probability(window) >= VAD_THRESHOLD)
            if utterance is not None:
                finished.append(utterance)
        return finished

    def flush(self) -> Utterance | None:
        """Emit whatever is still buffered. Called when the stream ends.

        Without this, the last thing anyone said before hanging up is
        silently dropped -- it never got its closing silence.
        """
        if not self._in_speech:
            return None
        return self._emit(reason="stream_end", trim_tail=False)

    # ---- state machine --------------------------------------------------

    def _consume(self, window: np.ndarray, is_speech: bool) -> Utterance | None:
        self._consumed += VAD_WINDOW_SAMPLES

        if not self._in_speech:
            if not is_speech:
                # Silence before anyone speaks is not worth transcribing and
                # not worth keeping -- except the last few windows, in case
                # speech starts in the next one.
                self._preroll.append(window)
                return None

            # Speech begins. Adopt the pre-roll so the onset is intact.
            self._current = list(self._preroll)
            self._current.append(window)
            self._start_sample = self._consumed - len(self._current) * VAD_WINDOW_SAMPLES
            self._preroll.clear()
            self._in_speech = True
            self._silence_run = 0
            self._speech_windows = 1
            return None

        self._current.append(window)
        if is_speech:
            self._silence_run = 0
            self._speech_windows += 1
        else:
            self._silence_run += 1

        if self._silence_run >= SILENCE_WINDOWS:
            return self._emit(reason="silence", trim_tail=True)

        # Checked only on a speech window. During a pause the buffer is about
        # to be cut at the pause anyway, and letting the cap fire first would
        # split an utterance a few windows before its natural boundary --
        # producing a spurious "unfinished" cut right where the speaker
        # actually stopped.
        if is_speech and self._speech_windows >= MAX_WINDOWS:
            # Cut mid-flow because the speaker has not stopped. Flagged as
            # unfinished, and the next utterance continues immediately from
            # here rather than waiting for a fresh onset -- otherwise the
            # first words after the cut would be treated as pre-roll and the
            # sentence would lose them.
            utterance = self._emit(reason="length_cap", trim_tail=False)
            self._in_speech = True
            self._current = []
            self._start_sample = self._consumed
            self._silence_run = 0
            self._speech_windows = 0
            return utterance

        return None

    def _emit(self, reason: str, trim_tail: bool) -> Utterance | None:
        windows = self._current

        if trim_tail:
            # Drop the detected silence but keep a short tail, so the final
            # word does not sound clipped to the recogniser.
            keep = max(len(windows) - self._silence_run + TAIL_WINDOWS, 1)
            windows = windows[:keep]

        speech_windows = self._speech_windows
        self._reset_speech_state()

        if speech_windows < MIN_SPEECH_WINDOWS:
            # A cough, a chair, a single click. Transcribing a fragment of
            # noise makes Whisper confidently invent words for it, and the
            # invented text then pollutes both the transcript and the index.
            #
            # Measured against speech windows, not against len(windows):
            # that count includes pre-roll and trailing silence, which are
            # padding and are not evidence that anyone said anything.
            return None

        audio = np.concatenate(windows)
        start = self._start_sample / SAMPLE_RATE
        return Utterance(
            audio=audio,
            start=start,
            end=start + len(audio) / SAMPLE_RATE,
            reason=reason,
        )

    def _reset_speech_state(self) -> None:
        self._current = []
        self._in_speech = False
        self._silence_run = 0
        self._speech_windows = 0
        self._preroll.clear()

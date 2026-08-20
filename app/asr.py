"""
The ASR layer: audio in, timestamped text segments out.

Both entry points -- the batch script and the streaming server -- come
through here, so there is one place that decides how the model is loaded
and what a "segment" looks like.
"""

import io
from dataclasses import dataclass
from typing import Iterator

from faster_whisper import WhisperModel

from app.config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL_SIZE

# Module-level cache. Loading Whisper means allocating memory and reading
# weights off disk -- expensive enough that doing it per request would
# dominate the runtime. Loading it once and reusing it is the whole reason
# the streaming version is a long-lived server rather than a script.
_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    """Load the model on first use, then hand back the same instance."""
    global _model
    if _model is None:
        print(f"Loading Whisper '{WHISPER_MODEL_SIZE}' "
              f"({WHISPER_DEVICE}, {WHISPER_COMPUTE_TYPE})...")
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print("Whisper ready.")
    return _model


@dataclass
class Segment:
    """One timestamped piece of speech.

    A plain dataclass rather than passing faster-whisper's own objects
    around: it keeps library types at the edge of the system, so swapping
    Whisper for another ASR engine later means rewriting this file only.
    """
    text: str
    start: float
    end: float
    confidence: float


def _to_segments(raw_segments) -> Iterator[Segment]:
    for s in raw_segments:
        text = s.text.strip()
        if not text:
            continue  # Whisper emits empty segments for silence.

        # faster-whisper gives no clean 0-1 confidence. avg_logprob is the
        # closest proxy: an average log-probability, so 0 is perfectly
        # confident and more negative is less confident. Roughly, above
        # -0.5 is solid, below -1.0 is usually garbage. We store it raw
        # and threshold later, at the point where it matters -- deciding
        # whether a chunk is trustworthy enough to feed into retrieval.
        yield Segment(
            text=text,
            start=s.start,
            end=s.end,
            confidence=s.avg_logprob,
        )


def transcribe_path(audio_path: str) -> tuple[Iterator[Segment], object]:
    """Transcribe a file on disk. Used by the batch script."""
    raw, info = get_model().transcribe(audio_path, beam_size=5)
    return _to_segments(raw), info


def transcribe_bytes(audio_bytes: bytes) -> tuple[list[Segment], float]:
    """Transcribe an in-memory audio chunk. Used by the streaming server.

    Returns (segments, duration_seconds).

    Returns a LIST, not a generator. faster-whisper's segment generator is
    lazy -- transcription only actually happens as you iterate it. Since
    the server calls this on a worker thread specifically to keep the CPU
    work off the event loop, we must force the work to complete HERE,
    inside the thread. Return a lazy generator instead and the real
    computation happens back on the event loop when the caller iterates,
    which defeats the entire point.

    The duration is returned because the caller needs it, and this is the
    only place that knows it. Every chunk is transcribed in isolation, so
    the timestamps below are relative to the START OF THIS CHUNK -- the
    fortieth second of a meeting comes back labelled 0.0. Whoever is
    assembling a whole meeting has to add a running offset, and the
    duration of each chunk is what advances that offset.
    """
    raw, info = get_model().transcribe(io.BytesIO(audio_bytes), beam_size=5)
    # info.duration is the length of the decoded audio in seconds. Read it
    # AFTER forcing the segment list: faster-whisper populates parts of
    # `info` during transcription, not before it.
    segments = list(_to_segments(raw))
    return segments, info.duration

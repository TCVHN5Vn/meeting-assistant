"""
The ASR layer: audio in, timestamped text segments out.

Both entry points -- the batch script and the streaming server -- come
through here, so there is one place that decides how the model is loaded
and what a "segment" looks like.
"""

import threading

from dataclasses import dataclass
from typing import Iterator

from faster_whisper import WhisperModel

from app.config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL_SIZE

# Module-level cache. Loading Whisper means allocating memory and reading
# weights off disk -- expensive enough that doing it per request would
# dominate the runtime. Loading it once and reusing it is the whole reason
# the streaming version is a long-lived server rather than a script.
_model: WhisperModel | None = None
_model_lock = threading.Lock()


def get_model() -> WhisperModel:
    """Load the model on first use, then hand back the same instance."""
    global _model
    # Double-checked locking. Everything that loads a model now runs on a
    # worker thread, and with several participants two threads can reach
    # this at the same moment -- which loads Whisper twice, wastes the
    # memory, and in practice crashed the process. The cheap check outside
    # the lock keeps the common path (already loaded) free of contention.
    if _model is not None:
        return _model
    with _model_lock:
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


def transcribe_audio(audio) -> list[Segment]:
    """Transcribe a float32 mono waveform at 16 kHz. Used by the server.

    Takes a numpy array rather than encoded bytes. The server holds a
    continuous stream of raw samples and cuts it at pauses, so re-encoding
    each utterance to WAV purely to hand it back to a decoder would be work
    done for nobody -- faster-whisper reads the array directly.

    Returns a LIST, not a generator. faster-whisper's segment generator is
    lazy: transcription only really happens as you iterate it. Since the
    server calls this on a worker thread specifically to keep the CPU work
    off the event loop, the work must complete HERE, inside the thread.
    Return a lazy generator and the real computation happens back on the
    event loop when the caller iterates -- defeating the entire point.
    """
    raw, _info = get_model().transcribe(audio, beam_size=5)
    return list(_to_segments(raw))

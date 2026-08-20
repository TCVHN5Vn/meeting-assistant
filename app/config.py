"""
One place for every path and model name in the project.

Why bother, instead of writing "meeting_assistant.db" in five files?
Because the moment two files disagree about where the database lives,
you get a bug that looks like "my data vanished" but is really "you
wrote to a different file than you read from". Constants that more
than one module needs belong in exactly one module.
"""

from pathlib import Path

# __file__ is this file. .parent is app/, .parent.parent is the project
# root. Deriving paths this way means the scripts work no matter which
# directory you happen to run them from -- unlike a bare relative path
# like "data/foo.db", which silently depends on your shell's cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

DB_PATH = DATA_DIR / "meeting_assistant.db"

# The FAISS index is a binary file of raw vectors. It deliberately lives
# OUTSIDE the database -- see app/rag/store.py for why.
FAISS_INDEX_PATH = DATA_DIR / "documents.faiss"

# --- ASR ---------------------------------------------------------------
# "base" is the CPU-friendly starting point. int8 quantization trades a
# little accuracy for much faster inference on CPU.
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# --- Voice activity detection --------------------------------------------
# Audio arrives as a continuous stream and has to be cut somewhere before
# Whisper sees it. Cutting on a stopwatch severs words mid-syllable; cutting
# at pauses does not. See app/vad.py.
SAMPLE_RATE = 16000
# Silero requires exactly this many samples per call at 16 kHz.
VAD_WINDOW_SAMPLES = 512
# p(speech) at or above this counts as speech.
VAD_THRESHOLD = 0.5
# Silence needed to declare an utterance finished. Too short and it cuts at
# the pause between clauses; too long and every answer waits for it. 700ms
# is comfortably longer than an inter-word gap and shorter than a turn gap.
VAD_SILENCE_MS = 700
# Ignore blips: a cough or a door is not an utterance.
VAD_MIN_SPEECH_MS = 400
# Hard cap, so an uninterrupted monologue still produces transcript rather
# than growing a buffer forever. An utterance cut by this cap is marked as
# unfinished -- see Utterance.ended_on_silence.
VAD_MAX_UTTERANCE_MS = 20000
# Audio kept from just BEFORE speech was detected. Silero reports speech a
# beat after it starts, so without this the first consonant is clipped.
VAD_PREROLL_MS = 300
# Silence kept on the end, so the last word does not sound truncated to the
# recogniser.
VAD_TAIL_MS = 200

# --- Embeddings --------------------------------------------------------
# all-MiniLM-L6-v2: 384-dimensional vectors, ~90MB, fast on CPU. Small
# and old-ish by 2026 standards, but it is the reference model everyone
# knows, it runs locally for free, and it is more than good enough to
# demonstrate that retrieval works.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Chunking ----------------------------------------------------------
# Measured in characters, not tokens -- simpler, and precise enough at
# this scale. See app/rag/chunking.py for why these numbers.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Transcript windowing ------------------------------------------------
# ASR segments average ~50 characters -- far too small to embed one at a
# time. They are grouped into windows before indexing. See
# app/rag/transcripts.py for what each of these controls.
TRANSCRIPT_WINDOW_CHARS = 1000
TRANSCRIPT_OVERLAP_CHARS = 200
# A window covering half an hour is not "a moment in the meeting" any more,
# however few words were said in it.
TRANSCRIPT_MAX_SPAN_SECONDS = 180.0
# A pause longer than this is treated as a topic boundary -- the transcript
# equivalent of a paragraph break.
TRANSCRIPT_GAP_SECONDS = 3.0
# avg_logprob floor for indexing. 0 is perfectly confident, more negative is
# less. -1.0 drops only clearly failed transcription; raise it toward -0.7
# for a cleaner but smaller index.
MIN_ASR_CONFIDENCE = -1.0

# --- Live meeting Q&A ----------------------------------------------------
# How the assistant decides to speak up mid-meeting. See app/rag/questions.py
# for why 'wake' is the default rather than 'questions'.
#   'wake'      only when addressed by name  (default)
#   'questions' anything question-shaped     (noisy; good for demos)
#   'off'       explicit ask_query events only
AUTO_ANSWER_MODE = "wake"
ASSISTANT_NAME = "assistant"
# Greetings that may precede the name. The odd-looking entries -- "he",
# "hay", "hei" -- are not typos: they are what speech recognition actually
# returns for "hey" often enough to matter. See app/rag/questions.py.
WAKE_GREETINGS = ("hey", "hi", "hello", "ok", "okay", "he", "hay", "hei")
# After a wake phrase is heard, how long to keep listening before answering.
# Audio is sliced every 5 seconds regardless of whether anyone is mid-
# sentence, so a spoken question is routinely cut in half by a chunk
# boundary. Slightly longer than one chunk, so exactly one continuation
# chunk can land. See LiveSession._note_question in app/server.py.
QUESTION_CONTINUATION_SECONDS = 6.0

# How much of the meeting so far to put in the prompt verbatim, alongside
# whatever retrieval found. See build_live_user_prompt in app/llm/prompts.py.
LIVE_CONTEXT_SECONDS = 240.0

# --- Authentication ------------------------------------------------------
JWT_ALGORITHM = "HS256"
# Short enough that a leaked token stops working the same day, long enough
# that a meeting does not end with everyone logged out. There is no refresh
# token here, so this is the whole session lifetime.
JWT_TTL_HOURS = 12
# How long a WebSocket may stay open before authenticating. See ws_meeting.
WS_AUTH_TIMEOUT_SECONDS = 5.0

# --- LLM ---------------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b-instruct"


def ensure_dirs() -> None:
    """Create the runtime directories if they are missing.

    data/ is in .gitignore, so a fresh clone of this repo will not have
    it. Rather than making people read the README to find that out, we
    just create it on demand.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

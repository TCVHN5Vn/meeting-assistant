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

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

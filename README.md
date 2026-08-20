# Multimodal Meeting Assistant

A meeting assistant that transcribes audio in real time, stores structured
transcripts, and answers questions about company documents using retrieval-
augmented generation — **running entirely on the local machine**. No API
keys, no per-token cost, and no meeting audio or company document ever
leaves the device.

Built as a study of how ASR, RAG, an LLM, and a database fit together in one
system, rather than as a wrapper around a hosted API.

```
   microphone / audio file
             │
             ▼
   ┌──────────────────┐   WebSocket    ┌──────────────────┐
   │  client (audio   │───────────────▶│  FastAPI server  │
   │  chunks, 5s)     │◀───────────────│                  │
   └──────────────────┘  transcript    └────────┬─────────┘
                            chunks              │
                                                ▼
                                    ┌───────────────────────┐
                                    │  faster-whisper (ASR) │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                       ┌─────────────────┐
                                       │  SQLite         │
                                       │  transcripts,   │
                                       │  documents,     │
                                       │  chunk metadata │
                                       └─────────────────┘

   company documents                          question
       (.md/.txt/.pdf)                            │
             │                                    ▼
             ▼                          ┌──────────────────┐
   ┌────────────────────┐               │  embed the query │
   │ chunk → embed      │               └────────┬─────────┘
   │ (MiniLM, 384-dim)  │                        │
   └─────────┬──────────┘                        ▼
             │                          ┌──────────────────┐
             ▼                          │  FAISS: top-k    │
   ┌────────────────────┐   ◀───────────│  nearest vectors │
   │ FAISS index (disk) │               └────────┬─────────┘
   └────────────────────┘                        │
                                                 ▼
                                       ┌───────────────────┐
                                       │ vector_id → text  │
                                       │ (SQLite lookup)   │
                                       └────────┬──────────┘
                                                ▼
                                       ┌───────────────────┐
                                       │ Ollama (qwen2.5)  │
                                       │ grounded answer   │
                                       │ + citations       │
                                       └───────────────────┘
```

## Status

| Sprint | Scope | State |
|--------|-------|-------|
| 1 | Batch + streaming transcription, WebSocket, transcript storage | **Done** |
| 2 | Chunking, embeddings, FAISS index, search API, grounded Q&A | **Done** |
| 3 | Real-time Q&A during a live meeting | Not started |
| 4 | Action-item extraction into structured tasks | Not started |
| 5 | Authentication, frontend | Not started |

## Stack

| Concern | Choice | Why |
|---|---|---|
| ASR | faster-whisper (`base`, int8) | 4× faster than reference Whisper on CPU, and releases the GIL during inference |
| API | FastAPI + WebSockets | Native async; WebSocket is the right transport for a continuous audio stream |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) | Runs locally on CPU in milliseconds; small enough to be practical, good enough to be useful |
| Vector search | FAISS `IndexFlatIP` | Exact search, no approximation error at this corpus size |
| LLM | Ollama + `qwen2.5:7b-instruct` | Local, free, private; strong at instruction-following and structured output |
| Database | SQLite | Zero setup; the schema is written so a Postgres migration is mechanical |

## Setup

```bash
# System dependencies
brew install python@3.12 ffmpeg ollama
brew services start ollama
ollama pull qwen2.5:7b-instruct     # ~4.7 GB

# Python environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model weights download automatically on first use: Whisper `base` (~150 MB)
and MiniLM (~90 MB). Both are cached afterwards.

## Usage

All commands run from the project root, with the venv active.

### Transcribe a file

```bash
python -m scripts.transcribe_file sample_data/audio/short_recording.m4a
```

### Stream audio to the live server

Two terminals.

```bash
# Terminal 1 — the server. Wait for "Server ready."
uvicorn app.server:app --reload

# Terminal 2 — a simulated live microphone
python -m scripts.client_simulator sample_data/audio/short_recording.m4a
```

The client chops a finished recording into 5-second slices and sends them
with a real 5-second delay between each, so the server experiences it
exactly as it would a live microphone. Responses should arrive roughly 5
seconds apart — that pacing is the proof it is streaming and not batching.

### Build the document index

```bash
python -m scripts.ingest_documents                 # sample_data/documents
python -m scripts.ingest_documents docs/ --force   # a different folder, forced
```

### Ask a question

```bash
python -m scripts.ask "what is the deployment process?"
python -m scripts.ask "how much annual leave do I get?" --no-llm
```

`--no-llm` prints the retrieved chunks without generating an answer. When an
answer looks wrong, this tells you in one command whether the problem is
**retrieval** (the right text was never found) or **generation** (it was
found and the model still got it wrong). Those have different fixes.

### HTTP API

```bash
curl localhost:8000/api/v1/index/stats

curl -X POST localhost:8000/api/v1/search \
  -H 'content-type: application/json' \
  -d '{"query": "on-call compensation"}'

curl -X POST localhost:8000/api/v1/ask \
  -H 'content-type: application/json' \
  -d '{"question": "who approves lifting a deploy freeze?"}'
```

Interactive docs at `http://localhost:8000/docs`.

## Tests

```bash
pytest tests/ -q
```

The suite covers the chunker and the database schema — the deterministic
parts. Model behaviour is not unit-tested; that belongs in evaluation
against a labelled question set, which is a separate discipline.

## Layout

```
app/
  config.py            paths, model names, chunk sizes — one source of truth
  db.py                schema and all SQL for the core tables
  asr.py               Whisper loading and transcription
  server.py            FastAPI app: WebSocket + REST endpoints
  rag/
    chunking.py        splitting text into overlapping windows
    embeddings.py      text → 384-dim vectors
    store.py           FAISS index: build, save, load, search
    loaders.py         reading .txt / .md / .pdf, content hashing
    ingest.py          the full ingestion pipeline
    retrieve.py        question → ranked chunks
    qa.py              retrieve → prompt → generate → answer + sources
  llm/
    ollama_client.py   HTTP client for the local model
    prompts.py         prompt construction, kept as reviewable code
scripts/               command-line entry points
sample_data/documents/ example corpus
tests/                 unit tests
data/                  database + FAISS index (gitignored, regenerable)
```

Design decisions and their tradeoffs are written up in
[docs/design-notes.md](docs/design-notes.md).

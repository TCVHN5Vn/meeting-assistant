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
   ┌──────────────────┐   audio chunks ┌──────────────────┐
   │  client          │───────────────▶│  FastAPI server  │
   │                  │◀───────────────│                  │
   └──────────────────┘  transcript,   └────────┬─────────┘
                         answers               │
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

   company documents         meeting transcripts          question
    (.md/.txt/.pdf)          (from the DB above)              │
          │                          │                        ▼
          ▼                          ▼              ┌──────────────────┐
   ┌─────────────┐        ┌────────────────────┐    │  embed the query │
   │ chunk by    │        │ window by silence  │    └────────┬─────────┘
   │ characters  │        │ + time + confidence│             │
   └──────┬──────┘        └─────────┬──────────┘             ▼
          │                         │               ┌──────────────────┐
          └───────────┬─────────────┘               │  FAISS: top-k    │
                      ▼                   ◀─────────│  nearest vectors │
            ┌───────────────────┐                   └────────┬─────────┘
            │  embed (MiniLM)   │                            │
            │  ONE shared index │                            ▼
            └───────────────────┘                  ┌───────────────────┐
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

Documents and transcripts share **one** index, so a single question can be
answered from both — "does what we agreed in the meeting match the written
policy?" is one search, not two. Transcript citations carry timestamps you
can seek to: `Special General Meeting @ 33:46-35:05`.

## Status

| Sprint | Scope | State |
|--------|-------|-------|
| 1 | Batch + streaming transcription, WebSocket, transcript storage | **Done** |
| 2 | Chunking, embeddings, FAISS index, search API, grounded Q&A | **Done** |
| 2.5 | Transcripts indexed alongside documents, in one shared index | **Done** |
| 3 | Real-time Q&amp;A during a live meeting | **Done** |
| — | Voice-activity detection: cut audio at pauses, not on a clock | **Done** |
| 4 | Action-item extraction into structured tasks | **Done** |
| 5a | JWT authentication, meeting ownership | **Done** |
| 5b | Frontend | Not started |

## Stack

| Concern | Choice | Why |
|---|---|---|
| ASR | faster-whisper (`base`, int8) | 4× faster than reference Whisper on CPU, and releases the GIL during inference |
| Segmentation | Silero VAD | Cuts audio at pauses rather than on a clock; rejects applause and noise that an energy threshold calls speech |
| API | FastAPI + WebSockets | Native async; WebSocket is the right transport for a continuous audio stream |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) | Runs locally on CPU in milliseconds; small enough to be practical, good enough to be useful |
| Vector search | FAISS `IndexFlatIP` | Exact search, no approximation error at this corpus size |
| LLM | Ollama + `qwen2.5:7b-instruct` | Local, free, private; strong at instruction-following and structured output |
| Database | SQLite | Zero setup; the schema is written so a Postgres migration is mechanical |

## Authentication

Every `/api/v1` endpoint and the WebSocket require a bearer token. Create an
account from the machine running the server:

```bash
python -m scripts.create_user you@example.com "Your Name"
python -m scripts.create_user --list
```

There is deliberately **no `/register` endpoint** — open registration is a
product decision, and shipping one by default decides it silently.

```bash
# Get a token
curl -X POST localhost:8000/api/v1/auth/login \
  -H 'content-type: application/json' \
  -d '{"email": "you@example.com", "password": "..."}'

export MEETING_ASSISTANT_TOKEN=<the access_token>
```

Set a signing key in production, or tokens are invalidated on every restart:

```bash
export MEETING_ASSISTANT_SECRET=$(python -c 'import secrets; print(secrets.token_hex(32))')
```

A key shorter than 32 bytes is refused at startup rather than warned about.

The WebSocket authenticates with its **first message**, not a query
parameter — a browser cannot set headers on a WebSocket, and `?token=` would
write a live credential into access logs and browser history. See §20 of the
design notes.

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
python -m scripts.client_simulator sample_data/audio/short_recording.m4a \
  --email you@example.com
```

The client sends raw PCM in 1-second frames, in real time, so the server
experiences it exactly as it would a live microphone. **The frames are not
the transcription unit** — the server buffers them and cuts at pauses using
voice-activity detection, so transcript arrives when someone stops speaking
rather than on a fixed cadence. On the same 90 seconds of meeting audio that
halves the number of low-confidence segments (see §18 of the design notes).

### Extract action items

```bash
python -m scripts.extract_tasks --list
python -m scripts.extract_tasks <meeting_id>
python -m scripts.extract_tasks <meeting_id> --show   # re-read without re-running
```

Every task carries the verbatim transcript line it came from and the moment
it was said, because a task list you cannot check against the recording is a
list you have to take on faith. The quote is also **verified in code** —
if the sentence is not in the transcript, the task is discarded.

That guard matters more than it looks. On six windows of a governance
discussion containing no action items at all, a naive prompt returned **33**
of them; the prompt in [app/tasks.py](app/tasks.py) returned **0**. See §19
of the design notes.

### Ask a question during the meeting

Two ways. Say it out loud, if the recording contains a wake phrase:

```bash
python -m scripts.client_simulator sample_data/audio/wake_demo.m4a
```

Or send it as an event partway through the stream:

```bash
python -m scripts.client_simulator sample_data/audio/special_general_meeting.m4a \
  --ask "what are the three main goals of this meeting?" --after 3
```

What to watch for: **transcript lines continue arriving while the answer is
being written.** Generation takes 10–20 seconds on a local model, and the
meeting does not pause for it.

```
│ [   20.0s] and Juana Arrujo-Keypert and Administration Lee.
[   23.0s] Thank you.
The three main goals of this meeting are: ...
```

By default the assistant only speaks when addressed — `"Hey assistant, …"`.
A meeting is full of questions people ask each other, and answering all of
them makes an assistant that gets switched off. Set `AUTO_ANSWER_MODE` in
[app/config.py](app/config.py) to `"questions"` to answer anything
question-shaped, or `"off"` for explicit events only.

### Build the document index

```bash
python -m scripts.ingest_documents                 # sample_data/documents
python -m scripts.ingest_documents docs/ --force   # a different folder, forced
```

### Index meeting transcripts

```bash
python -m scripts.index_transcripts --list        # what is available
python -m scripts.index_transcripts               # everything
python -m scripts.index_transcripts <meeting_id>  # one meeting
```

Raw ASR segments average ~50 characters — far too small to embed one at a
time. They are grouped into windows bounded by size, by elapsed time, and by
silence, with segments below a confidence floor dropped. A 53-minute meeting
of 723 segments becomes 52 windows.

### Extract action items

```bash
python -m scripts.extract_tasks --list
python -m scripts.extract_tasks <meeting_id>
python -m scripts.extract_tasks <meeting_id> --show   # re-read without re-running
```

Every task carries the verbatim transcript line it came from and the moment
it was said, because a task list you cannot check against the recording is a
list you have to take on faith. The quote is also **verified in code** —
if the sentence is not in the transcript, the task is discarded.

That guard matters more than it looks. On six windows of a governance
discussion containing no action items at all, a naive prompt returned **33**
of them; the prompt in [app/tasks.py](app/tasks.py) returned **0**. See §19
of the design notes.

### Ask a question

```bash
python -m scripts.ask "what is the deployment process?"
python -m scripts.ask "how much annual leave do I get?" --no-llm
python -m scripts.ask "what was decided about finances?" --transcripts
python -m scripts.ask "what does the policy require?" --documents
```

`--transcripts` and `--documents` scope the search, which is how you ask
what was **said** separately from what the policy **says** — and then compare
them.

`--no-llm` prints the retrieved chunks without generating an answer. When an
answer looks wrong, this tells you in one command whether the problem is
**retrieval** (the right text was never found) or **generation** (it was
found and the model still got it wrong). Those have different fixes.

### HTTP API

All calls need `Authorization: Bearer $MEETING_ASSISTANT_TOKEN`.

```bash
curl -H "Authorization: Bearer $MEETING_ASSISTANT_TOKEN" \
  localhost:8000/api/v1/index/stats

curl -X POST localhost:8000/api/v1/search \
  -H 'content-type: application/json' \
  -d '{"query": "on-call compensation"}'

curl -X POST localhost:8000/api/v1/ask \
  -H 'content-type: application/json' \
  -d '{"question": "who approves lifting a deploy freeze?"}'

# Scope to one meeting
curl -X POST localhost:8000/api/v1/search \
  -H 'content-type: application/json' \
  -d '{"query": "reserves", "meeting_id": "<uuid>"}'

# Make a meeting searchable
curl -X POST localhost:8000/api/v1/meetings/<uuid>/index

# Action items
curl -X POST localhost:8000/api/v1/meetings/<uuid>/tasks   # extract
curl localhost:8000/api/v1/meetings/<uuid>/tasks           # read
curl -X PATCH localhost:8000/api/v1/tasks/<uuid> \
  -H 'content-type: application/json' -d '{"status": "done"}'
```

Interactive docs at `http://localhost:8000/docs`.

### WebSocket events

```
client -> server   binary frame                         an audio chunk
client -> server   {"event":"ask_query","data":{...}}    a typed question

server -> client   transcript_chunk   text as it is recognised
server -> client   qa_listening       heard a question, waiting for the rest
server -> client   qa_started         sources, sent before generation begins
server -> client   qa_delta           answer fragments, as they are generated
server -> client   qa_response        the complete answer
server -> client   qa_busy            already answering something else
server -> client   qa_error / error
```

`qa_started` goes out before the first token because retrieval takes
milliseconds and generation takes seconds — the client can show what is
being read from while the model is still writing.

## Tests

```bash
pytest tests/ -q
```

121 tests covering the document chunker, the transcript windower, question
detection, audio segmentation, action-item verification, authentication,
and the schema — the deterministic parts. Model behaviour is not
unit-tested; that belongs in evaluation against a labelled question set,
which is a separate discipline.

The windowing tests earned their place immediately: one of them caught that
overlap was being applied across silence boundaries, which is exactly where
it should not be. See §12 of the design notes.

Most of the detection tests are about what must **not** fire. Precision
matters more than recall for a thing that interrupts a meeting.

## Layout

```
app/
  config.py            paths, model names, chunk sizes — one source of truth
  auth.py              password hashing, tokens, access control
  vad.py               cutting the audio stream into utterances at pauses
  tasks.py             extracting action items, and verifying they are real
  db.py                schema and all SQL for the core tables
  asr.py               Whisper loading and transcription
  server.py            FastAPI app: WebSocket + REST endpoints
  rag/
    questions.py       deciding which utterances deserve an answer
    chunking.py        splitting documents into overlapping chunks
    transcripts.py     grouping ASR segments into windows; timestamps
    embeddings.py      text → 384-dim vectors
    store.py           FAISS index: build, save, load, search
    loaders.py         reading .txt / .md / .pdf, content hashing
    ingest.py          document ingestion
    indexing.py        the single shared index rebuild + consistency check
    retrieve.py        question → ranked chunks, with source filters
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

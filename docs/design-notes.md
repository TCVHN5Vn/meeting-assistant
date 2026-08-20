# Design notes

Why this system is built the way it is, and what the alternatives cost.
Each section is a decision that could reasonably have gone the other way.

---

## 1. Why a server and a client, rather than one script

Stage 1 was a single script: audio file in, transcript out, process exits.
That is the correct shape for a batch job and the wrong shape for a meeting
assistant, for two reasons.

**Model loading.** Whisper takes seconds to load into memory. In a script
that cost is paid once and amortised over the whole file. In a
request-per-process design it would be paid on every single request, and
would dominate the runtime of a 5-second audio chunk.

**Latency.** A meeting assistant that produces its transcript after the
meeting has ended is a transcription tool, not an assistant. Answering
questions *during* the meeting requires text to exist *during* the meeting,
which requires processing audio as it arrives.

So the server is a long-lived process that loads the model once at startup
and stays up. WebSocket rather than HTTP because the transport needs to be
bidirectional and continuous: audio flows continuously in one direction while
transcript chunks flow back in the other, on the same connection, without
the client polling.

`client_simulator.py` exists because testing this needs an audio source, and
wiring up a live microphone is a different problem from the one being
solved. It chops a finished recording into 5-second slices and sends them
with a real 5-second sleep between each. The server cannot tell the
difference. Swapping it for real microphone capture changes one file and
touches no server code.

---

## 2. Why `asyncio.to_thread` around Whisper

The first version of the server called `model.transcribe()` directly inside
the `async def` WebSocket handler. That looks fine and is a real bug.

An `async def` function only yields control to the event loop at an `await`.
A blocking CPU call inside one occupies the event loop thread for its whole
duration — so while one client's 5-second chunk is being transcribed, every
other connection is frozen, and even `/health` does not respond. The code
was *shaped* like concurrent code without *being* concurrent.

`asyncio.to_thread` moves the work to a worker thread and awaits the result,
freeing the loop.

**The part that matters:** a thread only helps because faster-whisper's
backend (CTranslate2) is C++ and releases the GIL while computing. For
CPU-bound work written in pure Python, the GIL would keep everything
serialised and a thread would buy nothing — that case needs a process pool.
Knowing which of the two you have is the actual skill; "use to_thread for
blocking calls" without that distinction is cargo cult.

There is a related trap in `app/asr.py`: `transcribe_bytes` returns a
`list`, not a generator. faster-whisper's segment generator is lazy, so
transcription only really happens as you iterate it. Return the generator
and the computation moves back onto the event loop when the caller iterates
it — reintroducing exactly the bug the thread was meant to fix.

---

## 3. Why the vectors are not in the database

SQLite cannot do nearest-neighbour search. A B-tree index sorts values along
a line; it has no concept of "close" in 384 dimensions. Finding nearest
neighbours means comparing a query vector against every stored vector —
arithmetic over a matrix of floats, which is what FAISS is for.

So there are two stores, joined by one column:

```
SQLite  document_chunks.text        FAISS  position 0, 1, 2, ...
        document_chunks.vector_id ──────▶
```

FAISS returns integer positions and distances. It stores no text. Turning
"vector #7" back into readable text is a lookup on `vector_id`.

**The cost of this design is consistency.** The two stores can drift, and
when they do, nothing errors — search returns text that does not correspond
to the vector that matched, confidently and silently. This project handles
it by never mutating the index in place: any change to the corpus rebuilds
the whole index and reassigns every `vector_id` in one deterministic ordered
pass (`_rebuild_index` in `app/rag/ingest.py`). Slow in principle, a few
seconds in practice, and it makes the failure mode impossible rather than
unlikely. `/api/v1/index/stats` reports whether the two counts still agree.

**Postgres with pgvector** — what the original architecture document
specified — collapses both stores into one, so the chunk row and its vector
are written in the same transaction and cannot drift. That is the strongest
argument for it. The cost is running a database server. At the point where
rebuilding the index stops being cheap, that becomes the right trade.

---

## 4. Why `IndexFlatIP` and normalised vectors

`IndexFlatIP` computes inner products. `IndexFlatL2` computes Euclidean
distance. For vectors normalised to length 1, the inner product **is** the
cosine of the angle between them — so normalising at embedding time and
using IP gives cosine similarity, bounded in [-1, 1].

Cosine is the right measure for text because it compares *direction* (what
the text is about) and ignores *magnitude* (roughly, length and emphasis).
Two passages on the same topic at different lengths should score as similar;
Euclidean distance would penalise the length difference.

"Flat" means exhaustive: every query is compared against every vector, so
results are exactly correct. That is O(n), which is fine for thousands or
hundreds of thousands of chunks. Only at millions does an approximate index
(IVF, HNSW) become worth its recall loss. Reaching for the approximate index
first is premature optimisation, and an interviewer asking "why not HNSW?"
is usually checking whether you know that.

---

## 5. Chunking: 1000 characters, 200 overlap

**Why chunk at all** — an embedding is one fixed-size vector regardless of
input length. Embedding a whole handbook averages every topic in it into one
blurry point that is vaguely near everything and strongly near nothing.
Chunking gives each idea its own vector. Secondarily, retrieved text has to
fit in the model's context window alongside the question.

**Why overlap** — boundaries are arbitrary and land mid-thought. If a
condition ends up at the tail of chunk 4 and its consequence at the head of
chunk 5, neither answers the question alone. Repeating ~200 characters means
any short passage survives intact somewhere. The cost is storage and
near-duplicate results.

**The tradeoff** — smaller chunks give sharper retrieval and lose context;
larger chunks carry more context per hit but blur the vector and crowd the
context window.

**The honest answer to "how did you pick 1000/200?"** is that it is a
common default, and that picking it properly means building an evaluation
set of real questions with known-correct source passages and measuring
retrieval quality (recall@k) across settings. Claiming a tuned number
without that measurement is the wrong answer.

Chunking is character-based rather than token-based. Tokens are more
accurate — models think in tokens and the characters-per-token ratio varies
by language — but characters need no tokenizer and are close enough here.
English averages roughly 4 characters per token.

---

## 6. Refusing before generating, not after

`answer_question` returns a fixed refusal *without calling the LLM* when
nothing clears the relevance floor.

This matters because vector search always returns its top-k, no matter how
poor the matches are. Ask an HR corpus about lasagna and it still hands back
chunks — just with low scores. Pass those to a model and it will produce a
fluent answer built from irrelevant context or, worse, from its training
data. The system prompt says "answer only from the context", but a prompt is
a request, not a guarantee.

A score threshold enforced in code is a guarantee. It also saves the
generation call entirely, which on a local 7B model is the expensive part.

Calibrating the threshold is model-specific. With MiniLM on this corpus,
>0.5 is a solid match, 0.3–0.5 is loosely related, below 0.3 is noise; 0.25
is the floor here. Copying a threshold from another project's blog post
without checking it against your own data is how this quietly stops working.

---

## 7. Two endpoints, `/search` and `/ask`

`/search` returns what retrieval found. `/ask` returns what the LLM
generated from it. They could have been one endpoint with a flag.

Keeping them separate means that when an answer is wrong, one request tells
you which half is at fault. Bad retrieval and bad generation look identical
from the outside and have completely different fixes — chunk size, embedding
model, and threshold for one; prompt, temperature, and model size for the
other. Without this split you are guessing. The `--no-llm` flag on
`scripts/ask.py` exists for the same reason.

---

## 8. Local models instead of a hosted API

**For:** zero marginal cost, works offline, and — the real argument for a
meeting recorder — audio and company documents never leave the machine.
That is a compliance story, not just a cost saving, and it is often the
deciding factor for legal or regulated customers.

**Against:** a 7B model on a laptop is meaningfully weaker than a frontier
hosted model, and slower to first token. Quality on multi-step reasoning and
on strict structured output is visibly lower.

The decision is made reversible by keeping every model call behind
`app/llm/ollama_client.py`. Swapping to a hosted API means writing one more
module exposing the same three functions; no pipeline code changes. That is
the point of the module boundary — not tidiness, but keeping a decision
cheap to revisit.

Streaming (`chat_stream`) does not make generation faster. It makes it *feel*
fast, because text appears in under a second instead of after twenty. For
someone waiting mid-meeting, that is the difference between a tool that gets
used and one that does not.

---

## 9. Confidence scores are stored raw, not thresholded

faster-whisper gives no clean 0–1 confidence. `avg_logprob` is the closest
proxy: an average log-probability, so 0 is perfectly confident and more
negative is less confident.

It is stored as-is rather than filtered at write time, because the threshold
depends on what the data is being used for. A transcript shown to a human
should include a low-confidence chunk — the reader can judge it. The same
chunk being embedded into the retrieval index probably should not, because
garbage text produces a garbage vector that will then match queries it has
nothing to do with, and quietly poison results. Same number, two different
thresholds. Deciding at read time keeps both options open; filtering at
write time throws information away permanently.

---

## 10. Chunk-relative timestamps had to be offset

Found by reading the output of a real streaming run rather than by reading
the code, which is the only way this kind of bug gets found.

Every chunk is handed to Whisper in isolation, so Whisper timestamps it from
zero. The fortieth second of the meeting came back labelled `start_ts: 0.0`,
the same as the first. Nothing errored. The transcript still looked correct
in the terminal, because the chunks happened to arrive in order.

What it broke was everything built on top: `ORDER BY start_ts` returned
essentially random order, "jump to this moment in the recording" was
impossible, and action items extracted later would have carried meaningless
timestamps. The batch path was unaffected — one file, one call, real
timestamps — so the two code paths silently disagreed about what `start_ts`
meant.

The fix is a running offset per connection, advanced by each chunk's audio
duration. The subtlety is advancing by the AUDIO duration rather than by the
end of the last segment: a chunk that ends in silence produces no segment
for that silence, so using the last segment's end would quietly lose those
seconds and the offset would drift further behind as the meeting went on. A
drifting clock is much harder to notice than a stopped one.

The general lesson: when the same data is produced by two code paths, the
invariant they are supposed to share — here, "timestamps are relative to the
start of the meeting" — needs to be stated somewhere and checked. Otherwise
one path can violate it for a week without anything failing.

---

## 11. One index for documents and transcripts, not two

The alternative was a second table and a second FAISS index for transcripts,
kept parallel to the document ones. It looks tidier and it is worse.

It fails on any question that needs both — "does what we agreed in the
meeting match the written policy?" — because two indexes return two
separately-ranked lists whose scores cannot honestly be compared. Each
similarity is computed against a different corpus, so 0.51 in one index and
0.51 in the other do not mean the same thing. Merging them means inventing a
fusion rule and defending it.

One index over one embedding space makes that question a single search, and
cross-source ranking comes free. In practice a real query returns them
interleaved — transcript 0.644, document 0.574, transcript 0.484 — which is
only meaningful because all four numbers came out of the same space.

The cost is filtering. `IndexFlat` stores nothing but vectors: no metadata,
no `WHERE`. So "search only this meeting" has to be applied after the search,
in SQL, which means the search can return k results that the filter then
discards, leaving fewer than k. The mitigation is over-fetching — ask for
8× and filter down — and it is a heuristic, not a guarantee: a filter
matching 1% of the corpus can still starve.

**That is the clearest practical argument for a real vector database**
(Qdrant, Weaviate, pgvector) over a bare index. They push the filter into
the search itself and return exactly k matching results. Worth being able to
name as a limit of this design rather than pretending it away.

A consequence: `rag_chunks.source_id` points into either `documents` or
`meetings` depending on `source_type`, which SQL cannot express as a foreign
key. The database therefore cannot stop an orphan, so `index_stats()` counts
them instead. Losing a constraint is a real cost of the polymorphic design,
not a detail.

---

## 12. Transcripts need a different chunker, not the same one

The tempting shortcut is to concatenate a meeting's segments into one string
and run the document chunker over it. That discards the two things that make
a transcript a transcript.

**Time.** Segments carry timestamps. Flatten them and a retrieved passage
can no longer say "this was said 34 minutes in" — which is most of the value
of citing a meeting, because it is what lets someone go and listen and judge
for themselves. A citation nobody can follow is decoration. This is also why
`Hit.citation` formats the two source types differently: "chunk 7 of the
meeting" is useless to a human, `Special General Meeting @ 33:46-35:05` is
not.

**Boundaries.** Written text signals structure with blank lines and full
stops. Speech signals it with SILENCE. A three-second pause is the spoken
equivalent of a paragraph break and a far better cut point than a character
count.

The unit of grouping differs too. ASR segments here average about 50
characters — roughly ten words. Embedding one per vector produces vectors
for fragments like "Five to ten." which mean nothing on their own and match
queries at random. 723 segments became 52 windows: a 14:1 compression, and
the difference between an index of fragments and an index of ideas.

### The subtlety: overlap belongs to arbitrary cuts only

Caught by a unit test, not by reading the code.

The first version applied the overlap carry-over on *every* split, including
splits caused by a silence. That is wrong twice over. It drags the end of
one topic into the head of the next, defeating the reason for cutting at a
pause at all — and it stretches the new window's `start_ts` back across the
silence, so its citation points at a moment when nobody was speaking.

Overlap exists to stop a passage being severed by an *arbitrary* cut. A
size-based cut is arbitrary and the content continues across it, so overlap
applies. A silence is not arbitrary: it is where the speaker stopped, and
the content genuinely changed. The two cases look identical in the code and
need opposite treatment.

---

## 13. The UNIQUE constraint that only broke on the fourth document

`rag_chunks.vector_id` is UNIQUE, and rebuilding reassigns every one of them.
Updating row by row therefore collides: a row about to be given id 5 hits
whichever row still holds 5 from the previous build.

It stayed invisible through every rebuild of an unchanged corpus, because
the ordering came out identical and each row was reassigned the value it
already had. Adding a fourth document changed the ordering and it failed
immediately. **A bug that only appears when the data changes shape is a bug
that will appear in production and not in testing.**

The fix is one statement: blank every `vector_id` to NULL first, then
assign. NULL is exempt from UNIQUE in SQLite (and in the SQL standard), so
the reassignment has a clear field.

Two things worth taking from it:

- The write order was also wrong, and that mattered more. The index file was
  saved *before* the database was committed, so the failed run left 72
  vectors in FAISS against 67 rows in SQLite. Committing the database first
  means a later failure leaves vector_ids pointing at a missing index, which
  `load_index` reports as a clear error. The reverse leaves a new index
  paired with old vector_ids — which returns real text for the wrong vectors,
  and looks like nothing worse than the model giving poor answers.
- `index_stats()["in_sync"]` caught it, immediately and precisely. Cheap
  consistency checks over data that two systems must agree on repay
  themselves the first time they fire.

---

## Known limitations

Worth being able to name unprompted — being asked "what would you fix
first?" and having a real answer is worth more than an unbroken defence.

1. **Chunk boundaries cut words.** The client slices audio every 5 seconds
   regardless of whether anyone is mid-sentence, and each slice is
   transcribed independently with no context from the previous one. Whisper
   loses the word at the seam. The fix is voice-activity detection to cut at
   natural pauses, plus a rolling audio buffer with overlap — the same
   overlap idea used for text chunks, applied to audio.

2. **No speaker diarization.** `speaker_id` exists in the schema and is
   always NULL. Adding pyannote would populate it, at real CPU cost.

3. **A `sqlite3` connection is opened per WebSocket connection** and every
   insert commits individually. Fine for one client; at scale this needs
   batched commits and WAL mode, and SQLite's single-writer model becomes
   the ceiling that forces the Postgres migration.

4. **Citation numbering is unreliable.** The 7B model sometimes attributes a
   claim to the wrong `[n]` — in one observed answer it sourced a policy
   requirement to a transcript chunk. The retrieved text was correct and the
   claim was correct; only the number was wrong. Worth knowing because it is
   the failure mode a reader is least likely to check. Mitigations, none of
   them free: verify each cited claim against its chunk in a second pass,
   constrain the output format, or use a larger model.

5. **Indexing transcripts is a manual step.** `scripts/index_transcripts.py`
   has to be run after a meeting. In a product this would be triggered by
   the WebSocket disconnecting. It is manual here because re-running it on
   demand is what you need while tuning the window size or the confidence
   floor.

6. **Near-duplicate meetings are not detected.** The same audio transcribed
   twice produces two meetings whose chunks are near-identical, and they
   will fill the top-k with the same passage twice. Handled by hand here (one
   of the two is simply not indexed). A real fix would deduplicate on
   content similarity at ingest time.

7. **No evaluation set.** Retrieval quality is assessed by trying queries
   and looking at the output. That is fine for development and not fine as
   a claim of quality. A labelled question→passage set with recall@k is what
   would turn tuning from guesswork into measurement.

8. **No authentication.** Sprint 5. Every endpoint is currently open.

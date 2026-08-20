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

## 14. Answering must not block transcription

The one architectural requirement of live Q&A. Generation on a local 7B
model takes ten to twenty seconds. `await`ing it inside the WebSocket
receive loop would stop the loop consuming audio — so chunks would pile up
in the socket buffer while the meeting carried on talking, and transcription
would fall behind by however long the answer took and never catch up.
**The meeting does not pause for the assistant.**

So the answer is started with `asyncio.create_task` and not awaited. The
next line of the receive loop is immediately ready for the next audio chunk.
You can see it working in a real run — transcript lines arriving in the
middle of the answer being written:

```
│ [   20.0s] and [name] and administration lead.
[   23.0s] Thank you.
The three main goals of this meeting are: ...
```

Three consequences follow, and each needed handling:

**A send lock.** Two coroutines now write to one socket — the receive loop
sending `transcript_chunk`, the answer task sending `qa_delta` — and
interleaving two writes on a WebSocket can split a frame. An `asyncio.Lock`
makes each send atomic against the other. It was not needed before, because
there was only ever one writer.

**A done-callback.** An exception inside a bare `create_task` is swallowed
and resurfaces as "Task exception was never retrieved" at garbage
collection, long after the fact and detached from the question that caused
it. The callback logs it against the meeting immediately.

**One answer at a time.** Two concurrent generations on one machine compete
for the same CPU and both come back slower than either alone — and a second
answer arriving while the first is still printing is not readable anyway.
The client is told `qa_busy` rather than being silently queued, because
"your question was dropped" is information and silence is not.

### Bridging a blocking generator into asyncio

`ollama_client.chat_stream` is an ordinary synchronous generator: each
`next()` blocks until the model emits a fragment. Iterating it inside a
coroutine would hold the event loop for the whole generation — the same
mistake as calling Whisper inline, but much harder to see, because a `for`
loop over a generator does not look like a blocking call.

`aiter_blocking` runs the iteration on a worker thread that pushes items
onto an `asyncio.Queue` via `loop.call_soon_threadsafe` — the supported way
to touch loop-owned objects from another thread; calling `put_nowait`
directly from the thread would be a data race.

Its honest limitation: **a Python thread cannot be cancelled from outside.**
If the client disconnects mid-generation the thread runs to completion and
its output is discarded. Detaching is the best available; genuinely stopping
it would mean giving the producer a flag to poll and having the HTTP client
abort the request.

---

## 15. Detection is a precision problem, not a recall problem

The sprint's one-line description is "detect questions in the transcript
stream". Implemented literally, that produces something unusable.

A meeting is full of questions. Real ones, from the transcript in this
repository: *"How have those been remembered for the past year?"*, *"What is
the special general meeting?"*. Every one is a genuine question and not one
is addressed to an assistant. They are people talking to each other.

The asymmetry decides the design. A missed question costs one person typing
it out. A false positive costs everyone's attention, mid-meeting, plus some
credibility. **An assistant that interrupts gets switched off in the first
ten minutes.** So the default is explicit address — the assistant answers
when spoken to — with an `AUTO_ANSWER_MODE = "questions"` setting for
demonstrating the pipeline, honest about being noisy.

Detection is string matching on purpose. It runs on every chunk, roughly
every five seconds, so it has to be free. Asking an LLM "is this for you?"
would cost a full generation per chunk on a machine already running Whisper,
and would make the assistant slow at the thing it must be fast at. Cheap
gate first, expensive model only once the gate passes.

### The wake phrase is the least reliable part of the sentence

The first version matched the literal string `"hey assistant"` and never
fired once on real audio. Whisper transcribed it as **"he assistant"**.

That is structural, not bad luck. A wake phrase is short, said quickly and
flatly, and sits at a phrase boundary where the recogniser has no
surrounding words to constrain its guess. *The part of the utterance you are
keying on is the part most likely to come back wrong.*

So the name is matched exactly — "assistant" is long and distinctive enough
to survive — while the greeting in front of it is matched against a set that
includes the mishearings actually observed (`he`, `hay`, `hei`). Requiring
*some* greeting is what preserves precision: bare "assistant" occurs in
ordinary sentences ("the assistant will circulate the notes") and would fire
on them.

---

## 16. Waiting for the rest of the question

Even once the wake phrase matched, the answer was wrong — because the
question was cut in half by an audio chunk boundary:

```
[ 5.0s] ...he assistant, what is the notice period?
[10.0s] for a general meeting, let us see what it comes back with
```

Answering on the first chunk asks "what is the notice period?" and drops the
half saying *which* notice period. The answer would have been confident and
about the wrong thing.

Nor can punctuation be trusted to signal the end: note that the recogniser
put a question mark exactly at the cut point. It punctuates from prosody,
and a boundary sounds like a pause.

The fix is a short timer. On hearing a wake phrase the question is held, any
chunk arriving before the timer fires is appended to it, and both paths end
in the same place — which keeps "answer exactly once" in one spot rather
than two. It costs about five seconds against generation that already takes
ten to twenty.

**This is compensation, not a cure.** The cause is upstream: audio sliced on
a stopwatch rather than at pauses. Voice-activity detection removes the
cause, and it is the first thing worth building next.

---

## 17. Recent discussion is pasted in, not retrieved

The reflex is to run everything through the retriever. For the last few
minutes of the meeting that is slower and strictly worse.

Recent context is small, bounded and guaranteed relevant, and it fits in the
context window comfortably. There is nothing for retrieval to do except risk
leaving out the sentence that was just spoken. **Retrieval exists to find
the few relevant passages among thousands that do not fit** — recent context
is the opposite case. Not everything needs RAG.

There is a practical reason too: a meeting in progress is not in the index.
Indexing runs when it ends.

One consequence worth noticing: the live path does **not** refuse when
retrieval comes back empty, though the post-meeting path does and should.
"What did we just decide?" is answerable entirely from the recent discussion
and will legitimately retrieve nothing. Refusing on empty retrieval would
break exactly the questions a live assistant is most useful for.

---

## 18. Cutting audio at pauses instead of on a stopwatch

The root cause behind three separate problems, fixed at the source rather
than patched where each one surfaced.

Audio arrives continuously and has to be cut somewhere before Whisper sees
it. The original client cut every five seconds on a clock, which severs
words mid-syllable and hands the recogniser a fragment with no beginning.
That single choice produced: garbled words at every seam, a wake phrase
misheard as "he", and questions split across two chunks needing a timer to
reassemble. Two of those had already been patched where they showed up.

Silero VAD gives a probability of speech per 32ms window; an utterance ends
after enough consecutive silent windows. The cut then lands in a gap.

### It is measurably better, not just better in principle

Same 90 seconds of meeting audio, both ways:

| | fixed 5s | VAD |
|---|---|---|
| mean `avg_logprob` | −0.612 | **−0.432** |
| segments below −0.7 | 6 | **2** |
| audio pieces sent to Whisper | 18 | **8** |

And on the wake-phrase clip, the same sentence before and after:

```
before:  What the policy actually says about this, he assistant,
         what is the notice period?
         for a general meeting, let us see what it comes back with
         final meeting.  |  until report.

after:   I think we should check what the policy actually says about this.
         Hey assistant, what is the notice period for a general meeting?
         And then we can move on to the financial report.
```

**"he assistant" became "Hey assistant".** The tolerant matching added in
Sprint 3 is still there and still worth having, but the mishearing it was
compensating for largely stopped happening once the audio stopped being cut
through the middle of the phrase.

### Why a neural VAD and not an energy threshold

RMS energy is the obvious cheap approach and it cannot tell loud noise from
speech. Measured here, Silero returns p=0.005 on random noise at speech
amplitude — an energy detector scores that as loudly voiced. Meetings have
chairs, doors, typing and coughs, all energetic, none worth transcribing.

The recording in this repository proves the point better than the synthetic
test. Its first two minutes have peak amplitude near clipping and
p(speech) ≈ 0 — it is **applause**. The old pipeline fed it to Whisper,
which hallucinated `"Thank you."` four times over it. VAD drops it.

The better detector costs almost nothing: 0.1ms per window, roughly 300×
faster than realtime.

### The flag that turns a guess into a fact

`Utterance.reason` records *why* the audio was cut — `silence`,
`length_cap`, or `stream_end` — and that is worth as much as the quality
improvement. Sprint 3 could not know whether a question was finished, so it
waited five seconds for a continuation on *every* question. Now a question
cut at a pause is known to be complete and is answered at once; only one cut
by the cap waits.

VAD also made a second Sprint 3 workaround unnecessary. Detection used to
run on the whole chunk's text joined together, because a wake phrase and its
question routinely landed in different five-second chunks. With whole
sentences per utterance, detection runs per segment instead — so the
question is the sentence carrying the wake phrase, not that sentence plus
everything after it.

### Two bugs the tests caught

Both had one root: **the minimum-speech floor counted padding as speech.**

A 250ms cough plus 300ms of pre-roll cleared a 400ms floor and got
transcribed — and Whisper invents words for noise, which then pollutes the
transcript and the index. The floor now counts speech windows only.

The same miscounting let a pure-silence utterance be emitted after a
length-cap cut. Fixing it exposed a second issue: the cap was being checked
on silent windows too, so it could fire a few windows before a pause the
speaker was already arriving at, producing a spurious "unfinished" cut right
at a natural boundary. The cap is now checked only on speech.

### What it costs

Latency is no longer constant. Nothing is transcribed until the speaker
pauses, so transcript arrives in bursts rather than every five seconds, and
a long uninterrupted sentence produces nothing and then all of itself. The
20-second cap bounds the worst case; on real audio the median utterance is
about 6 seconds, so typical latency is close to what fixed chunking gave.

The proper fix for that is interim results — transcribing the in-progress
buffer periodically and emitting provisional text, finalising at the pause,
which is how hosted streaming ASR works. It costs re-transcribing a growing
buffer repeatedly, which is expensive on CPU. Not done here.

One more thing VAD requires: an explicit end-of-stream signal. A speaker
mid-sentence when the audio stops looks exactly like a speaker who has not
paused yet, so the server cannot tell them apart and the last utterance
would sit in the buffer forever. The client sends `end_audio`; a dropped
connection falls back to flushing on disconnect, where the transcript is
still written to the database even though there is no longer anyone to send
it to.

---

## 19. Extracting action items is a precision problem too

Asking a model to list the action items in a passage works on the first
try. The problem is that it works just as readily when there are none.

Meetings are full of language that sounds like commitment and is not — "we
should probably look at that", "someone ought to check". A model asked to
find tasks will find them. And a task list with three invented entries is
worse than no task list, because every line has to be checked against the
recording, which is the work it was supposed to save.

### The measurement

Six windows of the governance meeting in this repository — a discussion
about board appointments, containing no action items at all:

| prompt | items returned | with a quote not in the transcript |
|---|---|---|
| "Extract the action items from this transcript." | **33** | 2 |
| the restrained prompt in `app/tasks.py` | **0** | 0 |

Thirty-three fabricated commitments from one meeting's worth of debate. The
restraint is not decoration; it is most of the feature. The prompt names
what *not* to extract, and says explicitly that an empty list is a correct
and useful answer — because a model that believes it has failed by
returning nothing will find something.

### Quotes make the claim checkable

Every task must come with a quote copied verbatim from the transcript, and
the quote is then checked against the source **in code**. No match, no task.

That is the load-bearing part. A model that invents a task must also invent
the sentence it came from, and an invented sentence is not in the
transcript. It converts an unfalsifiable claim into one the program can test
for itself — the same shape as the relevance floor in §6: a rule enforced in
code, not a request made in a prompt.

Two details that matter:

- **Comparison is on normalised text.** Models tidy punctuation and
  capitalisation while copying, even when told not to. Rejecting a genuine
  quote over a comma would throw away real tasks.
- **A word-count floor.** "I will do that" verifies perfectly and appears in
  almost every meeting. Word count stands in for specificity: below five
  words, matching the transcript is not evidence of anything.

### Structured output constrains shape, not content

Ollama takes a JSON schema in `format` and constrains decoding to it — at
each step the sampler is restricted to tokens that can still produce a valid
document. The result is *guaranteed* to parse and to have the declared
shape. That is categorically stronger than asking nicely and retrying.

It guarantees nothing about the values. Asked for a deadline that was never
mentioned, a constrained model does not return malformed JSON — it returns
`due: "soon"`, because the schema said a string goes there. Observed
directly, along with `assignee: "Unassigned (volunteer)"`.

So a nullable field needs a schema that permits null, a prompt that asks for
null, **and** a pass in code that turns the model's stand-ins back into
null. All three, because the first two demonstrably do not suffice.

**Structured output removes parsing errors, not hallucination.**

### On the real meeting, it found two things

Twenty-three windows of a 53-minute governance debate produced two action
items, both from garbled transcript. That looks like a weak result and is
probably close to correct: the meeting was a discussion about mandate and
finances, not a project stand-up. The 33-vs-0 comparison above is what makes
that credible rather than merely hopeful.

Verified against a meeting that does contain tasks — a synthetic stand-up
with four explicit commitments and one deliberate distractor ("it would be
good to look at that at some point, but nothing is decided yet") — it found
all four, with assignees and deadlines, and left the distractor alone.

### The bug in locating a quote

A task said 26 seconds in was stored at 0:00. Its quote straddled two
segments, and neither contained the other:

```
segment: "...nothing is decided yet. One more thing, Naomi, please
          update the on call"
segment: "rota before the end of the month. Understood, I will do that."
quote:   "Naomi, please update the on call rota before the end of the month."
```

Both containment tests failed, so it fell back to the whole window.

The first fix — the fraction of a segment's words appearing anywhere in the
quote — failed on the very case it was written for: those two segments score
0.43 and 0.58 against a 0.6 bar, because each carries a trailing clause the
quote does not. **Set overlap counts scattered words, so it is diluted by
exactly the extra text that makes the case hard**, and raising the threshold
would have made it worse.

A contiguous run is the right signal. Six words in a row in the same order
is a quotation; sharing "the" and "of" six times over is not. Those segments
share runs of six and seven words.

The lesson is about the metric, not the threshold: when a similarity measure
fails, check whether it is measuring the thing that actually distinguishes
the cases before reaching for a different cutoff.

---

## 20. Authentication, and the mistakes it is easy to ship

Security code has a property that ordinary code does not: **when it is
wrong, everything still works.** A system that accepts forged tokens, or
stores passwords reversibly, behaves exactly like one that does not — right
up until it matters. There is no failing request to notice, so the decisions
have to be made deliberately rather than discovered.

### A dependency, not middleware

Every protected route names `Depends(current_user)` in its signature.

The alternative — middleware protecting everything except a list of exempt
paths — **fails open**. Add a route, forget to think about the list, and it
is public with nothing to notice. Here an unprotected route is unprotected
because someone left the dependency out, which is visible in the diff.

### 404, not 403

Asking for someone else's meeting returns "not found", the same as asking
for one that does not exist.

403 would confirm the meeting is real. Any signed-in user could then
enumerate valid meeting ids by watching which ones answer 403 and which
answer 404. The same rule applies to tasks: the check is on the meeting the
task belongs to, and both failures produce one 404.

### Login must not reveal which emails are registered

The first version looked the user up and returned early if there was none.
That is a timing oracle: a missing account returns in microseconds, a real
one takes the ~170ms bcrypt costs, and the difference alone enumerates the
user table.

So the password is verified against a dummy hash even when the account does
not exist. Measured: 179ms for a real account, 171ms for a missing one —
7ms apart, which is noise. Both return the same message, because "no such
user" and "wrong password" are the same answer to anyone who is not the
account holder.

### The secret has no default

`get_secret()` falls back to a key generated at import, not to a constant.

A hardcoded development default is the most reliably shipped vulnerability
in this class of code, precisely because it works — nobody notices, and the
published value then forges tokens in production. A random fallback is safe
by construction and self-announcing: tokens stop working when the server
restarts, which is annoying in development and harmless anywhere else.

A configured secret shorter than 32 bytes is **refused**, not warned about.
RFC 7518 wants an HMAC key at least as long as the hash output, and PyJWT
warns rather than refusing — so nothing otherwise stops a six-character
secret in an env var. A warning in a startup log is a note nobody reads, not
a control.

### Why the WebSocket authenticates in its first message

The REST API uses `Authorization: Bearer`, which is right there and wrong
here: a browser's WebSocket API cannot set headers at all.

The usual workaround is `?token=...` in the URL. It works everywhere, and it
writes a live credential to access logs, proxy logs, browser history and
`Referer` headers — three systems that were never thinking about secrets.

So the token is the first message on the socket. One extra round trip, works
in every client. The timeout is the part that matters: without it an
unauthenticated connection sits open holding a slot, which is a free denial
of service against a server that loads a speech model per process. Anything
that is not an auth frame — audio included — closes the connection with 1008
rather than being buffered. No work is done for a caller who has not proved
who they are.

### Password storage

bcrypt at cost 12, roughly 170ms per hash here. **The slowness is the
feature.** A general-purpose hash like SHA-256 is fast, which is exactly
wrong: fast means billions of guesses per second against a stolen table.

The salt is generated per password and stored inside the hash string, so
there is no second column to manage and no way to reuse one. Unique salts
are what stop a single rainbow table cracking every account at once, and
what stop two users with the same password having visibly identical hashes.

Argon2id is the stronger modern choice, being memory-hard as well as slow,
which blunts GPU parallelism specifically. bcrypt remains acceptable and is
chosen here for having exactly one parameter to get wrong.

### `algorithms=` is not optional

`jwt.decode(token, key, algorithms=["HS256"])`. A decoder that trusts the
`alg` header in the token itself accepts `alg: none` — an unsigned token
that verifies against anything. PyJWT makes the list mandatory, which is a
library making the safe thing compulsory rather than merely available.
There is a test for it.

### The migration trap

Adding `created_by` to the schema changed nothing for any database that
already existed, because `CREATE TABLE IF NOT EXISTS` does exactly nothing
when the table is there — including when the definition has grown a column
since. The code then failed reading a column SQLite said did not exist.

That is the flaw in "the schema is just a CREATE TABLE script": it is
correct only for a fresh database. Real migrations need a real mechanism.
`_migrate` in `app/db.py` is the smallest honest one — SQLite has no ADD
COLUMN IF NOT EXISTS, so the existing columns are read first and only the
missing ones are added.

### Accounts are created from a script, not a /register endpoint

Open registration is a product decision, and shipping one by default decides
it silently. On a self-hosted meeting recorder, "anyone who can reach the
port can create an account" is almost never what was wanted. A registration
endpoint can be added deliberately, with whatever invite or domain
restriction the deployment actually needs.

---

## 21. The UI, and why it is not React

The architecture document said React. This is one HTML file of vanilla
JavaScript, served by the API that backs it, with no build step.

React earns its place when shared state across many components becomes the
hard part. Here there are two panes and one WebSocket, and the state is a
token, a meeting id, and a list of transcript lines. Against that, a
bundler, a `node_modules` tree and a second toolchain are a real cost — in a
project whose entire claim is that it runs locally with no ceremony, and
which a reader should be able to clone and run with `pip install`. Adding
React would have been resume-driven rather than reasoned.

That is a defensible answer *because it names the condition under which it
flips*. "React is overkill here" is an argument; "React is bloat" is a
slogan.

### The browser is now the microphone

This is what makes the UI more than a viewer: it removes the last piece of
scaffolding. Until now audio could only come from a file replayed by a
script. `getUserMedia` plus an `AudioWorklet` makes it a real microphone,
and the server cannot tell the difference — which was the point of the
client/server split back in Sprint 1.

Three details that are easy to get wrong:

**Ask the AudioContext for 16 kHz** rather than downsampling in JavaScript.
`new AudioContext({sampleRate: 16000})` makes the browser resample with the
anti-alias filtering that naive decimation skips — and skipping it folds
high frequencies down into the speech band as noise, which sounds like a
worse microphone and is actually a worse pipeline.

**The worklet runs on the audio thread**, which must never block or the
audio glitches. So it does the minimum: buffer to one second, convert to
16-bit, hand the buffer over as a transferable. All the interesting work
stays on the server.

**A worklet is only pulled if it reaches the destination.** Connecting it
straight to `ctx.destination` would play the microphone back into the room;
routing through a gain node set to zero keeps it running silently.

### The screenshot found a bug the tests could not

The panel accumulated a dead "listening for the rest of the question…" card
for every question asked out loud: the placeholder shown on hearing a wake
phrase was never replaced by the real answer, only followed by it.

Nothing was broken. Every request succeeded, every answer was correct, and
no test could have failed — there was no assertion to make. It was visible
in one glance at a rendered page and invisible everywhere else. **A class of
defect that only surfaces when you look at the thing.**

### Testing a UI without a human

Driven through the Chrome DevTools Protocol: launch headless Chrome with a
debugging port, connect over a WebSocket, and use `Runtime.evaluate` to fill
the login form, click Record, and read the resulting DOM. No test framework,
about eighty lines.

Real microphone audio came from `--use-file-for-fake-audio-capture`, which
replaces the microphone with a WAV file. **The app runs completely
unmodified** — it calls `getUserMedia` and gets a real `MediaStream`, and
every layer below is the production path.

Two false starts worth recording, because both looked like application bugs
and neither was:

- The first run produced silence — `peak: 0` on every frame. The audio file
  was suffixed `%noloop`, so Chrome played it once at browser startup, long
  before anything called `getUserMedia`.
- The second produced silence too, and Chrome had already said why in a log
  nobody had read: `Failed to read /tmp/fakemic.wav ... Try disabling the
  sandbox`. The fix was a flag.

The frames were the right size and arriving at the right rate the whole
time. **Instrumenting the amplitude rather than the plumbing is what
separated "my code is broken" from "my test rig is broken"** — and the
difference between those two is worth ten minutes of measurement every time.

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

5. **The VAD threshold is not tuned per environment.** 0.5 works on both
   recordings here, but a noisy room or a distant microphone would want it
   lower, and there is no calibration step or per-connection adaptation.

6. **Assignees are names, not people.** `tasks.assignee` holds the name as
   spoken ("Karen", "the treasurer"). The architecture document specifies a
   foreign key to a users table, which needs both a users table and speaker
   diarization to resolve a name to a person. Neither exists.

7. **Deadlines are phrases, not dates.** `due` stores "before Friday" as
   said. Parsing it needs the meeting's date and a decision about which
   Friday, and a wrong date looks authoritative in a way a wrong phrase
   does not.

8. **Extraction is slow.** One model call per window, several minutes on a
   50-minute meeting, and it holds the HTTP request open. A production
   version would return 202 and a job id.

9. **Indexing transcripts is a manual step.** `scripts/index_transcripts.py`
   has to be run after a meeting. In a product this would be triggered by
   the WebSocket disconnecting. It is manual here because re-running it on
   demand is what you need while tuning the window size or the confidence
   floor.

10. **Near-duplicate meetings are not detected.** The same audio transcribed
   twice produces two meetings whose chunks are near-identical, and they
   will fill the top-k with the same passage twice. Handled by hand here (one
   of the two is simply not indexed). A real fix would deduplicate on
   content similarity at ingest time.

11. **No evaluation set.** Retrieval quality is assessed by trying queries
   and looking at the output. That is fine for development and not fine as
   a claim of quality. A labelled question→passage set with recall@k is what
   would turn tuning from guesswork into measurement.

12. **The UI has no meeting history.** It records and asks against the
    current session; browsing past meetings, their transcripts and their
    tasks still means the API or a script.

13. **No token refresh or revocation.** A token is valid for 12 hours and
    cannot be invalidated before then — revoking access means rotating the
    signing key, which logs everyone out. A refresh token plus a short-lived
    access token is the usual answer; a deny-list is the other.

14. **No rate limiting on login.** Nothing slows down an attacker trying
    passwords beyond bcrypt's own cost, which is a real but shallow defence.

15. **Documents are not scoped to a user.** Meetings and tasks are; the RAG
    corpus is shared by everyone with an account. That is deliberate for a
    single-organisation deployment and wrong for a multi-tenant one.

"""
The FastAPI application: a live meeting session over a WebSocket, plus
REST endpoints for reading back what was captured.

Run with:
    uvicorn app.server:app --reload

The WebSocket carries a live meeting in both directions at once: audio in,
transcript and answers out. See the docstring on ws_meeting for the full
event list.
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import numpy as np

from app import asr, db
from app.config import QUESTION_CONTINUATION_SECONDS
from app.vad import UtteranceBuffer
from app.llm import ollama_client
from app.rag import indexing, qa, questions, transcripts
from app.tasks import extract_meeting
from app.rag.qa import DEFAULT_MIN_SCORE, DEFAULT_TOP_K
from app.rag.retrieve import retrieve


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown work for the whole application.

    Loading Whisper here, at boot, rather than lazily on the first
    connection, means the cost is paid before anyone is waiting. If it
    were lazy, the very first client would sit through a multi-second
    model load with no idea why nothing was happening.
    """
    db.init_db().close()
    asr.get_model()
    print("Server ready.")
    yield
    print("Server shutting down.")


app = FastAPI(title="Meeting Assistant", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


def _hit_to_dict(hit) -> dict:
    return {
        "text": hit.text,
        "score": hit.score,
        "source_type": hit.source_type,
        "source_id": hit.source_id,
        "source_title": hit.source_title,
        "chunk_index": hit.chunk_index,
        "start_ts": hit.start_ts,
        "end_ts": hit.end_ts,
        "citation": hit.citation,
    }


async def aiter_blocking(make_iterator):
    """Consume a BLOCKING generator from async code without stalling the loop.

    `ollama_client.chat_stream` is an ordinary synchronous generator: each
    `next()` on it blocks until the model emits the next fragment. Iterating
    it directly inside a coroutine would hold the event loop for the whole
    generation -- the same mistake as calling Whisper inline, just harder to
    see, because a `for` loop over a generator does not look like a blocking
    call.

    So the iteration runs on a worker thread which pushes each item onto an
    asyncio queue, and this coroutine yields them as they land.
    `loop.call_soon_threadsafe` is the supported way to touch loop-owned
    objects from another thread; calling `queue.put_nowait` directly from
    the thread would be a data race.

    Caveat worth knowing: a Python thread cannot be cancelled from outside.
    If the consumer goes away mid-generation the thread keeps running until
    the model finishes, and its remaining output is discarded. Detaching is
    the best that can be done here; genuinely stopping it would mean giving
    the producer something to poll and having the HTTP client abort.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    def pump():
        try:
            for item in make_iterator():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the loop side
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, DONE)

    worker = asyncio.create_task(asyncio.to_thread(pump))
    try:
        while True:
            item = await queue.get()
            if item is DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        if not worker.done():
            worker.cancel()


class LiveSession:
    """One WebSocket connection: one meeting, in progress.

    Holds the per-connection state that the handler used to keep in local
    variables, plus the two new things Sprint 3 needs -- a send lock and a
    single in-flight answer task.
    """

    def __init__(self, websocket: WebSocket, conn, meeting_id: str):
        self.ws = websocket
        self.conn = conn
        self.meeting_id = meeting_id

        # Accumulates the incoming stream and cuts it at pauses rather than
        # on a stopwatch. It also owns the meeting clock: every utterance
        # carries an absolute start time derived from the total number of
        # samples consumed, which is exact. The old code accumulated an
        # offset from chunk durations instead, which could drift.
        self.buffer = UtteranceBuffer()

        # TWO coroutines now write to this socket: the receive loop sending
        # transcript_chunk events, and an answer task sending qa_delta
        # events, at the same time. Interleaving two writes on one WebSocket
        # can split a frame. The lock makes each send atomic with respect to
        # the other. It was not needed before Sprint 3 because there was only
        # ever one writer.
        self.send_lock = asyncio.Lock()

        # At most one answer generating at a time. See _spawn_answer.
        self.answer_task: asyncio.Task | None = None

        # A question heard but not yet answered, because it may not be
        # finished. See _note_question.
        self.pending_question: str | None = None
        self.pending_trigger: str | None = None
        self.pending_timer: asyncio.Task | None = None

    async def send(self, event: str, **data) -> None:
        async with self.send_lock:
            await self.ws.send_json({"event": event, "data": data})

    # ---- inbound audio -------------------------------------------------

    async def handle_audio(self, frame: bytes) -> None:
        """Take one frame of raw PCM off the wire and process any speech in it.

        Frames are raw signed 16-bit mono at 16 kHz -- no container, no
        header. The client sends short frames (~1s) so the buffer has the
        granularity to find a pause; the frames are NOT the transcription
        unit. Where the audio gets cut is decided here, by the voice
        detector, not by whatever size the client happened to send.
        """
        # int16 -> float32 in [-1, 1], which is what both the VAD and Whisper
        # expect. Dividing by 32768 rather than 32767 keeps the scaling
        # exact for the most negative sample.
        pcm = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0

        # Runs on the event loop rather than a worker thread, and that is a
        # measured decision, not an oversight. The detector costs ~0.1ms per
        # 32ms window: about 3ms of work per second of audio, roughly 300x
        # faster than realtime. A thread hop per frame would cost a
        # meaningful fraction of that to save it.
        #
        # "Never block the event loop" is really "never block it for long".
        # Whisper at seconds per call clearly must move off; this clearly
        # need not. The line worth changing your mind at is around ten
        # milliseconds per call.
        for utterance in self.buffer.add(pcm):
            await self.handle_utterance(utterance)

    async def handle_utterance(self, utterance, notify: bool = True) -> None:
        """Transcribe one utterance, store it, and optionally send it out.

        `notify=False` is for the flush after the client has gone: the
        transcript still belongs in the database even though there is no
        longer anyone to send it to. Persisting and notifying are separate
        concerns and only one of them needs a live socket.
        """
        print(f"[{self.meeting_id}] utterance {utterance.start:.1f}-"
              f"{utterance.end:.1f}s ({utterance.duration:.1f}s, cut on "
              f"{utterance.reason})")

        # Whisper inference is CPU-bound and takes seconds. Called directly it
        # would run ON the event loop thread and freeze the entire server.
        # A thread helps here only because CTranslate2 releases the GIL while
        # computing in C++; pure-Python CPU work would need a process pool.
        segments = await asyncio.to_thread(asr.transcribe_audio, utterance.audio)

        for segment in segments:
            # Whisper timestamps each piece of audio it is given from zero,
            # so segment times are relative to this utterance. The utterance
            # knows where it sits in the meeting.
            start_ts = utterance.start + segment.start
            end_ts = utterance.start + segment.end

            db.insert_transcript_chunk(
                self.conn,
                chunk_id=str(uuid.uuid4()),
                meeting_id=self.meeting_id,
                text=segment.text,
                start_ts=start_ts,
                end_ts=end_ts,
                confidence=segment.confidence,
            )
            if notify:
                await self.send(
                    "transcript_chunk",
                    text=segment.text,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    confidence=segment.confidence,
                )

        if not notify:
            return

        spoken = [segment.text for segment in segments if segment.text.strip()]
        if not spoken:
            return

        if self.pending_question is not None:
            # A question was left open because its utterance was cut mid-
            # sentence. This is the rest of it.
            self.pending_question = f"{self.pending_question} {spoken[0]}".strip()
            return

        # Per SEGMENT, not over the joined text: Whisper's segments are
        # sentences, so the sentence carrying the wake phrase is the question
        # and the ones after it are not. Joining would swallow them.
        detected = questions.detect_in_segments(spoken)
        if not detected:
            return

        if utterance.is_complete:
            # The speaker stopped, or the audio ended. Either way nothing
            # more is coming, so the question is whole -- answer at once.
            #
            # Sprint 3 could not know this and waited five seconds for every
            # question, for a continuation that usually never came. The voice
            # detector does not only improve transcription; it reports WHY
            # the audio was cut, which turns a guess into a fact.
            self._spawn_answer(detected.text, trigger=detected.trigger)
        else:
            await self._note_question(detected.text, detected.trigger)

    async def flush_audio(self, notify: bool = True) -> None:
        """Force out whatever speech is still buffered."""
        final = self.buffer.flush()
        if final is not None:
            await self.handle_utterance(final, notify=notify)

    async def _note_question(self, question: str, trigger: str) -> None:
        """A question whose utterance was cut mid-sentence. Wait for the rest.

        Only reached when the speaker was still talking when the length cap
        hit, so the question probably continues. Answering now would ask half
        of it -- which is what Sprint 3 did on every question, before the
        voice detector could distinguish the two cases:

            [ 5.0s] ...he assistant, what is the notice period?
            [10.0s] for a general meeting, let us see what it comes back with

        Answering on the first half asks "what is the notice period?" and
        drops which notice period. Punctuation is no help either -- note the
        recogniser put a question mark exactly at the cut. It punctuates from
        prosody, and a boundary sounds like a pause.

        A short timer holds the question, any utterance arriving before it
        fires is appended, and both paths end in the same place, keeping
        "answer exactly once" in one spot rather than two.
        """
        self.pending_question = question
        self.pending_trigger = trigger
        await self.send("qa_listening", question=question)

        async def fire_when_settled() -> None:
            await asyncio.sleep(QUESTION_CONTINUATION_SECONDS)
            text, trig = self.pending_question, self.pending_trigger
            self.pending_question = self.pending_trigger = None
            if text:
                self._spawn_answer(text, trigger=trig)

        self.pending_timer = asyncio.create_task(fire_when_settled())

    # ---- inbound control messages ---------------------------------------

    async def handle_text(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            await self.send("error", message="expected JSON")
            return

        event = message.get("event")
        if event == "end_audio":
            # The client knows when it has stopped sending; the server cannot
            # infer it. Without an explicit marker the last utterance sits in
            # the buffer waiting for a closing silence that never arrives --
            # and a speaker who was still talking when the audio ended is
            # exactly the case where the buffer holds the most.
            await self.flush_audio()
            await self.send("audio_ended")
        elif event == "ask_query":
            question = (message.get("data") or {}).get("text", "").strip()
            if not question:
                await self.send("error", message="ask_query needs data.text")
                return
            self._spawn_answer(question, trigger="ask_query")
        else:
            await self.send("error", message=f"unknown event: {event!r}")

    # ---- answering -------------------------------------------------------

    def _spawn_answer(self, question: str, trigger: str) -> None:
        """Start answering, WITHOUT waiting for it to finish.

        This is the whole point of Sprint 3. Generation on a local 7B model
        takes ten to twenty seconds. Awaiting it here would stop the receive
        loop -- so audio chunks would queue up in the socket buffer while the
        meeting carried on talking, and transcription would fall behind by
        however long the answer took and never catch up. The meeting does not
        pause for the assistant.

        `create_task` schedules the work and returns immediately, so the very
        next line of the receive loop is ready for the next audio chunk.

        One answer at a time, deliberately. Two concurrent generations on one
        machine compete for the same CPU and both come back slower than
        either would alone -- and a second answer arriving while the first is
        still printing is not something a person can read anyway. Telling the
        client it is busy is more useful than silently queueing.
        """
        if self.answer_task and not self.answer_task.done():
            asyncio.create_task(
                self.send("qa_busy", question=question,
                          message="still answering the previous question")
            )
            return

        self.answer_task = asyncio.create_task(self._answer(question, trigger))

        # Without this, an exception inside the task is swallowed and only
        # surfaces as "Task exception was never retrieved" at garbage
        # collection, long after the fact and with no connection to the
        # question that caused it.
        self.answer_task.add_done_callback(self._answer_finished)

    def _answer_finished(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            print(f"[{self.meeting_id}] answer task failed: {exc!r}")

    async def _answer(self, question: str, trigger: str) -> None:
        print(f"[{self.meeting_id}] answering ({trigger}): {question!r}")

        try:
            # Retrieval and prompt assembly are blocking (embedding the query,
            # reading SQLite), so they go to a thread too. What comes back is
            # a generator that has not started -- the HTTP request to Ollama
            # is only made on first iteration.
            hits, recent, stream = await asyncio.to_thread(
                qa.answer_live_stream, question, self.meeting_id
            )
        except RuntimeError as exc:
            await self.send("qa_error", question=question, message=str(exc))
            return
        except ollama_client.OllamaError as exc:
            await self.send("qa_error", question=question, message=str(exc))
            return

        # Sources go out BEFORE the first token. Retrieval takes milliseconds
        # and generation takes seconds, so the client can show what is being
        # read from while the model is still writing.
        await self.send(
            "qa_started",
            question=question,
            trigger=trigger,
            sources=[_hit_to_dict(h) for h in hits],
            live_context_chars=len(recent),
        )

        pieces: list[str] = []
        try:
            async for fragment in aiter_blocking(lambda: stream):
                pieces.append(fragment)
                await self.send("qa_delta", text=fragment)
        except ollama_client.OllamaError as exc:
            await self.send("qa_error", question=question, message=str(exc))
            return

        # The complete event named in the architecture doc. Sent at the end
        # with the whole answer, so a client that ignores the deltas entirely
        # still gets a correct result from this one event.
        await self.send(
            "qa_response",
            question=question,
            text="".join(pieces),
            trigger=trigger,
            sources=[_hit_to_dict(h) for h in hits],
        )

    async def close(self) -> None:
        """Stop any in-flight work and release the database connection.

        Flushes the audio buffer first: whatever was being said when the
        connection dropped never got its closing silence, and would
        otherwise be discarded.
        """
        try:
            # notify=False: the socket is gone, but the transcript still
            # belongs in the database.
            await self.flush_audio(notify=False)
        except Exception as exc:  # noqa: BLE001 - best effort on a dead socket
            print(f"[{self.meeting_id}] could not flush final utterance: {exc!r}")

        if self.pending_timer and not self.pending_timer.done():
            self.pending_timer.cancel()
        if self.answer_task and not self.answer_task.done():
            self.answer_task.cancel()
            # Wait for the cancellation to actually take effect rather than
            # leaving a task pointing at a closed socket.
            try:
                await self.answer_task
            except asyncio.CancelledError:
                pass
        self.conn.close()


@app.websocket("/ws/meetings/{meeting_id}")
async def ws_meeting(websocket: WebSocket, meeting_id: str):
    """One WebSocket connection = one live meeting session.

    Protocol, both directions on the same socket:

        client -> server  binary            raw PCM16 mono @ 16 kHz
        client -> server  {"event": "ask_query", "data": {"text": ...}}
    client -> server  {"event": "end_audio"}   no more audio is coming
        server -> client  transcript_chunk   text as it is recognised
        server -> client  qa_started         sources, before generation
        server -> client  qa_delta           answer fragments, as generated
        server -> client  qa_response        the complete answer
        server -> client  audio_ended        the buffer has been flushed
    server -> client  qa_busy / qa_error / error
    """
    await websocket.accept()
    conn = db.init_db()
    db.create_meeting(conn, meeting_id, title="Live Meeting")
    print(f"[{meeting_id}] client connected")

    session = LiveSession(websocket, conn, meeting_id)

    try:
        while True:
            # receive() rather than receive_bytes(), because this socket now
            # carries two kinds of message: audio as binary frames and control
            # events as JSON text. receive_bytes() would reject the text ones.
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))

            if (data := message.get("bytes")) is not None:
                await session.handle_audio(data)  # raw PCM16 @ 16 kHz
            elif (text := message.get("text")) is not None:
                await session.handle_text(text)

    except WebSocketDisconnect:
        print(f"[{meeting_id}] client disconnected")
    finally:
        await session.close()


@app.get("/api/v1/meetings")
def list_meetings():
    conn = db.init_db()
    try:
        rows = conn.execute(
            """SELECT m.id, m.title, m.created_at, COUNT(c.id) AS chunk_count
               FROM meetings m
               LEFT JOIN transcript_chunks c ON c.meeting_id = m.id
               GROUP BY m.id
               ORDER BY m.created_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/v1/meetings/{meeting_id}/transcript")
def get_transcript(meeting_id: str):
    conn = db.init_db()
    try:
        meeting = conn.execute(
            "SELECT id, title, created_at FROM meetings WHERE id = ?", (meeting_id,)
        ).fetchone()
        if meeting is None:
            raise HTTPException(status_code=404, detail="meeting not found")

        chunks = db.get_transcript(conn, meeting_id)
        return {"meeting": dict(meeting), "chunks": [dict(c) for c in chunks]}
    finally:
        conn.close()


# --- RAG endpoints (Sprint 2) -------------------------------------------
#
# Deliberately two endpoints, not one. /search returns what RETRIEVAL found;
# /ask returns what the LLM GENERATED from it. Being able to hit them
# separately is the difference between debugging a RAG system and guessing
# at it -- when an answer is wrong, /search tells you in one request whether
# the right text was even retrieved.

class SearchRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    # Optional scoping: 'document' or 'transcript', and/or one meeting.
    source_type: str | None = None
    meeting_id: str | None = None


class AskRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    source_type: str | None = None
    meeting_id: str | None = None


@app.get("/api/v1/index/stats")
def index_stats_endpoint():
    """Corpus size, and whether SQLite and FAISS still agree with each other."""
    return indexing.index_stats()


@app.post("/api/v1/meetings/{meeting_id}/index")
async def index_meeting_endpoint(meeting_id: str):
    """Make one meeting's transcript searchable.

    In a finished product this would be triggered automatically when the
    meeting ends. Exposed as an endpoint so it can be re-run after tuning
    the window size or the confidence floor.

    On a worker thread because it embeds every chunk in the corpus, which
    takes seconds and would otherwise block every other request.
    """
    def work() -> dict:
        conn = db.init_db()
        try:
            windows = transcripts.index_meeting(conn, meeting_id)
            indexing.rebuild_index(conn)
            return {"meeting_id": meeting_id, "windows_indexed": windows}
        finally:
            conn.close()

    try:
        return await asyncio.to_thread(work)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --- Tasks (Sprint 4) ----------------------------------------------------

class TaskStatusRequest(BaseModel):
    status: str


@app.get("/api/v1/meetings/{meeting_id}/tasks")
def list_tasks(meeting_id: str):
    """Action items already extracted from this meeting."""
    conn = db.init_db()
    try:
        return {"meeting_id": meeting_id,
                "tasks": [dict(row) for row in db.get_tasks(conn, meeting_id)]}
    finally:
        conn.close()


@app.post("/api/v1/meetings/{meeting_id}/tasks")
async def extract_tasks(meeting_id: str):
    """Extract action items from a meeting's transcript.

    Slow -- one model call per window, minutes on a long meeting -- and
    entirely blocking, so it runs on a worker thread. A production version
    would return 202 with a job id rather than holding the request open;
    this keeps it synchronous because a script and a curl are the only
    clients, and a job queue with one consumer is machinery without a
    purpose.
    """
    def work() -> dict:
        conn = db.init_db()
        try:
            result = extract_meeting(conn, meeting_id)
            result["tasks"] = [dict(row) for row in db.get_tasks(conn, meeting_id)]
            return result
        finally:
            conn.close()

    try:
        return await asyncio.to_thread(work)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ollama_client.OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.patch("/api/v1/tasks/{task_id}")
def update_task(task_id: str, request: TaskStatusRequest):
    """Mark a task done, or reopen it."""
    allowed = {"open", "done", "cancelled"}
    if request.status not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"status must be one of {sorted(allowed)}")

    conn = db.init_db()
    try:
        if not db.set_task_status(conn, task_id, request.status):
            raise HTTPException(status_code=404, detail="task not found")
        return {"id": task_id, "status": request.status}
    finally:
        conn.close()


@app.post("/api/v1/search")
def search_documents(request: SearchRequest):
    """Semantic search over the ingested documents. No LLM involved."""
    try:
        hits = retrieve(
            request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            source_type=request.source_type,
            meeting_id=request.meeting_id,
        )
    except RuntimeError as exc:
        # No index built yet -- a 409 rather than a 500, because the server
        # is fine; the corpus just has not been ingested.
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {"query": request.query, "hits": [_hit_to_dict(h) for h in hits]}


@app.post("/api/v1/ask")
async def ask(request: AskRequest):
    """Full RAG: retrieve, then generate a grounded answer with citations.

    Wrapped in to_thread for the same reason as Whisper in the WebSocket
    handler: httpx's synchronous client blocks, and generation takes many
    seconds. Left on the event loop it would stall every other request for
    the whole duration of one answer.
    """
    try:
        answer = await asyncio.to_thread(
            qa.answer_question, request.question, request.top_k,
            request.min_score, request.source_type, request.meeting_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ollama_client.OllamaError as exc:
        # 503: the request was valid, a dependency we need is unavailable.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "question": request.question,
        "answer": answer.text,
        "used_context": answer.used_context,
        "sources": [_hit_to_dict(h) for h in answer.sources],
    }

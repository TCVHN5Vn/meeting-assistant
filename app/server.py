"""
The FastAPI application: a live meeting session over a WebSocket, plus
REST endpoints for reading back what was captured.

Run with:
    uvicorn app.server:app --reload

WebSocket protocol (from the architecture doc):
    client -> server : audio_chunk       (raw bytes)
    server -> client : transcript_chunk  (JSON)
"""

import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app import asr, db
from app.llm import ollama_client
from app.rag import indexing, qa, transcripts
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


@app.websocket("/ws/meetings/{meeting_id}")
async def ws_meeting(websocket: WebSocket, meeting_id: str):
    """One WebSocket connection = one live meeting session."""
    await websocket.accept()
    conn = db.init_db()
    db.create_meeting(conn, meeting_id, title="Live Meeting")
    print(f"[{meeting_id}] client connected")

    # Seconds of audio received so far on this connection. See the comment
    # in the loop below for why this has to exist.
    offset = 0.0

    try:
        while True:
            # Suspends this coroutine until the next chunk arrives, WITHOUT
            # blocking the process -- other connections keep being served
            # while we wait here. This is the part async is genuinely good at.
            audio_bytes = await websocket.receive_bytes()
            print(f"[{meeting_id}] received chunk: {len(audio_bytes)} bytes")

            # ---- The important line in this file -------------------------
            #
            # Whisper inference is CPU-bound work that takes seconds. Called
            # directly, it would run ON the event loop thread and freeze the
            # entire server for its duration -- every other connection
            # stalled, health checks unanswered. An `async def` function
            # only yields control at an `await`; a plain blocking call
            # inside one gives the loop no opportunity to do anything else.
            #
            # asyncio.to_thread hands the work to a worker thread and awaits
            # the result, so the event loop stays free.
            #
            # A thread only helps because CTranslate2 (faster-whisper's
            # backend) releases the GIL while computing in C++. For pure
            # Python CPU work the GIL would keep it serialised and you would
            # need a process pool instead. Knowing which of the two applies
            # is the actual skill here.
            segments, chunk_duration = await asyncio.to_thread(
                asr.transcribe_bytes, audio_bytes
            )

            for segment in segments:
                # Whisper timestamps each chunk in isolation, so they all
                # start again from 0.0. Left uncorrected, every segment in
                # a 40-minute meeting claims to happen in its first few
                # seconds -- which silently destroys ordering, makes
                # "jump to this moment in the audio" impossible, and would
                # put nonsense timestamps on extracted action items later.
                # Adding the running offset converts chunk-relative time
                # into meeting-relative time.
                start_ts = offset + segment.start
                end_ts = offset + segment.end

                db.insert_transcript_chunk(
                    conn,
                    chunk_id=str(uuid.uuid4()),
                    meeting_id=meeting_id,
                    text=segment.text,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    confidence=segment.confidence,
                )
                await websocket.send_json({
                    "event": "transcript_chunk",
                    "data": {
                        "text": segment.text,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "confidence": segment.confidence,
                    },
                })

            # Advance by the AUDIO duration, not by the end of the last
            # segment. A chunk ending in silence produces no segment for
            # that silence, so using the last segment's end would lose
            # those seconds and the offset would drift earlier and earlier
            # as the meeting went on.
            offset += chunk_duration

    except WebSocketDisconnect:
        print(f"[{meeting_id}] client disconnected")
    finally:
        conn.close()


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

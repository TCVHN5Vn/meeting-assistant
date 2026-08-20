"""
Simulates a live meeting: streams a recording to the server in real time and
renders whatever comes back.

Chops a finished recording into fixed-length windows and sends them one at a
time with a real delay between sends, so the server experiences this exactly
as it would a live microphone. Swapping this for real microphone capture is a
small change -- the server cannot tell the difference either way.

Usage (from the project root):
    python -m scripts.client_simulator <audio> [meeting_id]
    python -m scripts.client_simulator <audio> --ask "what is the notice period?"
    python -m scripts.client_simulator <audio> --ask "..." --after 3

--ask sends an `ask_query` event partway through the stream, which is how you
watch the assistant answer a question WHILE transcription carries on. That
interleaving is the thing to look for: transcript_chunk events should keep
arriving in between the answer's fragments, not stop and wait for it.
"""

import asyncio
import io
import json
import sys
import uuid

import websockets
from pydub import AudioSegment

CHUNK_MS = 5000  # 5 seconds of audio per chunk

# ANSI colours: transcript in plain text, the assistant's answer in colour, so
# the interleaving is visible at a glance rather than needing to be read.
DIM, CYAN, YELLOW, RED, RESET = "\033[2m", "\033[36m", "\033[33m", "\033[31m", "\033[0m"


def extract_wav_chunks(audio_path, chunk_ms=CHUNK_MS):
    """Load the file, downsample to 16kHz mono (what Whisper expects), and
    yield WAV-encoded byte chunks of `chunk_ms` each.

    pydub needs ffmpeg installed for non-wav formats like m4a.
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    for start_ms in range(0, len(audio), chunk_ms):
        chunk = audio[start_ms:start_ms + chunk_ms]
        buffer = io.BytesIO()
        chunk.export(buffer, format="wav")
        yield buffer.getvalue()


def render(message: str) -> None:
    """Print one server event in a readable form."""
    payload = json.loads(message)
    event, data = payload.get("event"), payload.get("data", {})

    if event == "transcript_chunk":
        print(f"{DIM}[{data['start_ts']:7.1f}s]{RESET} {data['text']}")

    elif event == "qa_started":
        trigger = data.get("trigger")
        print(f"\n{CYAN}┌─ answering ({trigger}): {data['question']}{RESET}")
        for i, source in enumerate(data.get("sources", []), start=1):
            print(f"{CYAN}│  [{i}] {source['citation']} "
                  f"({source['score']:.3f}){RESET}")
        if data.get("live_context_chars"):
            print(f"{CYAN}│  + {data['live_context_chars']} chars of this "
                  f"meeting so far{RESET}")
        print(f"{CYAN}│{RESET} ", end="", flush=True)

    elif event == "qa_delta":
        # flush=True, or the fragments sit in Python's buffer and streaming
        # looks exactly like not streaming.
        print(f"{CYAN}{data['text']}{RESET}", end="", flush=True)

    elif event == "qa_response":
        print(f"\n{CYAN}└─ complete ({len(data['text'])} chars){RESET}\n")

    elif event == "qa_busy":
        print(f"\n{YELLOW}[busy] {data['message']}{RESET}")

    elif event in ("qa_error", "error"):
        print(f"\n{RED}[{event}] {data.get('message')}{RESET}")

    else:
        print(f"{DIM}[{event}] {data}{RESET}")


async def stream_file(audio_path, meeting_id, ask=None, after=2):
    uri = f"ws://localhost:8000/ws/meetings/{meeting_id}"

    async with websockets.connect(uri, max_size=None) as websocket:
        print(f"Connected to {uri}\n")

        async def receiver():
            # Runs concurrently with sending, so responses appear as they
            # arrive rather than after every chunk has been sent.
            async for message in websocket:
                render(message)

        recv_task = asyncio.create_task(receiver())

        chunks = list(extract_wav_chunks(audio_path))
        print(f"Streaming {len(chunks)} chunks (~{CHUNK_MS / 1000:.0f}s each)...\n")

        for i, chunk_bytes in enumerate(chunks):
            await websocket.send(chunk_bytes)

            if ask and i == after:
                print(f"\n{YELLOW}>> asking: {ask}{RESET}")
                await websocket.send(json.dumps({
                    "event": "ask_query",
                    "data": {"text": ask},
                }))

            # A real delay matching the chunk length, so audio arrives in
            # real time rather than all at once.
            await asyncio.sleep(CHUNK_MS / 1000)

        # Generation can still be running after the last chunk was sent.
        await asyncio.sleep(45)
        recv_task.cancel()


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        sys.exit(1)

    def take(flag, default=None):
        if flag in argv:
            index = argv.index(flag)
            value = argv[index + 1]
            del argv[index:index + 2]
            return value
        return default

    ask = take("--ask")
    after = int(take("--after", "2"))

    audio_path = argv[0]
    meeting_id = argv[1] if len(argv) > 1 else str(uuid.uuid4())
    print(f"meeting_id = {meeting_id}")

    asyncio.run(stream_file(audio_path, meeting_id, ask=ask, after=after))


if __name__ == "__main__":
    main()

"""
Simulates a live audio feed by chopping a finished recording into
fixed-length windows and sending them to the server one at a time,
with a real delay between sends -- so the server experiences this
exactly like it would a real live microphone stream, even though
we're feeding it from a file for easy testing.

Later, swapping this file-based sender for an actual microphone
capture is a small change -- the server side doesn't need to know
the difference; it just receives audio_chunk bytes either way.

Usage:
    python -m scripts.client_simulator sample_data/audio/short_recording.m4a [meeting_id]
"""

import asyncio
import io
import sys
import uuid

import websockets
from pydub import AudioSegment

CHUNK_MS = 5000  # 5 seconds of audio per chunk


def extract_wav_chunks(audio_path, chunk_ms=CHUNK_MS):
    """
    Load the whole file (pydub needs ffmpeg installed for non-wav
    formats like m4a), downsample it to 16kHz mono (what Whisper
    expects), then yield WAV-encoded byte chunks of `chunk_ms` each.
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    for start_ms in range(0, len(audio), chunk_ms):
        chunk = audio[start_ms:start_ms + chunk_ms]
        buffer = io.BytesIO()
        chunk.export(buffer, format="wav")
        yield buffer.getvalue()


async def stream_file(audio_path, meeting_id):
    uri = f"ws://localhost:8000/ws/meetings/{meeting_id}"

    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")

        async def receiver():
            # Runs concurrently with sending -- prints responses from
            # the server as soon as they arrive, without waiting for
            # all chunks to finish sending first.
            async for message in websocket:
                print(f"<- {message}")

        recv_task = asyncio.create_task(receiver())

        chunks = list(extract_wav_chunks(audio_path))
        print(f"Streaming {len(chunks)} chunks (~{CHUNK_MS/1000:.0f}s each)...")

        for i, chunk_bytes in enumerate(chunks):
            print(f"-> sending chunk {i} ({len(chunk_bytes)} bytes)")
            await websocket.send(chunk_bytes)
            # Real delay, matching the chunk length, to simulate audio
            # actually arriving in real time rather than all at once.
            await asyncio.sleep(CHUNK_MS / 1000)

        await asyncio.sleep(3)  # let the last response arrive
        recv_task.cancel()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.client_simulator path/to/audio.m4a [meeting_id]")
        sys.exit(1)

    audio_path = sys.argv[1]
    meeting_id = sys.argv[2] if len(sys.argv) > 2 else str(uuid.uuid4())
    print(f"meeting_id = {meeting_id}")

    asyncio.run(stream_file(audio_path, meeting_id))

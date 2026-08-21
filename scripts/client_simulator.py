"""
Simulates a live meeting: streams a recording to the server in real time and
renders whatever comes back.

Chops a finished recording into fixed-length windows and sends them one at a
time with a real delay between sends, so the server experiences this exactly
as it would a live microphone. Swapping this for real microphone capture is a
small change -- the server cannot tell the difference either way.

Usage (from the project root):
    python -m scripts.client_simulator <audio> --email you@example.com
    python -m scripts.client_simulator <audio> --token <jwt>
    python -m scripts.client_simulator <audio> --ask "what is the notice period?"

The server requires authentication. Pass --email to log in (you will be
prompted for the password), or --token directly, or set
MEETING_ASSISTANT_TOKEN in the environment.

--ask sends an `ask_query` event `--after` seconds into the stream, which is
how you watch the assistant answer WHILE transcription carries on. That
interleaving is the thing to look for: transcript_chunk events should keep
arriving in between the answer's fragments, not stop and wait for it.

Note that transcript arrives in bursts rather than steadily. That is the
voice detector working: nothing is transcribed until the speaker pauses, so
a long sentence produces nothing and then all of itself at once.
"""

import asyncio
import getpass
import json
import os
import sys
import uuid

import httpx
import websockets
from pydub import AudioSegment

SERVER = "http://localhost:8000"

# How much audio goes in each network frame. NOT the transcription unit --
# the server decides where to cut, at pauses, using voice activity detection.
# Frames only need to be small enough to give it granularity to find one.
FRAME_MS = 1000

# ANSI colours: transcript in plain text, the assistant's answer in colour, so
# the interleaving is visible at a glance rather than needing to be read.
DIM, CYAN, YELLOW, RED, RESET = "\033[2m", "\033[36m", "\033[33m", "\033[31m", "\033[0m"


def extract_pcm_frames(audio_path, frame_ms=FRAME_MS):
    """Load the file and yield raw PCM frames the server can concatenate.

    Downsampled to 16 kHz mono signed 16-bit -- what both the voice detector
    and Whisper expect. Sent raw, with no WAV header: the server holds one
    continuous stream per connection and simply appends, so a container per
    frame would be a header to write and parse for nothing.

    pydub needs ffmpeg installed for non-wav formats like m4a.
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    raw = audio.raw_data
    bytes_per_frame = int(16000 * 2 * frame_ms / 1000)
    for start in range(0, len(raw), bytes_per_frame):
        yield raw[start:start + bytes_per_frame]


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

    elif event == "authenticated":
        pass  # already reported by stream_file

    elif event == "audio_ended":
        print(f"{DIM}[end of audio]{RESET}")

    elif event == "indexing_started":
        print(f"{DIM}[making this meeting searchable...]{RESET}")

    elif event == "indexed":
        print(f"{DIM}[searchable: {data['windows']} passages]{RESET}")

    elif event == "qa_busy":
        print(f"\n{YELLOW}[busy] {data['message']}{RESET}")

    elif event in ("qa_error", "error"):
        print(f"\n{RED}[{event}] {data.get('message')}{RESET}")

    else:
        print(f"{DIM}[{event}] {data}{RESET}")


def get_token(email=None, token=None) -> str:
    """Find a token: the flag, the environment, or by logging in."""
    token = token or os.environ.get("MEETING_ASSISTANT_TOKEN")
    if token:
        return token
    if not email:
        print("Need --email or --token (or MEETING_ASSISTANT_TOKEN).")
        sys.exit(1)

    password = getpass.getpass(f"Password for {email}: ")
    response = httpx.post(f"{SERVER}/api/v1/auth/login",
                          json={"email": email, "password": password}, timeout=30)
    if response.status_code != 200:
        print(f"Login failed: {response.json().get('detail')}")
        sys.exit(1)
    return response.json()["access_token"]


async def stream_file(audio_path, meeting_id, token, ask=None, after=15):
    uri = f"ws://localhost:8000/ws/meetings/{meeting_id}"

    async with websockets.connect(uri, max_size=None) as websocket:
        # The token goes in the first MESSAGE, not the URL. A browser cannot
        # set headers on a WebSocket, and "?token=..." would write a live
        # credential into access logs, proxy logs and browser history.
        await websocket.send(json.dumps({"event": "auth", "data": {"token": token}}))

        reply = json.loads(await websocket.recv())
        if reply.get("event") != "authenticated":
            print(f"Authentication failed: {reply}")
            return
        print(f"Connected to {uri} as {reply['data']['user']}\n")

        async def receiver():
            # Runs concurrently with sending, so responses appear as they
            # arrive rather than after every chunk has been sent.
            async for message in websocket:
                render(message)

        recv_task = asyncio.create_task(receiver())

        frames = list(extract_pcm_frames(audio_path))
        print(f"Streaming {len(frames)} frames of {FRAME_MS}ms "
              f"(~{len(frames) * FRAME_MS / 1000:.0f}s of audio)...\n")

        asked = False
        for i, frame in enumerate(frames):
            await websocket.send(frame)

            if ask and not asked and i * FRAME_MS / 1000 >= after:
                print(f"\n{YELLOW}>> asking: {ask}{RESET}")
                await websocket.send(json.dumps({
                    "event": "ask_query",
                    "data": {"text": ask},
                }))
                asked = True

            # A real delay matching the frame length, so audio arrives in
            # real time rather than all at once.
            await asyncio.sleep(FRAME_MS / 1000)

        # Tell the server no more audio is coming, so it transcribes whatever
        # is still buffered. It cannot work this out for itself: a speaker
        # mid-sentence when the audio ends looks exactly like a speaker who
        # has simply not paused yet.
        await websocket.send(json.dumps({"event": "end_audio"}))

        # Generation can still be running after the last frame was sent.
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
    after = float(take("--after", "15"))
    email = take("--email")
    token = take("--token")

    audio_path = argv[0]
    meeting_id = argv[1] if len(argv) > 1 else str(uuid.uuid4())

    token = get_token(email=email, token=token)
    print(f"meeting_id = {meeting_id}")

    asyncio.run(stream_file(audio_path, meeting_id, token, ask=ask, after=after))


if __name__ == "__main__":
    main()

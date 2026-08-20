"""
Batch transcription: one audio file in, timestamped transcript rows out.

This is the original Stage 1 pipeline, now built on the shared app.asr
and app.db modules instead of its own private copies of them.

Usage (from the project root):
    python -m scripts.transcribe_file sample_data/audio/short_recording.m4a
"""

import sys
import uuid

from app import asr, db


def transcribe_file(audio_path: str, meeting_title: str = "Batch Transcription") -> str:
    conn = db.init_db()

    # Every run creates one "meeting" row so its chunks can be grouped and
    # queried together later -- which is exactly why the schema links
    # transcript_chunks to meetings with a foreign key.
    meeting_id = str(uuid.uuid4())
    db.create_meeting(conn, meeting_id, meeting_title)

    print(f"Transcribing {audio_path} ...")
    segments, info = asr.transcribe_path(audio_path)
    print(f"Detected language: {info.language} "
          f"(confidence {info.language_probability:.2f})")
    print("-" * 70)

    count = 0
    for segment in segments:
        db.insert_transcript_chunk(
            conn,
            chunk_id=str(uuid.uuid4()),
            meeting_id=meeting_id,
            text=segment.text,
            start_ts=segment.start,
            end_ts=segment.end,
            confidence=segment.confidence,
        )
        print(f"[{segment.start:7.2f}s -> {segment.end:7.2f}s] "
              f"(conf {segment.confidence:+.2f})  {segment.text}")
        count += 1

    conn.close()
    print("-" * 70)
    print(f"Saved {count} chunks.")
    print(f"meeting_id = {meeting_id}")
    return meeting_id


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    transcribe_file(sys.argv[1])

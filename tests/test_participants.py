"""
Tests for multi-speaker meetings.

Two things are being checked, and they are different in kind.

The ACCESS rules are ordinary logic: who may read a meeting back. The
SHARED CLOCK is the interesting one -- every connection measures time by
counting its own audio samples, which is exact and starts at zero when that
connection starts sending. Merging two of those without correction produces
a transcript that is confidently, silently in the wrong order.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app import auth, config, db


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    connection = db.init_db()
    yield connection
    connection.close()


def make_user(conn, email):
    user_id = str(uuid.uuid4())
    db.create_user(conn, user_id, email, email.split("@")[0],
                   auth.hash_password("password123"))
    return user_id


# --- access ---------------------------------------------------------------

def test_a_participant_can_read_the_meeting_back(conn):
    """Attending a meeting and then being unable to read it would be odd."""
    host = make_user(conn, "host@example.com")
    guest = make_user(conn, "guest@example.com")
    meeting = str(uuid.uuid4())
    db.create_meeting(conn, meeting, "Standup", created_by=host)

    assert not db.user_can_access_meeting(conn, guest, meeting)
    db.add_participant(conn, meeting, guest, joined_offset=12.0)
    assert db.user_can_access_meeting(conn, guest, meeting)


def test_someone_who_never_joined_cannot(conn):
    host = make_user(conn, "host@example.com")
    guest = make_user(conn, "guest@example.com")
    stranger = make_user(conn, "stranger@example.com")
    meeting = str(uuid.uuid4())
    db.create_meeting(conn, meeting, "Standup", created_by=host)
    db.add_participant(conn, meeting, guest, joined_offset=0.0)

    assert not db.user_can_access_meeting(conn, stranger, meeting)


def test_joining_twice_does_not_duplicate(conn):
    """A dropped connection that reconnects is still one participant."""
    user = make_user(conn, "a@example.com")
    meeting = str(uuid.uuid4())
    db.create_meeting(conn, meeting, "M", created_by=user)
    db.add_participant(conn, meeting, user, joined_offset=0.0)
    db.add_participant(conn, meeting, user, joined_offset=95.0)

    participants = db.get_participants(conn, meeting)
    assert len(participants) == 1
    # The FIRST offset is kept: it is when they actually started speaking,
    # and a reconnect must not shunt their earlier words down the timeline.
    assert participants[0]["joined_offset"] == 0.0


# --- the shared clock -----------------------------------------------------

def test_the_first_speaker_defines_the_origin(conn):
    meeting = str(uuid.uuid4())
    db.create_meeting(conn, meeting, "M")
    assert db.meeting_started_at(conn, meeting) is None

    first = datetime.now(timezone.utc).isoformat()
    assert db.start_meeting_clock(conn, meeting, first) == first


def test_a_later_speaker_does_not_move_the_origin(conn):
    """The bug this prevents: everyone who speaks resets time to zero, and
    the whole transcript collapses onto the same instant."""
    meeting = str(uuid.uuid4())
    db.create_meeting(conn, meeting, "M")

    origin = datetime.now(timezone.utc)
    db.start_meeting_clock(conn, meeting, origin.isoformat())

    later = (origin + timedelta(minutes=3)).isoformat()
    assert db.start_meeting_clock(conn, meeting, later) == origin.isoformat()


def test_offsets_place_speakers_on_one_timeline(conn):
    """A participant joining three minutes late must not start at zero."""
    meeting = str(uuid.uuid4())
    db.create_meeting(conn, meeting, "M")
    host = make_user(conn, "host@example.com")
    guest = make_user(conn, "guest@example.com")

    origin = datetime.now(timezone.utc)
    db.start_meeting_clock(conn, meeting, origin.isoformat())
    db.add_participant(conn, meeting, host, joined_offset=0.0)

    late = origin + timedelta(seconds=180)
    offset = (late - datetime.fromisoformat(db.meeting_started_at(conn, meeting))).total_seconds()
    db.add_participant(conn, meeting, guest, joined_offset=offset)

    by_offset = db.get_participants(conn, meeting)
    assert [p["joined_offset"] for p in by_offset] == [0.0, pytest.approx(180.0, abs=1)]


# --- attribution ----------------------------------------------------------

def test_transcript_is_attributed_and_ordered(conn):
    """The merged transcript is ordered by MEETING time, not by speaker.

    Both speakers' lines interleave, which is the whole point -- a
    conversation read in the order it happened.
    """
    meeting = str(uuid.uuid4())
    db.create_meeting(conn, meeting, "M")
    host = make_user(conn, "host@example.com")
    guest = make_user(conn, "guest@example.com")

    for user, start, text in [
        (host, 0.0, "first thing"),
        (guest, 8.0, "reply from the other side"),
        (host, 13.0, "back to me"),
    ]:
        db.insert_transcript_chunk(
            conn, str(uuid.uuid4()), meeting, text=text,
            start_ts=start, end_ts=start + 2, confidence=-0.2, speaker_id=user)

    rows = db.get_transcript_with_speakers(conn, meeting)
    assert [r["text"] for r in rows] == ["first thing", "reply from the other side", "back to me"]
    assert [r["speaker_name"] for r in rows] == ["host", "guest", "host"]


def test_lines_without_a_speaker_are_not_dropped(conn):
    """Everything recorded before speakers existed has a NULL speaker_id.

    An inner join would silently delete every one of those lines from the
    transcript -- the reader would simply never learn they were there.
    """
    meeting = str(uuid.uuid4())
    db.create_meeting(conn, meeting, "M")
    db.insert_transcript_chunk(
        conn, str(uuid.uuid4()), meeting, text="from before speakers",
        start_ts=0.0, end_ts=1.0, confidence=-0.2, speaker_id=None)

    rows = db.get_transcript_with_speakers(conn, meeting)
    assert len(rows) == 1
    assert rows[0]["speaker_name"] is None

"""
Tests for password storage, tokens, and access control.

Security code is worth testing for a reason that does not apply elsewhere:
when it is wrong, everything still works. A system that accepts forged
tokens, or stores passwords reversibly, behaves exactly like one that does
not, right up until it matters. There is no failing request to notice.

So these assert the properties, not the happy path.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app import auth, config, db


# --- password storage ----------------------------------------------------

def test_password_is_not_stored_in_the_hash():
    """The obvious property, and the one worth an explicit test."""
    password = "correct horse battery staple"
    assert password not in auth.hash_password(password)


def test_hash_verifies():
    h = auth.hash_password("s3cret-password")
    assert auth.verify_password("s3cret-password", h)
    assert not auth.verify_password("s3cret-passwore", h)
    assert not auth.verify_password("", h)


def test_same_password_hashes_differently():
    """Unique salts, which is what stops one rainbow table cracking every
    account and stops equal passwords being visibly equal in the table."""
    a = auth.hash_password("same password")
    b = auth.hash_password("same password")
    assert a != b
    assert auth.verify_password("same password", a)
    assert auth.verify_password("same password", b)


def test_cost_factor_is_recorded_in_the_hash():
    """bcrypt stores its own cost, so raising it later still verifies old
    hashes rather than locking everyone out."""
    assert auth.hash_password("x").startswith(f"$2b${auth.BCRYPT_ROUNDS:02d}$")


def test_malformed_hash_is_refused_not_raised():
    """A corrupt row must not turn login into a 500 that tells an attacker
    this particular account is interesting."""
    assert not auth.verify_password("anything", "not-a-bcrypt-hash")
    assert not auth.verify_password("anything", "")


# --- tokens ---------------------------------------------------------------

def test_token_round_trip():
    assert auth.decode_token(auth.create_token("user-1")) == "user-1"


def test_tampered_token_is_rejected():
    token = auth.create_token("user-1")
    assert auth.decode_token(token[:-4] + "AAAA") is None


def test_token_signed_with_another_key_is_rejected():
    """The signature is the whole guarantee. A token minted elsewhere with
    the right shape must not be accepted."""
    forged = jwt.encode({"sub": "admin", "exp": datetime.now(timezone.utc)
                         + timedelta(hours=1)}, "z" * 64,
                        algorithm=config.JWT_ALGORITHM)
    assert auth.decode_token(forged) is None


def test_expired_token_is_rejected():
    expired = jwt.encode(
        {"sub": "user-1",
         "exp": datetime.now(timezone.utc) - timedelta(seconds=1)},
        auth.get_secret(), algorithm=config.JWT_ALGORITHM)
    assert auth.decode_token(expired) is None


def test_unsigned_token_is_rejected():
    """The classic JWT vulnerability: a token declaring alg=none, which a
    decoder that trusts the token's own header will happily accept."""
    unsigned = jwt.encode({"sub": "admin"}, key="", algorithm="none")
    assert auth.decode_token(unsigned) is None


def test_garbage_is_rejected_without_raising():
    for value in ("", "not.a.token", "a.b.c", "x" * 500):
        assert auth.decode_token(value) is None


def test_short_configured_secret_is_refused(monkeypatch):
    """A weak signing key undermines every token the system issues.

    Refused rather than warned about: a warning in a startup log is a note
    nobody reads, not a control.
    """
    monkeypatch.setenv("MEETING_ASSISTANT_SECRET", "hunter2")
    with pytest.raises(RuntimeError, match="at least"):
        auth.get_secret()


def test_adequate_configured_secret_is_used(monkeypatch):
    monkeypatch.setenv("MEETING_ASSISTANT_SECRET", "k" * 64)
    assert auth.get_secret() == "k" * 64


def test_no_hardcoded_default_secret():
    """A constant development fallback is the most reliably shipped
    vulnerability in this class of code: it works, so nobody notices, and
    the published value then forges tokens in production."""
    secret = auth.get_secret()
    assert secret and len(secret) >= 32
    assert secret not in {"secret", "changeme", "dev", "development",
                          "meeting-assistant", "test"}


# --- access control -------------------------------------------------------

@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    connection = db.init_db()
    yield connection
    connection.close()


def make_user(conn, email="a@example.com"):
    user_id = str(uuid.uuid4())
    db.create_user(conn, user_id, email, "Name", auth.hash_password("password123"))
    return user_id


def test_owner_can_access_their_meeting(conn):
    user_id = make_user(conn)
    meeting_id = str(uuid.uuid4())
    db.create_meeting(conn, meeting_id, "Mine", created_by=user_id)
    assert db.user_can_access_meeting(conn, user_id, meeting_id)


def test_other_users_cannot(conn):
    owner = make_user(conn, "owner@example.com")
    intruder = make_user(conn, "intruder@example.com")
    meeting_id = str(uuid.uuid4())
    db.create_meeting(conn, meeting_id, "Private", created_by=owner)
    assert not db.user_can_access_meeting(conn, intruder, meeting_id)


def test_unowned_legacy_meetings_stay_readable(conn):
    """Meetings recorded before authentication existed. Deliberately not
    orphaned, and deliberately not reassigned to whoever registers first."""
    user_id = make_user(conn)
    meeting_id = str(uuid.uuid4())
    db.create_meeting(conn, meeting_id, "Legacy", created_by=None)
    assert db.user_can_access_meeting(conn, user_id, meeting_id)


def test_missing_meeting_is_not_accessible(conn):
    """False for "does not exist" as well as "not yours", so the caller can
    return one 404 for both and not leak which meeting ids are real."""
    assert not db.user_can_access_meeting(conn, make_user(conn), str(uuid.uuid4()))


# --- user records ----------------------------------------------------------

def test_email_is_normalised(conn):
    """UNIQUE in SQLite is case-sensitive, so "A@b.com" and "a@b.com" would
    otherwise become two accounts."""
    db.create_user(conn, str(uuid.uuid4()), "  Mixed@Example.COM ", "N",
                   auth.hash_password("password123"))
    assert db.get_user_by_email(conn, "mixed@example.com") is not None
    assert db.get_user_by_email(conn, "MIXED@EXAMPLE.COM") is not None


def test_duplicate_email_is_rejected(conn):
    make_user(conn, "dup@example.com")
    with pytest.raises(sqlite3.IntegrityError):
        make_user(conn, "DUP@example.com")


def test_migration_adds_created_by_to_an_old_meetings_table(conn):
    """CREATE TABLE IF NOT EXISTS does nothing to a table that already
    exists, including when the schema in the file has grown a column. Any
    database made before ownership existed needs a real migration."""
    conn.execute("DROP TABLE IF EXISTS meetings")
    conn.execute("CREATE TABLE meetings (id TEXT PRIMARY KEY, title TEXT, created_at TEXT)")
    conn.commit()
    db._migrate(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(meetings)")}
    assert "created_by" in columns

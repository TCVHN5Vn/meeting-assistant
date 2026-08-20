"""
Passwords, tokens, and who is allowed to see what.

Three separate jobs that get conflated:

  * AUTHENTICATION  -- proving who you are (login, token verification)
  * AUTHORIZATION   -- deciding what you may touch (meeting ownership)
  * PASSWORD STORAGE -- making a stolen database useless

Getting one right does not help with the others. A perfect JWT setup over
plaintext passwords is a breach waiting to be published.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.config import JWT_ALGORITHM, JWT_TTL_HOURS

# --- Password storage ----------------------------------------------------

# bcrypt's cost factor. Each increment doubles the work: 12 is roughly a
# quarter-second per hash on this machine, which is unnoticeable at login and
# ruinous for an attacker running a wordlist against a stolen table.
#
# That slowness IS the feature. A general-purpose hash like SHA-256 is fast,
# which is exactly wrong here -- fast means billions of guesses per second on
# a GPU. Argon2id is the stronger modern choice because it is memory-hard as
# well as slow, which blunts GPU parallelism specifically; bcrypt remains
# acceptable and is chosen here for having exactly one knob to get wrong.
BCRYPT_ROUNDS = 12


def hash_password(password: str) -> str:
    """Hash a password for storage. Never store the password itself.

    The salt is generated per password and stored inside the returned string
    along with the cost factor, so there is no second column to manage and no
    way to accidentally reuse a salt across users. Unique salts are what stop
    one rainbow table from cracking every account at once, and what stop two
    users with the same password from having visibly identical hashes.
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Check a password against a stored hash.

    bcrypt.checkpw compares in constant time. An ordinary `==` on the hashes
    would leak, through how long it takes to fail, how many leading bytes
    matched -- which is enough to reconstruct a value one byte at a time.
    """
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        # A malformed hash in the database. Refuse rather than raise: the
        # login route must not turn a corrupt row into a 500 that tells an
        # attacker this account is interesting.
        return False


# --- Tokens ---------------------------------------------------------------

# RFC 7518 requires an HMAC key at least as long as the hash output -- 32
# bytes for SHA-256. A shorter key weakens the signature, and PyJWT warns
# about it rather than refusing, so nothing stops a 6-character secret in an
# env var from reaching production.
MIN_SECRET_BYTES = 32


def get_secret() -> str:
    """The signing key, from the environment.

    Falls back to a random key generated at import, NOT to a constant. A
    hardcoded development default is the single most reliably shipped
    vulnerability in this class of code: it works, so nobody notices, and the
    published value then forges tokens in production.

    A random fallback is safe by construction and self-announcing -- tokens
    stop working when the server restarts, which is annoying enough in
    development to be noticed and harmless if it ever reaches production.
    """
    configured = os.environ.get("MEETING_ASSISTANT_SECRET")
    if configured:
        if len(configured.encode()) < MIN_SECRET_BYTES:
            # Refuse rather than warn. A weak key here undermines every token
            # the system issues, and a warning in a startup log is not a
            # control -- it is a note nobody reads.
            raise RuntimeError(
                f"MEETING_ASSISTANT_SECRET must be at least {MIN_SECRET_BYTES} "
                f"bytes (got {len(configured.encode())}). Generate one with: "
                "python -c 'import secrets; print(secrets.token_hex(32))'")
        return configured
    return _EPHEMERAL_SECRET


# 64 hex characters, comfortably above MIN_SECRET_BYTES.
_EPHEMERAL_SECRET = uuid.uuid4().hex + uuid.uuid4().hex


def create_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": user_id,
            "iat": now,
            # Expiry is not optional. A token without one is a password that
            # can never be changed: revoking access would mean rotating the
            # signing key and logging out every user at once.
            "exp": now + timedelta(hours=JWT_TTL_HOURS),
        },
        get_secret(),
        algorithm=JWT_ALGORITHM,
    )


def decode_token(token: str) -> str | None:
    """Return the user id in a valid token, or None.

    `algorithms=` is passed explicitly and this matters. A decoder that
    trusts the `alg` header in the token accepts `alg: none` -- an unsigned
    token that anything will verify -- which is the classic JWT
    vulnerability. PyJWT requires the list, which is the library making the
    safe thing mandatory rather than merely available.

    Returns None for every failure -- expired, wrong signature, malformed --
    because the caller's response is the same in each case and telling them
    apart only helps an attacker.
    """
    try:
        payload = jwt.decode(token, get_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        return None
    return payload.get("sub")

"""
Create a user account.

Usage (from the project root):
    python -m scripts.create_user alice@example.com "Alice Smith"
    python -m scripts.create_user alice@example.com "Alice Smith" --password s3cret
    python -m scripts.create_user --list

WHY THIS IS A SCRIPT AND NOT A /register ENDPOINT

Because open registration is a product decision, and shipping one by default
decides it silently. On a self-hosted meeting recorder, "anyone who can reach
the port can make themselves an account" is almost never what was wanted.
Creating accounts from the machine that runs the server is the conservative
default; a registration endpoint can be added deliberately, with whatever
invite or domain restriction the deployment actually needs.
"""

import getpass
import sys
import uuid

from app.auth import hash_password
from app.db import create_user, get_user_by_email, init_db


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    conn = init_db()
    try:
        if "--list" in sys.argv:
            rows = conn.execute(
                "SELECT email, name, role, created_at FROM users ORDER BY created_at"
            ).fetchall()
            if not rows:
                print("No users yet.")
            for row in rows:
                print(f"  {row['email']:32} {row['name'] or '':22} {row['role']}")
            return

        if not args:
            print(__doc__)
            sys.exit(1)

        email = args[0]
        name = args[1] if len(args) > 1 else email.split("@")[0]

        if get_user_by_email(conn, email) is not None:
            print(f"A user with email {email} already exists.")
            sys.exit(1)

        if "--password" in sys.argv:
            # Convenient for scripting a demo, and it puts the password in
            # your shell history. Fine here; not a habit to carry to a real
            # system.
            password = sys.argv[sys.argv.index("--password") + 1]
        else:
            password = getpass.getpass("Password: ")
            if password != getpass.getpass("Confirm: "):
                print("Passwords do not match.")
                sys.exit(1)

        if len(password) < 8:
            print("Password must be at least 8 characters.")
            sys.exit(1)

        create_user(conn, str(uuid.uuid4()), email, name, hash_password(password))
        print(f"Created {email}.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

"""
Getting plain text out of files.

Kept separate from ingest.py so that supporting a new file type is a change
in exactly one place, and so the ingest logic never has to care what format
anything arrived in.
"""

import hashlib
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def file_hash(path: Path) -> str:
    """SHA-256 of the file's bytes.

    Used to skip re-embedding files that have not changed since the last
    ingest. Hashing the CONTENTS rather than checking the modification time
    means a file that was touched, or copied, or checked out of git again
    without changing does not trigger pointless work -- and a file whose
    contents changed while somehow keeping its mtime still does.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in blocks rather than all at once, so a very large file does
        # not have to fit in memory.
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_text(path: Path) -> str:
    """Extract plain text from a supported file."""
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        # errors="replace" rather than letting a stray byte raise: one
        # malformed character should not abort ingestion of a whole corpus.
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        # PDF is a LAYOUT format, not a text format -- it stores glyphs at
        # coordinates, so extraction is always a reconstruction and is
        # frequently imperfect (tables collapse, columns interleave,
        # scanned pages yield nothing at all because there is no text layer,
        # only pixels -- those need OCR). If retrieval quality on PDFs
        # disappoints, always check what was actually extracted before
        # blaming the embedding model.
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    raise ValueError(f"unsupported file type: {suffix}")


def discover(directory: Path) -> list[Path]:
    """Every supported file under `directory`, recursively, sorted."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )

"""
Ingest company documents into the RAG index.

Usage (from the project root):
    python -m scripts.ingest_documents                    # sample_data/documents
    python -m scripts.ingest_documents path/to/docs       # a different folder
    python -m scripts.ingest_documents path/to/docs --force

--force re-processes files whose contents have not changed. You need it
after changing CHUNK_SIZE or EMBEDDING_MODEL_NAME in app/config.py, because
then the source file is identical but the chunks and vectors derived from
it are stale.
"""

import sys
from pathlib import Path

from app.rag.ingest import DEFAULT_DOCS_DIR, index_stats, ingest_directory


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv

    directory = Path(args[0]) if args else DEFAULT_DOCS_DIR

    result = ingest_directory(directory, force=force)
    print()
    print(f"Processed {result['documents']} document(s), "
          f"skipped {result['skipped']} unchanged.")

    stats = index_stats()
    # Broken down by source: the index is shared with meeting transcripts,
    # so a bare total would look wrong to anyone who just ingested 4 files.
    print(f"Index now holds {stats['indexed_chunks']} chunks: "
          f"{stats['chunks_from_documents']} from {stats['documents']} document(s), "
          f"{stats['chunks_from_transcripts']} from "
          f"{stats['meetings_indexed']} meeting(s).")
    if not stats["in_sync"]:
        print("WARNING: SQLite and FAISS disagree on chunk count. "
              "Re-run with --force.")


if __name__ == "__main__":
    main()

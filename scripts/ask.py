"""
Ask a question against the ingested documents, from the terminal.

Usage (from the project root):
    python -m scripts.ask "what is the deployment process?"
    python -m scripts.ask "..." --no-llm     # show retrieval only

--no-llm skips generation and prints the raw retrieved chunks. Use it when
an answer looks wrong: it tells you immediately whether the problem is bad
RETRIEVAL (the right text was never found) or bad GENERATION (the right
text was found and the model still got it wrong). Those have completely
different fixes, and guessing which one you have wastes hours.
"""

import sys

from app.llm import ollama_client
from app.rag.qa import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, answer_question_stream
from app.rag.retrieve import retrieve


def show_sources(hits) -> None:
    if not hits:
        print("No chunks cleared the relevance threshold "
              f"(min_score={DEFAULT_MIN_SCORE}).")
        return
    print("Sources:")
    for i, hit in enumerate(hits, start=1):
        print(f"  [{i}] {hit.document_title} "
              f"(chunk {hit.chunk_index}, score {hit.score:.3f})")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(1)

    question = " ".join(args)

    if "--no-llm" in sys.argv:
        hits = retrieve(question, top_k=DEFAULT_TOP_K, min_score=DEFAULT_MIN_SCORE)
        show_sources(hits)
        print()
        for i, hit in enumerate(hits, start=1):
            print(f"--- [{i}] {hit.document_title} (score {hit.score:.3f}) ---")
            print(hit.text)
            print()
        return

    if not ollama_client.is_available():
        print("Ollama is not running, or the model is not pulled.")
        print("  brew services start ollama")
        print("  ollama pull qwen2.5:7b-instruct")
        print("\nFalling back to retrieval only:\n")
        show_sources(retrieve(question, top_k=DEFAULT_TOP_K,
                              min_score=DEFAULT_MIN_SCORE))
        sys.exit(1)

    hits, stream = answer_question_stream(question)
    show_sources(hits)
    print("\nAnswer:")
    for fragment in stream:
        # flush=True so text appears as it arrives rather than sitting in
        # Python's output buffer until the line ends -- otherwise streaming
        # looks exactly like not streaming.
        print(fragment, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()

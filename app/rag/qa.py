"""
The full RAG pipeline, end to end:

    question -> retrieve -> build prompt -> generate -> answer + sources

This is the file that makes "RAG" concrete. Everything upstream of it is
plumbing; everything about answer quality is decided here or in retrieval.
"""

from dataclasses import dataclass
from typing import Iterator

from app.llm import ollama_client, prompts
from app.rag.retrieve import Hit, retrieve

# Chunks scoring below this are treated as irrelevant. See the note on
# min_score in app/rag/retrieve.py -- calibrate against your own corpus.
DEFAULT_MIN_SCORE = 0.25
DEFAULT_TOP_K = 5


@dataclass
class Answer:
    text: str
    sources: list[Hit]
    used_context: bool  # False when nothing cleared the relevance floor


def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> Answer:
    hits = retrieve(question, top_k=top_k, min_score=min_score)

    # Refuse BEFORE calling the model, not after. If retrieval found
    # nothing relevant there is no answer to be had, and asking the LLM
    # anyway is exactly how you get a fluent hallucination -- it will
    # happily answer from its training data unless it is stopped here.
    # This early return costs nothing and removes a whole class of failure.
    if not hits:
        return Answer(text=prompts.NO_CONTEXT_ANSWER, sources=[], used_context=False)

    text = ollama_client.chat(
        system=prompts.RAG_SYSTEM_PROMPT,
        user=prompts.build_rag_user_prompt(question, hits),
    )
    return Answer(text=text, sources=hits, used_context=True)


def answer_question_stream(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> tuple[list[Hit], Iterator[str]]:
    """Streaming variant.

    Returns the sources immediately and the answer as a generator, so a UI
    can render "Searching 3 documents..." with the real titles while the
    model is still writing. Retrieval takes milliseconds; generation takes
    seconds. No reason to make the user wait for the fast part.
    """
    hits = retrieve(question, top_k=top_k, min_score=min_score)

    if not hits:
        return [], iter([prompts.NO_CONTEXT_ANSWER])

    stream = ollama_client.chat_stream(
        system=prompts.RAG_SYSTEM_PROMPT,
        user=prompts.build_rag_user_prompt(question, hits),
    )
    return hits, stream

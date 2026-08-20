"""
Prompt construction.

Prompts are treated here as code, not as strings scattered through the
codebase: they live in one file, they are built by functions with
arguments, and they can be diffed and reviewed. When an answer comes out
wrong, the prompt is usually the thing you change, so it needs to be as
easy to find and modify as any other logic.
"""

RAG_SYSTEM_PROMPT = """You are a meeting assistant. You answer questions \
using ONLY the context provided to you.

The context may contain two kinds of source:
- Written company documents. Treat these as authoritative.
- Meeting transcripts, produced by automatic speech recognition. These are \
what people actually SAID, and they contain recognition errors: misheard \
words, missing words, wrong names. Read them for meaning rather than \
quoting them word for word, and do not treat an odd phrase as a technical \
term. Where a transcript contradicts a document, say so rather than \
silently picking one.

Rules:
- If the context does not contain the answer, say exactly: "I don't have \
that in the provided documents." Do not guess, and do not use knowledge \
from your training.
- Cite the source of each claim using the [n] numbers given in the context.
- Be concise and factual. Do not pad the answer.
- If the context is only partly relevant, answer the part you can and say \
plainly what is missing."""


def build_context(hits) -> str:
    """Format retrieved chunks into a numbered block for the prompt.

    The numbering is what makes citations possible. The model cannot see
    your database, so "[1]" has to be defined inside the prompt itself; then
    the caller maps those numbers back to real sources for the user.

    Ordering matters more than it should: models attend most strongly to the
    beginning and end of a long context and are measurably weaker in the
    middle -- the "lost in the middle" effect. Retrieval already returns
    best-first, so we keep that order and put the strongest match where it
    is most likely to be used.
    """
    blocks = []
    for i, hit in enumerate(hits, start=1):
        kind = "meeting transcript" if hit.source_type == "transcript" else "document"
        # Labelling the KIND of each source, not just its name, is what lets
        # the system prompt's instruction about ASR errors actually apply --
        # the model cannot treat transcripts differently if it cannot tell
        # which ones they are.
        blocks.append(
            f"[{i}] ({kind}: {hit.citation}, relevance {hit.score:.2f})\n{hit.text}"
        )
    return "\n\n".join(blocks)


def build_rag_user_prompt(question: str, hits) -> str:
    """Assemble the user turn: the context, then the question.

    Question LAST, after the context, on purpose. It is the final thing the
    model reads before it starts generating, which is the position with the
    most influence on what comes out. It also keeps the long, stable context
    block at the front -- the shape you want if you later add prompt
    caching, since a cache only helps on a shared prefix.
    """
    return (
        "Context:\n"
        f"{build_context(hits)}\n\n"
        "---\n"
        f"Question: {question}"
    )


NO_CONTEXT_ANSWER = "I don't have that in the provided documents."

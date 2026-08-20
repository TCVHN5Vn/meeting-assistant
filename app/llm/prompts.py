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


LIVE_SYSTEM_PROMPT = """You are a meeting assistant, answering a question \
that was asked out loud during a meeting that is still in progress.

You are given two things:
- RECENT DISCUSSION: a verbatim transcript of the last few minutes of this \
meeting. This is what is happening right now.
- RETRIEVED CONTEXT: passages from company documents and from earlier \
meetings, found by searching for the question. May be empty.

Both are automatic speech recognition output where they come from audio, so \
they contain recognition errors. Read for meaning; do not treat an odd \
phrase as a technical term.

Rules:
- Answer from what you are given. If neither source answers it, say so \
plainly and briefly.
- Prefer the recent discussion for anything about what was just said, \
decided or asked in this meeting. Prefer the retrieved context for policy, \
process and historical fact.
- Cite retrieved passages using their [n] numbers. The recent discussion \
needs no citation.
- Be brief. This is being read mid-meeting by someone who is also trying to \
follow a conversation. Two or three sentences unless more is genuinely \
needed."""


def build_live_user_prompt(question: str, hits, recent_transcript: str) -> str:
    """Assemble the prompt for a question asked during a live meeting.

    RECENT DISCUSSION IS NOT RETRIEVED, IT IS PASTED IN

    Worth being deliberate about, because the reflex is to run everything
    through the retriever. The last few minutes of the meeting are a small,
    bounded, guaranteed-relevant piece of text that comfortably fits in the
    context window -- so there is nothing for retrieval to do except risk
    leaving out the sentence that was just spoken.

    Retrieval exists to find the few relevant passages among thousands that
    do NOT fit. Recent context is the opposite case, and using RAG for it
    would be slower and strictly worse. Not everything needs retrieval.

    (There is also a practical reason: this meeting is still in progress, so
    its transcript is not in the index yet. Indexing runs when the meeting
    ends.)
    """
    parts = []
    if recent_transcript.strip():
        parts.append("RECENT DISCUSSION (this meeting, last few minutes):\n"
                     + recent_transcript.strip())
    if hits:
        parts.append("RETRIEVED CONTEXT:\n" + build_context(hits))
    if not parts:
        parts.append("(No context available.)")

    # Question last, as in the standard prompt: it is the final thing read
    # before generation begins, which is the position with most influence.
    return "\n\n---\n\n".join(parts) + f"\n\n---\n\nQuestion asked aloud: {question}"


NO_CONTEXT_ANSWER = "I don't have that in the provided documents."

"""
A thin client for the local Ollama HTTP API.

Ollama runs as a background service on this machine and exposes an HTTP
API on port 11434. Nothing leaves the laptop: no API key, no per-token
cost, and meeting audio and company documents never touch a third party --
which for a meeting recorder is a genuine selling point, not just a
cost saving.

The tradeoff is quality and speed. A 7B model running on a laptop CPU/GPU
is meaningfully weaker than a frontier hosted model and slower to first
token. Keeping ALL model access behind this one small module is what makes
that a reversible decision: swapping in a hosted API later means writing
one more file with the same three functions, not touching the pipeline.
"""

import json
from typing import Iterator

import httpx

from app.config import OLLAMA_HOST, OLLAMA_MODEL


class OllamaError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: float, stream: bool = False):
    url = f"{OLLAMA_HOST}{path}"
    try:
        if stream:
            return httpx.stream("POST", url, json=payload, timeout=timeout)
        return httpx.post(url, json=payload, timeout=timeout)
    except httpx.ConnectError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Is it running?\n"
            "  Start it with:  brew services start ollama"
        ) from exc


def is_available() -> bool:
    """True if the Ollama service is up and has our model pulled."""
    try:
        response = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=3.0)
        response.raise_for_status()
        names = [m["name"] for m in response.json().get("models", [])]
    except Exception:
        return False
    # Ollama reports names with an explicit tag ("qwen2.5:7b-instruct"),
    # so match on the prefix to tolerate a tag we did not spell out.
    return any(n.split(":")[0] == OLLAMA_MODEL.split(":")[0] for n in names)


def chat(
    system: str,
    user: str,
    temperature: float = 0.1,
    timeout: float = 180.0,
) -> str:
    """Send a prompt, wait for the whole answer, return it as a string.

    temperature=0.1 rather than the usual 0.7 default. Temperature controls
    how much randomness is injected when picking each next token. For
    creative writing you want some. For "answer strictly from this context"
    and "extract these fields as JSON" you want the model to take the most
    probable path every time -- both for accuracy and so that the same
    input gives you the same output, which is the only way to debug a
    prompt or write a test against one.
    """
    response = _post(
        "/api/chat",
        {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise OllamaError(f"Ollama returned {response.status_code}: {response.text}")

    return response.json()["message"]["content"].strip()


def chat_json(
    system: str,
    user: str,
    schema: dict,
    timeout: float = 300.0,
) -> dict:
    """Ask for a response that satisfies a JSON schema, and parse it.

    Ollama's `format` field takes a JSON schema and CONSTRAINS DECODING to
    it -- at each step the sampler is restricted to tokens that can still
    lead to a valid document. So the result is guaranteed to parse and to
    have the declared shape. That is categorically stronger than asking
    nicely in the prompt and retrying on failure, which is what you are
    stuck with against an API that has no such feature.

    What it does NOT guarantee is that the CONTENT is true. The shape is
    enforced; the values are still generated. Asked for a due date that
    was never mentioned, a constrained model will not return malformed
    JSON -- it will invent a plausible date, because the schema said a
    string goes there. Observed in practice: a task with no stated deadline
    came back with due="soon".

    Which is the general lesson. Structured output removes parsing errors,
    not hallucination. Verification still has to happen afterwards, in
    code -- see verify_quote in app/tasks.py.

    temperature=0.0 rather than the 0.1 used elsewhere: extraction should
    be reproducible, so that a changed result means a changed prompt.
    """
    response = _post(
        "/api/chat",
        {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0.0},
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise OllamaError(f"Ollama returned {response.status_code}: {response.text}")

    content = response.json()["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        # Should be unreachable given constrained decoding, but a truncated
        # response (hitting the context limit mid-document) can still get
        # here, and silently returning nothing would look like "no tasks
        # found" rather than "the model was cut off".
        raise OllamaError(f"model returned unparseable JSON: {content[:200]}") from exc


def chat_stream(
    system: str,
    user: str,
    temperature: float = 0.1,
    timeout: float = 180.0,
) -> Iterator[str]:
    """Same as chat(), but yields the answer token by token as it is produced.

    A 7B model on a laptop can take 10-20 seconds to finish a paragraph.
    Streaming does not make it faster, but it makes it FEEL fast, because
    text starts appearing in well under a second. In a meeting assistant,
    where someone is waiting mid-conversation, that difference is the
    difference between usable and abandoned.

    Ollama streams NDJSON: one JSON object per line, each carrying the next
    fragment, with a final object where "done" is true.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "options": {"temperature": temperature},
    }
    with _post("/api/chat", payload, timeout=timeout, stream=True) as response:
        if response.status_code != 200:
            response.read()
            raise OllamaError(f"Ollama returned {response.status_code}: {response.text}")

        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            fragment = data.get("message", {}).get("content", "")
            if fragment:
                yield fragment
            if data.get("done"):
                break

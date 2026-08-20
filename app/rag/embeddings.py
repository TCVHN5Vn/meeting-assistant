"""
Turning text into vectors.

WHAT AN EMBEDDING ACTUALLY IS

A function from a piece of text to a fixed-length list of numbers -- here,
384 of them -- produced by a neural network trained so that texts with
similar MEANING land close together in that 384-dimensional space, and
unrelated texts land far apart.

That is the whole trick behind semantic search, and it is why RAG beats
keyword search: "how much annual leave do I get?" and "the holiday
entitlement is 25 days" share almost no words, so a keyword index scores
them at nearly zero. Their embeddings sit close together, so a vector
search finds the second from the first.

The model here runs locally on the CPU. Nothing is sent anywhere.
"""

import numpy as np

from app.config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME

_model = None


def get_model():
    """Load the embedding model once and reuse it (same reason as Whisper)."""
    global _model
    if _model is None:
        # Imported lazily rather than at module top-level: sentence-
        # transformers pulls in torch, which takes seconds to import. No
        # reason to pay that just because something imported this module
        # to read a constant.
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' "
              f"(first run downloads ~90MB, then cached)...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model ready.")
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed many texts at once. Returns shape (len(texts), EMBEDDING_DIM).

    Batching matters: the model runs far faster on 32 texts handed over
    together than on 32 separate calls, because the matrix multiplications
    are done once over a batch rather than once per item.

    normalize_embeddings=True scales every vector to length 1. That is what
    lets us use a plain dot product as cosine similarity later -- see
    app/rag/store.py. Do it here, at the single point where vectors are
    created, so nothing downstream can forget to.

    float32 because that is what FAISS requires; numpy would otherwise
    hand us float64 and FAISS would reject it.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
    )
    return vectors.astype("float32")


def embed_query(text: str) -> np.ndarray:
    """Embed one search query. Returns shape (1, EMBEDDING_DIM).

    The (1, dim) shape rather than (dim,) is deliberate: FAISS's search API
    takes a BATCH of query vectors, so it expects a 2-D array even when
    there is only one query in it.

    Note that queries and documents go through the SAME model. They have to:
    two different models produce two unrelated coordinate systems, and
    distances between them would be meaningless noise. (Some setups do use
    an asymmetric pair of models trained jointly for exactly this -- but
    then they were trained together on purpose.)
    """
    return embed_texts([text])

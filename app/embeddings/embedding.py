"""
Embedding module — wraps Ollama-served Qwen3 embeddings for LangChain.

Uses `OllamaEmbeddings` from `langchain-ollama` pointed at a locally
running ``qwen3-embedding:8b`` model.  Exposes a thin factory so the rest
of the codebase gets a ready-to-use `Embeddings` instance without caring
about provider details.
"""

from __future__ import annotations

import logging

from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_MODEL: str = "qwen3-embedding:8b"
DEFAULT_BASE_URL: str = "http://localhost:11434"


class QwenEmbeddingError(Exception):
    """Raised when the embedding model is unreachable or misconfigured."""


def get_embedding_model(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> OllamaEmbeddings:
    """
    Return a configured `OllamaEmbeddings` instance.

    Parameters
    ----------
    model : str
        Ollama model tag (default ``"qwen3-embedding:8b"``).
    base_url : str
        Ollama server URL (default ``"http://localhost:11434"``).

    Returns
    -------
    OllamaEmbeddings
        Ready-to-use LangChain embeddings object.

    Raises
    ------
    QwenEmbeddingError
        If a quick smoke-test embed call fails (server down, model not
        pulled, etc.).
    """
    embeddings = OllamaEmbeddings(model=model, base_url=base_url)

    # Smoke-test: embed a single token to fail fast if Ollama isn't ready.
    try:
        embeddings.embed_query("ping")
    except Exception as exc:
        raise QwenEmbeddingError(
            f"Cannot reach Ollama at {base_url} with model '{model}'. "
            f"Make sure `ollama serve` is running and the model is pulled. "
            f"Original error: {exc}"
        ) from exc

    logger.info("Embedding model ready: %s @ %s", model, base_url)
    return embeddings


# Convenience alias expected by __init__.py
QwenEmbedding = get_embedding_model

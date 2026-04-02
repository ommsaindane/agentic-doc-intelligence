"""
Embedding module — wraps Ollama-served Qwen3 embeddings for LangChain.

Uses `OllamaEmbeddings` from `langchain-ollama` pointed at a locally
running ``qwen3-embedding:8b`` model.  Exposes a thin factory so the rest
of the codebase gets a ready-to-use `Embeddings` instance without caring
about provider details.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_MODEL: str = "qwen3-embedding:8b"
DEFAULT_BASE_URL: str = "http://127.0.0.1:11434"


class QwenEmbeddingError(Exception):
    """Raised when the embedding model is unreachable or misconfigured."""


def get_embedding_model(
    model: str | None = None,
    base_url: str | None = None,
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
    resolved_model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL
    resolved_base_url = (
        base_url
        or os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_HOST")
        or DEFAULT_BASE_URL
    )

    # Allow passing just "host:port".
    parsed = urlparse(resolved_base_url)
    if not parsed.scheme:
        resolved_base_url = f"http://{resolved_base_url}"

    embeddings = OllamaEmbeddings(model=resolved_model, base_url=resolved_base_url)

    # Smoke-test: embed a single token to fail fast if Ollama isn't ready.
    try:
        embeddings.embed_query("ping")
    except Exception as exc:
        raise QwenEmbeddingError(
            f"Cannot reach Ollama at {resolved_base_url} with model '{resolved_model}'. "
            f"Make sure Ollama is running and the model is pulled. "
            f"Original error: {exc}"
        ) from exc

    logger.info("Embedding model ready: %s @ %s", resolved_model, resolved_base_url)
    return embeddings


# Convenience alias expected by __init__.py
QwenEmbedding = get_embedding_model

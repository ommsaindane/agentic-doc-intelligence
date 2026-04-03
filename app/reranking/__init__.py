"""Reranking modules.

Optional Tier-1 component: cross-encoder reranking of retrieval candidates.

This package is intentionally small and deterministic:
- No network calls beyond model loading via Hugging Face (when enabled).
- No randomness in inference (model.eval()).
"""

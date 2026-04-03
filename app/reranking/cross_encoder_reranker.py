from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class RerankResult:
    doc: Document
    score: float


class CrossEncoderReranker:
    """Deterministic cross-encoder reranker.

    Designed for models like `BAAI/bge-reranker-base`.

    Notes
    -----
    - Uses raw logits as scores (monotonic w.r.t. sigmoid).
    - Runs with `model.eval()` and `torch.no_grad()`.
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        if not model_name:
            raise ValueError("model_name is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if max_length <= 0:
            raise ValueError("max_length must be > 0")

        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, *, query: str, candidates: Iterable[Document]) -> list[RerankResult]:
        if query is None or not str(query).strip():
            raise ValueError("query must be non-empty")

        docs = list(candidates)
        if not docs:
            return []

        results: list[RerankResult] = []
        with torch.no_grad():
            for start in range(0, len(docs), self.batch_size):
                batch_docs = docs[start : start + self.batch_size]
                pairs = [(query, d.page_content or "") for d in batch_docs]

                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)

                logits = outputs.logits
                if logits.ndim == 2 and logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                scores = logits.detach().float().cpu().tolist()

                if isinstance(scores, float):
                    scores = [float(scores)]

                for doc, score in zip(batch_docs, scores, strict=True):
                    results.append(RerankResult(doc=doc, score=float(score)))

        return results

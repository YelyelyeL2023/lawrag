"""Cross-encoder reranking.

Runs on CPU by default. On an 8 GB card the generator needs every megabyte, and
reranking 30–40 candidates on a high-core Ryzen costs a second or two — an
acceptable trade for a lookup tool where users expect to wait a moment, and a
bad trade for the VRAM it would otherwise consume.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from ..config import RerankConfig
from ..inference import MODEL_LOCK
from ..logging_setup import get_logger
from .hybrid import Hit

log = get_logger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]: ...


class PassthroughReranker:
    """Keeps the fused order. Correct behaviour when no reranker is installed —
    silently returning an unranked list would look identical from the outside."""

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        return list(hits)[:top_k]


class FlagReranker:
    """bge-reranker-v2-m3 (or any FlagEmbedding cross-encoder), loaded locally."""

    def __init__(self, cfg: RerankConfig) -> None:
        from FlagEmbedding import FlagReranker as _FlagReranker

        log.info("loading reranker %s on %s", cfg.model_path, cfg.device)
        self.cfg = cfg
        self._model = _FlagReranker(cfg.model_path,
                                    use_fp16=cfg.device.startswith("cuda"),
                                    device=cfg.device)

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        window = list(hits)[: self.cfg.top_k_in]
        pairs = [[query, f"{h.heading}\n{h.display_text}"] for h in window]
        with MODEL_LOCK:
            scores = self._model.compute_score(pairs, normalize=False)
        if isinstance(scores, float):
            scores = [scores]
        for hit, score in zip(window, scores):
            hit.rerank_score = float(score)
        return sorted(window, key=lambda h: -(h.rerank_score or 0.0))[:top_k]


class Qwen3Reranker:
    """Qwen3-Reranker-0.6B via sentence-transformers CrossEncoder."""

    def __init__(self, cfg: RerankConfig) -> None:
        from sentence_transformers import CrossEncoder

        self.cfg = cfg
        self._model = CrossEncoder(cfg.model_path, device=cfg.device)

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        window = list(hits)[: self.cfg.top_k_in]
        pairs = [(query, f"{h.heading}\n{h.display_text}") for h in window]
        with MODEL_LOCK:
            scores = self._model.predict(pairs, show_progress_bar=False)
        for hit, score in zip(window, scores):
            hit.rerank_score = float(score)
        return sorted(window, key=lambda h: -(h.rerank_score or 0.0))[:top_k]


def build_reranker(cfg: RerankConfig) -> Reranker:
    backend = cfg.backend.lower()
    if backend in ("none", "off", ""):
        return PassthroughReranker()
    if backend in ("bge-reranker", "bge", "flag"):
        return FlagReranker(cfg)
    if backend in ("qwen3-reranker", "qwen"):
        return Qwen3Reranker(cfg)
    raise ValueError(f"unknown rerank backend: {cfg.backend}")

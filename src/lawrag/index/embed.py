"""Embedding backends.

Two are shipped. ``bge-m3`` is the production choice: multilingual, 8192-token
context, and it emits dense and lexical weights from one pass, which is exactly
the shape the hybrid retriever wants. ``hash`` is a deterministic, dependency-
free stand-in so that a fresh clone can run and be tested end-to-end with no
model weights present — useful on the connected side, in CI, and when
diagnosing whether a retrieval problem is the pipeline or the model.

The hash backend is a real bag-of-features model, not random noise: it hashes
word tokens and character trigrams into buckets with sublinear term weighting.
It retrieves sensibly on small corpora and badly on large ones, which is the
honest behaviour for a placeholder.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

import numpy as np

from ..config import EmbeddingConfig
from ..logging_setup import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[\wЀ-ӿ]+", re.UNICODE)


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic hashed bag-of-features. No weights, no downloads."""

    name = "hash-v1"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    @staticmethod
    def _features(text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        feats: list[str] = list(tokens)
        for token in tokens:
            # Trigrams give partial credit across Kazakh case endings and
            # Russian declensions, where whole-token match fails.
            padded = f"^{token}$"
            feats.extend(padded[i:i + 3] for i in range(max(1, len(padded) - 2)))
        return feats

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                bucket = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                counts[bucket] = counts.get(bucket, 0.0) + sign
            for bucket, value in counts.items():
                out[row, bucket] = math.copysign(math.log1p(abs(value)), value)
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class BGEM3Embedder:
    """FlagEmbedding BGE-M3. Loaded from a local path; never downloads."""

    name = "bge-m3"

    def __init__(self, cfg: EmbeddingConfig) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self.dim = cfg.dim
        self.cfg = cfg
        log.info("loading BGE-M3 from %s on %s", cfg.model_path, cfg.device)
        self._model = BGEM3FlagModel(
            cfg.model_path,
            use_fp16=cfg.device.startswith("cuda"),
            device=cfg.device,
            normalize_embeddings=cfg.normalize,
        )

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        out = self._model.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            max_length=512 if is_query else self.cfg.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.asarray(out["dense_vecs"], dtype=np.float32)


class SentenceTransformerEmbedder:
    """Generic fallback for any local sentence-transformers model."""

    def __init__(self, cfg: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.name = cfg.model_path.rstrip("/").rsplit("/", 1)[-1]
        self._model = SentenceTransformer(cfg.model_path, device=cfg.device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), batch_size=self.cfg.batch_size,
                               normalize_embeddings=self.cfg.normalize,
                               show_progress_bar=False),
            dtype=np.float32,
        )


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    backend = cfg.backend.lower()
    if backend == "hash":
        log.warning(
            "using the hash embedder — a placeholder for pipeline testing. "
            "Set embedding.backend=bge-m3 before this system answers anything real."
        )
        return HashEmbedder(cfg.dim)
    if backend in ("bge-m3", "bgem3"):
        return BGEM3Embedder(cfg)
    if backend in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(cfg)
    raise ValueError(f"unknown embedding backend: {cfg.backend}")

"""Embedding backends.

Two are shipped. ``bge-m3`` is the production choice: multilingual, 8192-token
context, and it emits dense and lexical weights from one pass, which is exactly
the shape the hybrid retriever wants. ``hash`` is a deterministic, dependency-
free stand-in so that a fresh clone can run and be tested end-to-end with no
model weights present — useful on the connected side, in CI, and when
diagnosing whether a retrieval problem is the pipeline or the model.

``gemini`` is the hosted ``gemini-embedding-2`` model. It exists as the
commercial-API baseline the fine-tuned bge-m3 is compared against
(``lawrag train-eval gemini ...``); as a production backend it sends every
chunk and every query to Google.

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
from ..gemini_http import post_with_retry, require_api_key
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
        self._fp16 = cfg.device.startswith("cuda")
        log.info("loading BGE-M3 from %s on %s", cfg.model_path, cfg.device)
        self._model = BGEM3FlagModel(
            cfg.model_path,
            use_fp16=self._fp16,
            device=cfg.device,
            normalize_embeddings=cfg.normalize,
        )
        self._model_fp32 = None
        self.fp32_fallbacks = 0

    def _encode_with(self, model, texts: Sequence[str], is_query: bool) -> np.ndarray:
        out = model.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            max_length=512 if is_query else self.cfg.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.asarray(out["dense_vecs"], dtype=np.float32)

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        vectors = self._encode_with(self._model, texts, is_query)
        bad = np.where(~np.isfinite(vectors).all(axis=1))[0]
        if len(bad) == 0 or not self._fp16:
            return vectors
        # Half precision overflows on some long statutory chunks (a few per
        # code, and only in particular batch compositions), and the result
        # is a NaN vector that Postgres refuses. Those rows are recomputed in
        # full precision by a second copy of the model on the same device:
        # rare enough that the extra memory is the cheaper trade against
        # running everything in fp32.
        if self._model_fp32 is None:
            from FlagEmbedding import BGEM3FlagModel

            log.warning("fp16 embedding produced %d non-finite vector(s); loading an fp32 "
                        "copy of the model for those rows", len(bad))
            self._model_fp32 = BGEM3FlagModel(
                self.cfg.model_path, use_fp16=False, device=self.cfg.device,
                normalize_embeddings=self.cfg.normalize,
            )
        redo = self._encode_with(self._model_fp32, [texts[i] for i in bad], is_query)
        vectors[bad] = redo
        self.fp32_fallbacks += len(bad)
        still = np.where(~np.isfinite(vectors).all(axis=1))[0]
        if len(still):
            raise ValueError(f"embedding is non-finite even in fp32 for {len(still)} text(s); "
                             f"first: {texts[still[0]][:120]!r}")
        return vectors


class GeminiEmbedder:
    """gemini-embedding-2 over batchEmbedContents, truncated to ``cfg.dim``.

    The model takes no task_type parameter; the task goes into the text, in
    Google's asymmetric retrieval format (query prefix vs. document header).
    """

    QUERY_TEMPLATE = "task: search result | query: {text}"
    DOCUMENT_TEMPLATE = "title: none | text: {text}"

    def __init__(self, cfg: EmbeddingConfig) -> None:
        import httpx

        api_key = require_api_key(cfg.api_key, "LAWRAG_EMBEDDING_API_KEY")
        self.cfg = cfg
        self.dim = cfg.dim
        self.name = f"gemini-{cfg.api_model}-{cfg.dim}"
        self._model_ref = f"models/{cfg.api_model}"
        self._client = httpx.Client(
            base_url=cfg.api_base_url.rstrip("/"),
            timeout=cfg.timeout_s,
            headers={"x-goog-api-key": api_key},
        )

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        template = self.QUERY_TEMPLATE if is_query else self.DOCUMENT_TEMPLATE
        size = self.cfg.api_batch_size
        rows = [self._embed_batch([template.format(text=t) for t in texts[i:i + size]])
                for i in range(0, len(texts), size)]
        out = np.vstack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        payload = {"requests": [
            {"model": self._model_ref, "content": {"parts": [{"text": t}]},
             "output_dimensionality": self.dim}
            for t in texts
        ]}
        data = post_with_retry(self._client, f"/{self._model_ref}:batchEmbedContents", payload,
                               max_retries=self.cfg.max_retries,
                               backoff_s=self.cfg.retry_backoff_s)
        return np.asarray([e["values"] for e in data["embeddings"]], dtype=np.float32)


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
    if backend == "gemini":
        return GeminiEmbedder(cfg)
    if backend in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(cfg)
    raise ValueError(f"unknown embedding backend: {cfg.backend}")

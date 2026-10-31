"""Retrieval evaluation with no database: dense search over the chunk corpus.

Ranks every corpus chunk for every dev query by cosine similarity and reports,
per pair kind, how often the right article is at rank 1, in the top 5, in the
top 10, and the mean reciprocal rank. The unit of correctness is the article
(any chunk of it, in the positive's language), which is what the packet builder
works in.

This measures the dense branch alone. The production retriever also runs the
lexical and citation branches, fusion, the relevance floor and the reranker,
so absolute numbers here are lower than end-to-end recall; the *difference*
between two embedders on the same corpus and queries is what it is for.

The model argument ``gemini`` selects the hosted gemini-embedding-2
(``embedding.api_*`` settings) instead of a local directory, as the
commercial-API baseline; ``bm25`` selects the lexical baseline
(``lexical.py``), scored by the same harness on the same queries.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..config import EmbeddingConfig
from ..index.embed import GeminiEmbedder
from ..logging_setup import get_logger
from .lexical import BM25, BM25_MODEL
from .pairs import Pair

log = get_logger(__name__)

GEMINI_MODEL = "gemini"
# Corpus encoding is saved every this many rows, so an interrupted run over a
# paid API resumes instead of paying for the same chunks twice.
CHECKPOINT_ROWS = 2000


@dataclass(slots=True)
class KindResult:
    kind: str
    n: int = 0
    r1: float = 0.0
    r5: float = 0.0
    r10: float = 0.0
    mrr10: float = 0.0

    def to_dict(self) -> dict:
        return {"kind": self.kind, "n": self.n, "recall@1": round(self.r1, 4),
                "recall@5": round(self.r5, 4), "recall@10": round(self.r10, 4),
                "mrr@10": round(self.mrr10, 4)}


@dataclass(slots=True)
class EvalResult:
    model: str
    per_kind: list[KindResult] = field(default_factory=list)
    per_lang: list[KindResult] = field(default_factory=list)
    overall: KindResult = field(default_factory=lambda: KindResult("all"))
    # One record per query (query, kind, lang, article, rank, top1): what the
    # error analysis and the bootstrap confidence intervals are computed from.
    queries: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"model": self.model, "overall": self.overall.to_dict(),
                "per_kind": [k.to_dict() for k in self.per_kind],
                "per_lang": [k.to_dict() for k in self.per_lang]}

    @classmethod
    def from_queries(cls, model: str, queries: list[dict]) -> "EvalResult":
        """Rebuild the summaries from per-query records — for systems whose
        ranks come from a reranker rather than from ``rank_pairs``, and for
        results read back from disk."""
        res = cls(model=model, queries=queries)
        ranks = np.array([q["rank"] for q in queries], dtype=np.int64)
        for attr, field_ in (("per_kind", "kind"), ("per_lang", "lang")):
            values = np.array([q[field_] for q in queries])
            for v in sorted(set(values.tolist())):
                getattr(res, attr).append(summarise(v, ranks[values == v]))
        res.overall = summarise("all", ranks)
        return res


def ranks_from_scores(scores: np.ndarray, corpus_keys: Sequence[tuple[str, str]],
                      targets: Sequence[tuple[str, str]]) -> np.ndarray:
    """1-based rank of the first corpus row matching each query's target
    (article, lang); 0 when it is not in the corpus at all. Pure numpy, so the
    metric is testable without a model.

    Equal to the position of the first target row in a stable descending
    sort, without sorting: every row scoring higher, plus every row scoring
    the same that comes earlier in the corpus. The hyper-parameter searches
    call this thousands of times over 43k-row score matrices.
    """
    rows_of: dict[tuple[str, str], list[int]] = {}
    for j, key in enumerate(corpus_keys):
        rows_of.setdefault(key, []).append(j)
    out = np.zeros(len(targets), dtype=np.int64)
    for i, target in enumerate(targets):
        rows = rows_of.get(target)
        if not rows:
            continue
        s = scores[i]
        rows_arr = np.asarray(rows)
        best = s[rows_arr].max()
        first = int(rows_arr[s[rows_arr] == best].min())
        out[i] = 1 + int((s > best).sum()) + int((s[:first] == best).sum())
    return out


def summarise(kind: str, ranks: np.ndarray) -> KindResult:
    n = len(ranks)
    if n == 0:
        return KindResult(kind)
    hit = ranks > 0
    return KindResult(
        kind, n,
        r1=float(np.mean(hit & (ranks <= 1))),
        r5=float(np.mean(hit & (ranks <= 5))),
        r10=float(np.mean(hit & (ranks <= 10))),
        mrr10=float(np.mean(np.where(hit & (ranks <= 10), 1.0 / np.maximum(ranks, 1), 0.0))),
    )


# ------------------------------------------------------------- encoding --

def _tag(model_path: str | Path) -> str:
    return Path(str(model_path)).name or "model"


def corpus_digest(texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def load_model(model_path: str | Path, device: str | None = None, max_seq_length: int = 512):
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(str(model_path), device=device)
    model.max_seq_length = max_seq_length
    if device.startswith("cuda"):
        model.half()
    return model


def load_encoder(model_path: str | Path, device: str | None,
                 embedding_cfg: EmbeddingConfig | None):
    if str(model_path) == BM25_MODEL:
        return BM25()
    if str(model_path) == GEMINI_MODEL:
        return GeminiEmbedder(embedding_cfg or EmbeddingConfig())
    return load_model(model_path, device)


def encode(model, texts: Sequence[str], batch_size: int = 32,
           is_query: bool = False) -> np.ndarray:
    if isinstance(model, GeminiEmbedder):
        return model.encode(texts, is_query=is_query)
    return np.asarray(
        model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False),
        dtype=np.float32,
    )


def encode_corpus(model, model_path: str | Path, texts: Sequence[str],
                  cache_dir: Path | None, batch_size: int = 32) -> np.ndarray:
    """Corpus embeddings, cached per (model, corpus content).

    Encoding forty thousand statutory chunks is minutes on a laptop GPU and
    an hour on a CPU; hard-negative mining and evaluation both need it, and
    both are re-run far more often than the corpus changes.
    """
    tag = model.name if isinstance(model, GeminiEmbedder) else _tag(model_path)
    if cache_dir is None:
        log.info("encoding %d corpus chunks with %s", len(texts), tag)
        return encode(model, texts, batch_size=batch_size)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"emb.{tag}.{corpus_digest(texts)}.npy"
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == len(texts):
            log.info("corpus embeddings from cache %s", cache.name)
            return emb
    log.info("encoding %d corpus chunks with %s", len(texts), tag)
    emb = _encode_checkpointed(model, texts, batch_size, cache.with_suffix(".partial.npy"))
    _save_atomic(cache, emb)
    cache.with_suffix(".partial.npy").unlink(missing_ok=True)
    return emb


def _save_atomic(path: Path, array: np.ndarray) -> None:
    """np.save via a temporary file and a rename, so a crash never leaves a
    truncated cache that the next run would load."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def _encode_checkpointed(model, texts: Sequence[str], batch_size: int,
                         partial: Path) -> np.ndarray:
    blocks = [np.load(partial)] if partial.exists() else []
    done = sum(len(b) for b in blocks)
    if done:
        log.info("resuming corpus encoding at %d/%d", done, len(texts))
    for start in range(done, len(texts), CHECKPOINT_ROWS):
        blocks.append(encode(model, texts[start:start + CHECKPOINT_ROWS], batch_size=batch_size))
        _save_atomic(partial, np.vstack(blocks))
        log.info("encoded %d/%d", min(start + CHECKPOINT_ROWS, len(texts)), len(texts))
    return np.vstack(blocks)


# ------------------------------------------------------------- evaluate --

def rank_pairs(score: Callable[[list[str]], np.ndarray], pairs: list[Pair],
               corpus: list[dict], *, name: str, keep_top: int = 0) -> EvalResult:
    """Rank the corpus for every pair with ``score`` (queries -> n_queries x
    n_chunks) and summarise. Every system — BM25, the embedders, the
    hyper-parameter trials — goes through this one function, so they are
    measured identically.

    ``keep_top`` > 0 also records each query's top chunk indices
    (``"top"``) — the candidate list a second-stage reranker reorders.
    """
    keys = [(c["article"], c["lang"]) for c in corpus]
    langs = np.array([c["lang"] for c in corpus])
    result = EvalResult(model=name)
    all_ranks: list[np.ndarray] = []
    all_items: list[Pair] = []
    for kind in sorted({p.kind for p in pairs}):
        items = [p for p in pairs if p.kind == kind]
        scores = score([p.query for p in items])
        # A cross-language query is the other language's text of the same
        # node, which trivially ranks first; rank it among the target language
        # only, which is also the only competition that matters to it.
        for i, p in enumerate(items):
            if p.lang != p.pos_lang:
                scores[i, langs != p.pos_lang] = -np.inf
        ranks = ranks_from_scores(scores, keys, [(p.article, p.pos_lang) for p in items])
        top1 = np.argmax(scores, axis=1)
        tops = top_k_indices(scores, keep_top) if keep_top else None
        result.per_kind.append(summarise(kind, ranks))
        all_ranks.append(ranks)
        all_items.extend(items)
        for i, (p, r, t) in enumerate(zip(items, ranks, top1)):
            rec = {"query": p.query, "kind": p.kind, "lang": p.lang, "article": p.article,
                   "pos_lang": p.pos_lang, "rank": int(r), "top1": corpus[int(t)]["article"]}
            if tops is not None:
                rec["top"] = [int(j) for j in tops[i]]
                rec["top_scores"] = [float(scores[i, j]) for j in tops[i]]
            result.queries.append(rec)
    if all_ranks:
        ranks = np.concatenate(all_ranks)
        result.overall = summarise("all", ranks)
        for lang in sorted({p.lang for p in all_items}):
            mask = np.array([p.lang == lang for p in all_items])
            result.per_lang.append(summarise(lang, ranks[mask]))
    return result


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Per row, the column indices of the ``k`` highest scores, best first."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def evaluate(model_path: str | Path, pairs: list[Pair], corpus: list[dict], *,
             device: str | None = None, cache_dir: Path | None = None,
             batch_size: int = 32, max_queries: int | None = None,
             seed: int = 13, embedding_cfg: EmbeddingConfig | None = None,
             keep_top: int = 0, bm25: BM25 | None = None) -> EvalResult:
    """Score ``pairs`` against ``corpus`` with one system. ``bm25`` passes a
    configured (e.g. tuned) BM25 instead of the default one."""
    import random

    if max_queries is not None and len(pairs) > max_queries:
        rng = random.Random(seed)
        by_kind: dict[str, list[Pair]] = {}
        for p in pairs:
            by_kind.setdefault(p.kind, []).append(p)
        share = max(1, max_queries // max(1, len(by_kind)))
        pairs = [p for items in by_kind.values() for p in rng.sample(items, min(share, len(items)))]

    model = bm25 if bm25 is not None else load_encoder(model_path, device, embedding_cfg)
    texts = [c["text"] for c in corpus]
    if isinstance(model, BM25):
        # Lexical baseline: "fitting" is building the term statistics of
        # the corpus; there is nothing learned from the training pairs.
        if model._weights is None:
            model.fit(texts)
        score = model.score
    else:
        doc_emb = encode_corpus(model, model_path, texts, cache_dir, batch_size)

        def score(queries: list[str]) -> np.ndarray:
            q_emb = encode(model, queries, batch_size=batch_size, is_query=True)
            return q_emb @ doc_emb.T   # cosine: both sides are L2-normalised

    return rank_pairs(score, pairs, corpus, name=str(model_path), keep_top=keep_top)


def print_results(results: list[EvalResult], console=None) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    for res in results:
        table = Table(title=f"retrieval — {res.model}")
        for col in ("kind", "n", "recall@1", "recall@5", "recall@10", "mrr@10"):
            table.add_column(col, justify="right" if col != "kind" else "left")
        for k in res.per_kind + res.per_lang + [res.overall]:
            table.add_row(k.kind, str(k.n), f"{k.r1:.3f}", f"{k.r5:.3f}", f"{k.r10:.3f}",
                          f"{k.mrr10:.3f}")
        console.print(table)


def save_results(path: Path, results: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=1),
                    encoding="utf-8")

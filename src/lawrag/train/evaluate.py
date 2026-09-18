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
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..logging_setup import get_logger
from .pairs import Pair

log = get_logger(__name__)


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
    overall: KindResult = field(default_factory=lambda: KindResult("all"))

    def to_dict(self) -> dict:
        return {"model": self.model, "overall": self.overall.to_dict(),
                "per_kind": [k.to_dict() for k in self.per_kind]}


def ranks_from_scores(scores: np.ndarray, corpus_keys: Sequence[tuple[str, str]],
                      targets: Sequence[tuple[str, str]]) -> np.ndarray:
    """1-based rank of the first corpus row matching each query's target
    (article, lang); 0 when it is not in the corpus at all. Pure numpy, so the
    metric is testable without a model."""
    out = np.zeros(len(targets), dtype=np.int64)
    keys = list(corpus_keys)
    for i, target in enumerate(targets):
        order = np.argsort(-scores[i], kind="stable")
        for rank, j in enumerate(order, 1):
            if keys[j] == target:
                out[i] = rank
                break
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


def encode(model, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
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
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"emb.{_tag(model_path)}.{corpus_digest(texts)}.npy"
        if cache.exists():
            emb = np.load(cache)
            if emb.shape[0] == len(texts):
                log.info("corpus embeddings from cache %s", cache.name)
                return emb
    log.info("encoding %d corpus chunks with %s", len(texts), _tag(model_path))
    emb = encode(model, texts, batch_size=batch_size)
    if cache_dir is not None:
        np.save(cache, emb)
    return emb


# ------------------------------------------------------------- evaluate --

def evaluate(model_path: str | Path, pairs: list[Pair], corpus: list[dict], *,
             device: str | None = None, cache_dir: Path | None = None,
             batch_size: int = 32, max_queries: int | None = None,
             seed: int = 13) -> EvalResult:
    import random

    if max_queries is not None and len(pairs) > max_queries:
        rng = random.Random(seed)
        by_kind: dict[str, list[Pair]] = {}
        for p in pairs:
            by_kind.setdefault(p.kind, []).append(p)
        share = max(1, max_queries // max(1, len(by_kind)))
        pairs = [p for items in by_kind.values() for p in rng.sample(items, min(share, len(items)))]

    model = load_model(model_path, device)
    texts = [c["text"] for c in corpus]
    keys = [(c["article"], c["lang"]) for c in corpus]
    doc_emb = encode_corpus(model, model_path, texts, cache_dir, batch_size)

    langs = np.array([c["lang"] for c in corpus])
    result = EvalResult(model=str(model_path))
    all_ranks: list[np.ndarray] = []
    for kind in sorted({p.kind for p in pairs}):
        items = [p for p in pairs if p.kind == kind]
        q_emb = encode(model, [p.query for p in items], batch_size=batch_size)
        scores = q_emb @ doc_emb.T
        # A cross-language query is the other language's text of the same
        # node, which trivially ranks first; rank it among the target language
        # only, which is also the only competition that matters to it.
        for i, p in enumerate(items):
            if p.lang != p.pos_lang:
                scores[i, langs != p.pos_lang] = -np.inf
        ranks = ranks_from_scores(scores, keys, [(p.article, p.pos_lang) for p in items])
        result.per_kind.append(summarise(kind, ranks))
        all_ranks.append(ranks)
    if all_ranks:
        result.overall = summarise("all", np.concatenate(all_ranks))
    return result


def print_results(results: list[EvalResult], console=None) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    for res in results:
        table = Table(title=f"dense retrieval — {res.model}")
        for col in ("kind", "n", "recall@1", "recall@5", "recall@10", "mrr@10"):
            table.add_column(col, justify="right" if col != "kind" else "left")
        for k in res.per_kind + [res.overall]:
            table.add_row(k.kind, str(k.n), f"{k.r1:.3f}", f"{k.r5:.3f}", f"{k.r10:.3f}",
                          f"{k.mrr10:.3f}")
        console.print(table)


def save_results(path: Path, results: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=1),
                    encoding="utf-8")

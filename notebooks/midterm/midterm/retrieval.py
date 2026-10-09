"""Ranking the whole chunk index for every query, and the metrics.

Protocol (identical for every model, so numbers are comparable):

* index = all 43 857 chunks of both languages, text = ``embed_text`` (what the
  production index embeds);
* a hit is any chunk of the target article in the target language;
* a cross-language query (RU text -> KK article) is ranked among chunks of the
  target language only. Otherwise its own-language twin is always rank 1 and the
  target can never be first — Assignment 2 ranked it over both languages, which
  is why its xlang numbers were ~0 for a structural, not a modelling, reason.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import EvalConfig, TfidfConfig
from .data import KIND_ORDER

METRICS = ("recall@1", "recall@10", "MRR@10")
LANGS = ("ru", "kk")
ScoreFn = Callable[[pd.DataFrame], np.ndarray]


class Index:
    def __init__(self, chunks: pd.DataFrame):
        self.chunks = chunks
        keys = list(zip(chunks["article"], chunks["lang"]))
        self.key_codes = {k: i for i, k in enumerate(sorted(set(keys)))}
        self.chunk_keys = np.array([self.key_codes[k] for k in keys])
        self.langs = chunks["lang"].to_numpy()

    def target_ids(self, queries: pd.DataFrame) -> np.ndarray:
        return np.array([self.key_codes.get((a, l), -1)
                         for a, l in zip(queries["target_article"], queries["pos_lang"])])


def rank_queries(index: Index, queries: pd.DataFrame, score_fn: ScoreFn, cfg: EvalConfig) -> np.ndarray:
    """1-based rank of the first chunk of the target (article, lang); 0 = not in top-k."""
    targets = index.target_ids(queries)
    ranks = np.zeros(len(queries), dtype=int)
    for start in range(0, len(queries), cfg.query_block):
        block = queries.iloc[start:start + cfg.query_block]
        scores = _mask_cross_language(index, block, score_fn(block))
        top = np.argpartition(-scores, cfg.top_k, axis=1)[:, :cfg.top_k]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1), axis=1)
        for i, row in enumerate(index.chunk_keys[top]):
            hit = np.flatnonzero(row == targets[start + i])
            ranks[start + i] = hit[0] + 1 if hit.size else 0
    return ranks


def top1_articles(index: Index, queries: pd.DataFrame, score_fn: ScoreFn, cfg: EvalConfig) -> np.ndarray:
    articles = index.chunks["article"].to_numpy()
    out = []
    for start in range(0, len(queries), cfg.query_block):
        block = queries.iloc[start:start + cfg.query_block]
        scores = _mask_cross_language(index, block, score_fn(block))
        out.extend(articles[scores.argmax(axis=1)])
    return np.array(out)


def _mask_cross_language(index: Index, block: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    cross = (block["lang"] != block["pos_lang"]).to_numpy()
    wrong_lang = index.langs[None, :] != block["pos_lang"].to_numpy()[:, None]
    scores[cross[:, None] & wrong_lang] = -np.inf
    return scores


def metrics(ranks: np.ndarray) -> dict:
    if len(ranks) == 0:
        return {"n": 0, **{m: np.nan for m in METRICS}}
    hit = ranks > 0
    return {"n": len(ranks), "recall@1": float(np.mean(ranks == 1)), "recall@10": float(np.mean(hit)),
            "MRR@10": float(np.mean(np.where(hit, 1 / np.maximum(ranks, 1), 0)))}


def breakdown(ranks: np.ndarray, queries: pd.DataFrame) -> pd.DataFrame:
    rows = {"all": metrics(ranks)}
    for kind in KIND_ORDER:
        rows[kind] = metrics(ranks[(queries["kind"] == kind).to_numpy()])
    for lang in LANGS:
        rows[f"lang={lang}"] = metrics(ranks[(queries["lang"] == lang).to_numpy()])
    return pd.DataFrame(rows).T.astype({"n": int})


def bootstrap_ci(ranks: np.ndarray, metric: str, n_boot: int = 1000, seed: int = 13,
                 alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = [metrics(ranks[rng.integers(0, len(ranks), len(ranks))])[metric] for _ in range(n_boot)]
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


# ------------------------------------------------------------------ TF-IDF --

def fit_tfidf(index: Index, cfg: TfidfConfig) -> ScoreFn:
    vectorizer = TfidfVectorizer(analyzer=cfg.analyzer, ngram_range=(cfg.ngram_min, cfg.ngram_max),
                                 min_df=cfg.min_df, max_features=cfg.max_features, sublinear_tf=True,
                                 lowercase=True, dtype=np.float32)
    doc_matrix = vectorizer.fit_transform(index.chunks["embed_text"])

    def score(queries: pd.DataFrame) -> np.ndarray:
        return (vectorizer.transform(queries["query"]) @ doc_matrix.T).toarray()

    return score


# ------------------------------------------------------------------- dense --

def encode(model, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
    return np.asarray(model.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)


def _digest(texts: list[str], tag: str) -> str:
    h = hashlib.sha256(tag.encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def encode_corpus(model, tag: str, texts: list[str], prefix: str, batch_size: int, cache_dir: Path) -> np.ndarray:
    """Cached per (model tag, corpus content): 44k chunks take minutes to encode on a laptop."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{tag}.{_digest(texts, prefix)}.npy"
    if cache.exists():
        return np.load(cache)
    emb = encode(model, texts, prefix, batch_size)
    np.save(cache, emb)
    return emb


def dense_scorer(model, doc_emb: np.ndarray, query_prefix: str, batch_size: int) -> ScoreFn:
    def score(queries: pd.DataFrame) -> np.ndarray:
        return encode(model, queries["query"].tolist(), query_prefix, batch_size) @ doc_emb.T

    return score

"""Corpus and (query -> article) pairs with the Assignment 2 cleaning and split.

The cleaning steps and the StratifiedGroupKFold split reproduce
``notebooks/assignment2`` §5 and §8 exactly, so the test set here is the same
3 246 queries the TF-IDF baseline was scored on there. The result is cached as
JSONL: parsing 85 MB of HTML takes about a minute.
"""

from __future__ import annotations

import logging
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import Settings

SPLITS = ("train", "validation", "test")
KIND_ORDER = ("heading", "xref", "definition", "xlang")
EDITORIAL_NOTE_RE = re.compile(r"(вводится в действие с|қолданысқа енгізіледі|предусматрива\w+\s+(исключ|изменени))",
                               re.IGNORECASE)


def _lawrag(settings: Settings):
    src = str(settings.paths.root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from lawrag.config import ChunkConfig
    from lawrag.ingest import structure as S
    from lawrag.train import pairs as P
    return ChunkConfig, S, P


def _normalise(query: str) -> str:
    return " ".join(query.lower().split())


def _number_left_in_query(query: str, article: str) -> bool:
    number = article.split("/st")[1]
    return bool(re.search(rf"(?<![\d-]){re.escape(number)}(?![\d-])", query))


def _clean(pairs_raw: pd.DataFrame, S, P, seed: int) -> pd.DataFrame:
    df = pairs_raw.drop_duplicates(["query", "target_chunk"])
    df = df[df.groupby("query_norm")["target_article"].transform("nunique") <= 1]
    is_xref = df.kind == "xref"
    leaky = is_xref & np.array([_number_left_in_query(q, a) for q, a in zip(df["query"], df["target_article"])])
    df = df[~leaky]
    editorial = (df.kind == "xref") & df["query"].map(
        lambda q: S.is_amendment_line(q) or bool(EDITORIAL_NOTE_RE.search(q)))
    df = df[~editorial]
    capped = P.cap_per_kind([P.Pair(r.query, r.target_chunk, r.target_article, r.lang, r.pos_lang, r.kind)
                             for r in df.itertuples()], P.DEFAULT_CAPS, seed=seed)
    return pd.DataFrame([p.to_dict() for p in capped]).rename(
        columns={"pos": "target_chunk", "article": "target_article"}).reset_index(drop=True)


def _assign_split(pairs: pd.DataFrame, settings: Settings) -> pd.Series:
    cfg = settings.split
    folds = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold_of = np.empty(len(pairs), dtype=int)
    for fold, (_, idx) in enumerate(folds.split(pairs, y=pairs["kind"], groups=pairs["target_article"])):
        fold_of[idx] = fold
    return pd.Series(np.select([fold_of == cfg.test_fold, fold_of == cfg.val_fold], ["test", "validation"], "train"),
                     index=pairs.index)


def _build(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    ChunkConfig, S, P = _lawrag(settings)
    logging.getLogger("lawrag").setLevel(logging.ERROR)
    trees = P.load_trees(settings.paths.adilet)
    corpus = P.build_corpus(trees, ChunkConfig())
    raw = [*P.mine_headings(trees, corpus), *P.mine_xrefs(trees, corpus),
           *P.mine_xlang(corpus), *P.mine_definitions(trees, corpus)]
    pairs_raw = pd.DataFrame([p.to_dict() for p in raw]).rename(
        columns={"pos": "target_chunk", "article": "target_article"})
    pairs_raw["query_norm"] = pairs_raw["query"].map(_normalise)
    pairs = _clean(pairs_raw, S, P, settings.split.seed)
    pairs["split"] = _assign_split(pairs, settings)
    chunks = pd.DataFrame([{
        "chunk_id": c.id, "article": c.article_node_id, "lang": c.lang.value, "act": c.act_code,
        "repealed": c.repealed, "embed_text": c.embed_text, "display_text": c.display_text,
    } for c in corpus.chunks.values()])
    return chunks, pairs


def load(settings: Settings, rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(chunks, pairs). ``pairs.split`` is one of SPLITS; split unit is the article."""
    paths = settings.paths
    if rebuild or not (paths.corpus.exists() and paths.pairs.exists()):
        chunks, pairs = _build(settings)
        paths.work.mkdir(parents=True, exist_ok=True)
        chunks.to_json(paths.corpus, orient="records", lines=True, force_ascii=False)
        pairs.to_json(paths.pairs, orient="records", lines=True, force_ascii=False)
    chunks = pd.read_json(paths.corpus, orient="records", lines=True, dtype=False)
    pairs = pd.read_json(paths.pairs, orient="records", lines=True, dtype=False)
    return chunks, pairs


def split(pairs: pd.DataFrame, name: str) -> pd.DataFrame:
    return pairs[pairs["split"] == name].reset_index(drop=True)

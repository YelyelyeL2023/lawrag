"""The three systems compared in the midterm, each scored by the same protocol.

1. TF-IDF (word 1-2 grams)          — no training, Assignment 2 baseline;
2. multilingual-e5-small, pretrained — zero-shot dense retrieval;
3. multilingual-e5-small, fine-tuned on train pairs — the trained baseline.

Validation picks between variants; test is scored once per system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data as D
from . import retrieval as R
from . import train as T
from .config import Settings

TFIDF = "TF-IDF"
PRETRAINED = "e5-small (pretrained)"
FINETUNED = "e5-small (fine-tuned)"


@dataclass(slots=True)
class System:
    name: str
    score: R.ScoreFn


def tfidf_system(index: R.Index, settings: Settings) -> System:
    return System(TFIDF, R.fit_tfidf(index, settings.tfidf))


def dense_system(name: str, model, tag: str, index: R.Index, settings: Settings) -> tuple[System, np.ndarray]:
    cfg, ev = settings.train, settings.eval
    doc_emb = R.encode_corpus(model, tag, index.chunks["embed_text"].tolist(), cfg.passage_prefix,
                              ev.encode_batch_size, settings.paths.embeddings)
    return System(name, R.dense_scorer(model, doc_emb, cfg.query_prefix, ev.encode_batch_size)), doc_emb


def train_or_load(pretrained, doc_emb: np.ndarray, index: R.Index, pairs: pd.DataFrame,
                  settings: Settings, device: str, force: bool = False):
    """Fine-tuned model; trains only when no saved model exists (or ``force``)."""
    out = settings.paths.model_out
    if force or T.read_meta(out) is None:
        train = D.split(pairs, "train")
        held_out = set(pairs.loc[pairs["split"] != "train", "target_article"])
        negatives = (T.mine_hard_negatives(pretrained, train, index.chunks, doc_emb, held_out, settings)
                     if settings.train.hard_negatives else None)
        dataset = T.build_dataset(train, index.chunks, negatives, settings.train)
        model = T.load_model(settings.train.base_model, settings.train, device)
        T.finetune(model, dataset, settings, device)
    return T.load_model(out, settings.train, device), T.read_meta(out)


def score(system: System, index: R.Index, queries: pd.DataFrame, settings: Settings) -> np.ndarray:
    return R.rank_queries(index, queries, system.score, settings.eval)


def summary_table(results: dict[str, np.ndarray], queries: pd.DataFrame) -> pd.DataFrame:
    return pd.concat({name: R.breakdown(ranks, queries) for name, ranks in results.items()}, axis=1)

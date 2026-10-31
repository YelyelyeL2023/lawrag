"""Classical machine learning: a learned fusion of the lexical and dense rankings.

BM25 and the dense retriever fail on different queries (lexical overlap vs
meaning). Production fuses them with a hand-set rule; here the fusion is
*learned*: a binary classifier over (query, candidate chunk) pairs predicts
"this chunk belongs to the right article", and candidates are re-ordered by
the predicted probability (pointwise learning-to-rank).

* **Candidates** — the union of the top 30 chunks of both rankings.
* **Features** — 12 numbers per candidate, none of them words: both scores,
  both ranks, the score margin to each list's top, reciprocal-rank fusion,
  whether the query names this chunk's article number (the shortcut the
  midterm error analysis found), whether the candidate is in the same code
  as the dense top-1, query/chunk language and length.
* **Models** — ``HistGradientBoostingClassifier`` (tuned by randomized search)
  and ``LogisticRegression`` (tuned by grid search over C), each reported
  with sklearn defaults *before* tuning and with the selected parameters
  *after*.
* **Data** — fitted on the **validation** split only: the dense model never
  trained on those articles, so its scores there look like its scores on
  unseen text. Hyper-parameters are chosen by 5-fold ``GroupKFold`` (grouped
  by article, so no article is on both sides of a fold) on the retrieval
  metric itself, MRR@10 — not on log-loss. The test split is touched once.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..logging_setup import get_logger
from .pairs import Pair

log = get_logger(__name__)

FEATURES = (
    "dense_score", "dense_margin", "dense_log_rank",
    "bm25_norm", "bm25_margin", "bm25_log_rank",
    "rrf", "in_both_top",
    "names_this_article", "same_code_as_dense_top1",
    "query_has_number", "chunk_log_chars",
)
CANDIDATES_PER_LIST = 30

HGB_SPACE = {
    "learning_rate": ("loguniform", 0.02, 0.3),
    "max_leaf_nodes": [7, 15, 31, 63],
    "max_depth": [None, 3, 5, 8],
    "min_samples_leaf": [10, 20, 50, 100],
    "l2_regularization": ("loguniform", 1e-3, 10.0),
    "max_iter": [100, 200, 400],
}
HGB_TRIALS = 30
LR_GRID = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

_ART_NUM = re.compile(r"/st(\d+)")


@dataclass(slots=True)
class Candidates:
    """Every query's candidate rows with their features and labels."""

    query_ids: np.ndarray            # (n_rows,) index of the query
    rows: np.ndarray                 # (n_rows,) corpus row of the candidate
    X: np.ndarray                    # (n_rows, n_features)
    y: np.ndarray                    # (n_rows,) 1 = chunk of the right article
    fallback_rank: np.ndarray        # (n_queries,) rank to report when no candidate is right
    n_candidates: np.ndarray         # (n_queries,)
    groups: list[str] = field(default_factory=list)   # article per query, for GroupKFold


def _ranks_desc(row: np.ndarray) -> np.ndarray:
    """1-based descending rank of every entry (ties broken by position)."""
    order = np.argsort(-row, kind="stable")
    ranks = np.empty(len(row), dtype=np.int64)
    ranks[order] = np.arange(1, len(row) + 1)
    return ranks


def build_candidates(pairs: list[Pair], corpus: list[dict],
                     dense: Callable[[list[str]], np.ndarray],
                     bm25: Callable[[list[str]], np.ndarray],
                     k: int = CANDIDATES_PER_LIST, batch: int = 256) -> Candidates:
    langs = np.array([c["lang"] for c in corpus])
    arts = [c["article"] for c in corpus]
    codes = np.array([a.split("/", 1)[0] for a in arts])
    nums = [m.group(1) if (m := _ART_NUM.search(a)) else None for a in arts]
    log_chars = np.log1p(np.array([len(c["text"]) for c in corpus], dtype=np.float32))
    rows_of: dict[tuple[str, str], list[int]] = {}
    for j, c in enumerate(corpus):
        rows_of.setdefault((c["article"], c["lang"]), []).append(j)
    qid, rows, feats, labels = [], [], [], []
    fallback = np.zeros(len(pairs), dtype=np.int64)
    n_cand = np.zeros(len(pairs), dtype=np.int64)
    for start in range(0, len(pairs), batch):
        items = pairs[start:start + batch]
        texts = [p.query for p in items]
        d_all, b_all = dense(texts), bm25(texts)
        for i, p in enumerate(items):
            d, b = d_all[i].astype(np.float32), b_all[i].astype(np.float32)
            # Same restriction as the evaluation harness: a cross-language
            # query competes only among the target language.
            if p.lang != p.pos_lang:
                off = langs != p.pos_lang
                d, b = d.copy(), b.copy()
                d[off], b[off] = -np.inf, -np.inf
            rd, rb = _ranks_desc(d), _ranks_desc(b)
            cand = np.union1d(np.where(rd <= k)[0], np.where(rb <= k)[0])
            d_top, b_top = d.max(), max(float(b.max()), 1e-6)
            top1_code = codes[int(np.argmax(d))]
            qnums = set(re.findall(r"(?<!\d)\d{1,4}(?!\d)", p.query))
            target = np.array([arts[j] == p.article and langs[j] == p.pos_lang for j in cand])
            x = np.column_stack([
                d[cand], d[cand] - d_top, np.log1p(rd[cand]),
                b[cand] / b_top, (b[cand] - b_top) / b_top, np.log1p(rb[cand]),
                1.0 / (60 + rd[cand]) + 1.0 / (60 + rb[cand]),
                ((rd[cand] <= k) & (rb[cand] <= k)).astype(np.float32),
                np.array([nums[j] is not None and nums[j] in qnums for j in cand], dtype=np.float32),
                (codes[cand] == top1_code).astype(np.float32),
                np.full(len(cand), float(bool(qnums))),
                log_chars[cand],
            ]).astype(np.float32)
            q = start + i
            # When no candidate is right, the true article sits below every
            # candidate: report it there, at its dense rank past the lists.
            tgt_rows = [] if target.any() else rows_of.get((p.article, p.pos_lang), [])
            dense_rank = int(rd[tgt_rows].min()) if tgt_rows else 0
            fallback[q] = len(cand) + max(1, dense_rank - k) if tgt_rows else 0
            n_cand[q] = len(cand)
            qid.append(np.full(len(cand), q))
            rows.append(cand)
            feats.append(x)
            labels.append(target.astype(np.int8))
    return Candidates(np.concatenate(qid), np.concatenate(rows), np.vstack(feats),
                      np.concatenate(labels), fallback, n_cand,
                      groups=[p.article for p in pairs])


def ranks_from_probs(c: Candidates, probs: np.ndarray, query_mask: np.ndarray | None = None
                     ) -> np.ndarray:
    """Article-level rank of the right article per query after re-ordering
    the candidates by ``probs`` (``query_mask`` selects queries)."""
    n_q = len(c.fallback_rank)
    out = c.fallback_rank.copy()
    order = np.lexsort((-probs, c.query_ids))      # by query, then best first
    qs, ys = c.query_ids[order], c.y[order]
    starts = np.searchsorted(qs, np.arange(n_q))
    ends = np.searchsorted(qs, np.arange(n_q), side="right")
    for q in range(n_q):
        hit = np.flatnonzero(ys[starts[q]:ends[q]])
        if len(hit):
            out[q] = int(hit[0]) + 1
    return out if query_mask is None else out[query_mask]


def top_rows(c: Candidates, probs: np.ndarray, k: int) -> list[list[int]]:
    """Per query, the corpus rows of the ``k`` best candidates."""
    order = np.lexsort((-probs, c.query_ids))
    out: list[list[int]] = [[] for _ in range(len(c.fallback_rank))]
    for q, r in zip(c.query_ids[order], c.rows[order]):
        if len(out[q]) < k:
            out[q].append(int(r))
    return out


def mrr10(ranks: np.ndarray) -> float:
    return float(np.mean(np.where((ranks > 0) & (ranks <= 10), 1.0 / np.maximum(ranks, 1), 0.0)))


# ---------------------------------------------------------------- models --

def make_model(kind: str, params: dict | None = None, seed: int = 13):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    params = params or {}
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    if kind == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=seed, **params))
    raise ValueError(kind)


def _row_mask(c: Candidates, queries: np.ndarray) -> np.ndarray:
    return np.isin(c.query_ids, queries)


def cv_score(kind: str, params: dict, c: Candidates, n_splits: int = 5, seed: int = 13) -> float:
    """Mean MRR@10 over GroupKFold folds (grouped by article)."""
    from sklearn.model_selection import GroupKFold

    n_q = len(c.fallback_rank)
    scores = []
    for tr_q, te_q in GroupKFold(n_splits=n_splits).split(np.zeros(n_q), groups=c.groups):
        tr, te = _row_mask(c, tr_q), _row_mask(c, te_q)
        model = make_model(kind, params, seed).fit(c.X[tr], c.y[tr])
        probs = np.zeros(len(c.y))
        probs[te] = model.predict_proba(c.X[te])[:, 1]
        qmask = np.zeros(n_q, dtype=bool)
        qmask[te_q] = True
        scores.append(mrr10(ranks_from_probs(c, probs, qmask)))
    return float(np.mean(scores))


def sample_space(space: dict, n: int, seed: int = 13) -> list[dict]:
    """``n`` random points of ``space`` (lists = uniform choice, ("loguniform",
    lo, hi) = log-uniform): the RandomizedSearchCV sampling, explicit."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        point = {}
        for name, dom in space.items():
            if isinstance(dom, tuple) and dom[0] == "loguniform":
                point[name] = float(math.exp(rng.uniform(math.log(dom[1]), math.log(dom[2]))))
            else:
                point[name] = dom[int(rng.integers(len(dom)))]
        out.append(point)
    return out


def search(c: Candidates, seed: int = 13) -> dict:
    """Default vs tuned, for both models, by cross-validated MRR@10."""
    out: dict = {}
    hgb_trials = [{"params": {}, "cv_mrr@10": cv_score("hgb", {}, c, seed=seed), "default": True}]
    for params in sample_space(HGB_SPACE, HGB_TRIALS, seed):
        hgb_trials.append({"params": params, "cv_mrr@10": cv_score("hgb", params, c, seed=seed)})
        log.info("hgb trial %d/%d: %.4f", len(hgb_trials) - 1, HGB_TRIALS,
                 hgb_trials[-1]["cv_mrr@10"])
    out["hgb"] = {"search": "randomized", "space": {k: list(v) if isinstance(v, tuple) else v
                                                   for k, v in HGB_SPACE.items()},
                  "trials": hgb_trials, "default": hgb_trials[0],
                  "best": max(hgb_trials[1:], key=lambda t: t["cv_mrr@10"])}
    lr_trials = [{"params": {"C": C}, "cv_mrr@10": cv_score("logreg", {"C": C}, c, seed=seed),
                  "default": C == 1.0} for C in LR_GRID["C"]]
    out["logreg"] = {"search": "grid", "space": LR_GRID, "trials": lr_trials,
                     "default": next(t for t in lr_trials if t["default"]),
                     "best": max(lr_trials, key=lambda t: t["cv_mrr@10"])}
    return out


def feature_importance(model, c: Candidates, seed: int = 13) -> dict[str, float]:
    """Permutation importance (drop in average precision) per feature."""
    from sklearn.inspection import permutation_importance

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(c.y), size=min(40000, len(c.y)), replace=False)
    imp = permutation_importance(model, c.X[idx], c.y[idx], scoring="average_precision",
                                 n_repeats=3, random_state=seed)
    return {f: round(float(v), 4) for f, v in zip(FEATURES, imp.importances_mean)}

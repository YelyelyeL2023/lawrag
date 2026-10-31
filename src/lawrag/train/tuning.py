"""Systematic hyper-parameter search, always on the validation split.

Two searches, both exhaustive grids (``sklearn.model_selection.ParameterGrid``)
so the search space is explicit and every point in it is reported:

* **BM25** — stem length x k1 x b, 216 configurations, each scored on every
  validation query against the *full* corpus. Cheap: the corpus is tokenised
  once per stem length and only re-weighted per (k1, b).
* **bi-encoder fine-tuning** — learning rate x number of trainable top
  layers, 9 short runs. The layer axis is also the transfer-learning
  question: how much of the pre-trained network has to change for the new
  task.

A full fine-tuning run is ~75 min and re-encoding the corpus ~10 min, so the
fine-tuning grid is a *low-fidelity* search: each trial trains on the same
fixed sample of the training pairs, and is scored on a fixed **validation
pool** — every chunk of the validation articles, the top 30 chunks per
validation query under two independent systems (stock bge-m3 and BM25) and
5,000 random chunks — instead of the whole corpus. The pool keeps the hard
distractors that decide the ranking and drops the 30k chunks that no system
puts near any validation query. The winning configuration is then retrained
on all training pairs and scored on the full corpus.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..logging_setup import get_logger
from .evaluate import EvalResult, evaluate, rank_pairs
from .lexical import BM25
from .pairs import Pair

log = get_logger(__name__)

# The midterm's BM25 — textbook k1 and b, the 6-character stem chosen then.
BM25_DEFAULT = {"prefix": 6, "k1": 1.2, "b": 0.75}
BM25_GRID = {
    "prefix": [4, 5, 6, 7, 8, 0],                 # 0 = whole words, no stemming
    "k1": [0.5, 0.8, 1.2, 1.6, 2.0, 3.0],         # term-frequency saturation
    "b": [0.3, 0.5, 0.65, 0.75, 0.85, 1.0],       # document-length normalisation
}

FT_GRID = {
    "learning_rate": [1e-5, 2e-5, 5e-5],
    "train_layers": [4, 8, 12],                   # of 24; the rest stays frozen
}
FT_TRIAL_PAIRS = 4000

# What every search optimises, and how ties are broken.
SELECT_BY = ("mrr@10", "recall@10")


def metrics(res: EvalResult) -> dict:
    o = res.overall
    return {"recall@1": round(o.r1, 4), "recall@5": round(o.r5, 4),
            "recall@10": round(o.r10, 4), "mrr@10": round(o.mrr10, 4), "n": o.n}


def best_trial(trials: list[dict]) -> dict:
    return max(trials, key=lambda t: tuple(t[k] for k in SELECT_BY))


# ------------------------------------------------------------------ BM25 --

def bm25_grid_search(val: list[Pair], corpus: list[dict],
                     grid: dict | None = None) -> dict:
    from sklearn.model_selection import ParameterGrid

    grid = grid or BM25_GRID
    texts = [c["text"] for c in corpus]
    # Grouped by stem length: tokenising the corpus is the only expensive step.
    points = sorted(ParameterGrid(grid), key=lambda p: (p["prefix"], p["k1"], p["b"]))
    trials: list[dict] = []
    started = time.perf_counter()
    bm25: BM25 | None = None
    for params in points:
        if bm25 is None or bm25.prefix != params["prefix"]:
            bm25 = BM25(prefix=params["prefix"]).fit(texts)
        bm25.reweight(params["k1"], params["b"])
        res = rank_pairs(bm25.score, val, corpus, name="bm25")
        trials.append({**params, **metrics(res)})
        if len(trials) % 36 == 0:
            log.info("bm25 grid: %d/%d", len(trials), len(points))
    default = next(t for t in trials if all(t[k] == v for k, v in BM25_DEFAULT.items()))
    return {"space": grid, "n_trials": len(trials), "selected_by": list(SELECT_BY),
            "split": "validation", "seconds": round(time.perf_counter() - started),
            "default": default, "best": best_trial(trials), "trials": trials}


def make_bm25(params: dict, corpus: list[dict]) -> BM25:
    return BM25(k1=params["k1"], b=params["b"], prefix=params["prefix"]).fit(
        [c["text"] for c in corpus])


# ------------------------------------------------------- validation pool --

def validation_pool(val: list[Pair], corpus: list[dict], tops: list[list[int]], *,
                    n_random: int = 5000, seed: int = 13) -> list[int]:
    """Corpus row indices of the fixed pool the fine-tuning trials rank."""
    val_articles = {p.article for p in val}
    pool = {i for i, c in enumerate(corpus) if c["article"] in val_articles}
    for top in tops:
        pool.update(top)
    rng = np.random.default_rng(seed)
    rest = np.array([i for i in range(len(corpus)) if i not in pool])
    if len(rest):
        pool.update(int(i) for i in rng.choice(rest, size=min(n_random, len(rest)),
                                               replace=False))
    return sorted(pool)


# ------------------------------------------------------ fine-tuning grid --

def trial_name(params: dict) -> str:
    return f"ft-lr{params['learning_rate']:g}-L{params['train_layers']}"


def ft_grid_search(base: Path, fit: list[Pair], val: list[Pair], corpus: list[dict], *,
                   pool: list[int], held_out: frozenset[str], trial_dir: Path,
                   stage_dir: Path, cache_dir: Path, device: str | None,
                   grid: dict | None = None, max_pairs: int = FT_TRIAL_PAIRS,
                   keep_weights: bool = False) -> dict:
    """Train and score one model per grid point; resumable (a finished trial
    leaves a JSON file and is not re-run)."""
    from sklearn.model_selection import ParameterGrid

    from .finetune import TrainSettings, finetune

    grid = grid or FT_GRID
    stage_dir.mkdir(parents=True, exist_ok=True)
    sub = [corpus[i] for i in pool]
    trials: list[dict] = []
    for params in ParameterGrid(grid):
        name = trial_name(params)
        done = stage_dir / f"{name}.json"
        if done.exists():
            trials.append(json.loads(done.read_text(encoding="utf-8")))
            continue
        out = trial_dir / name
        log.info("fine-tuning trial %s", name)
        settings = TrainSettings(base=str(base), out=str(out), max_pairs=max_pairs,
                                 device=device, **params)
        finetune(settings, fit, corpus, cache_dir=cache_dir, exclude_articles=held_out)
        res = evaluate(out, val, sub, device=device, cache_dir=cache_dir)
        meta = json.loads((out / "lawrag_train.json").read_text(encoding="utf-8"))
        trial = {**params, **metrics(res), "train_seconds": meta["train_seconds"],
                 "pairs": meta["pairs"],
                 "loss_curve": [[h["epoch"], h["loss"]] for h in meta["log_history"]
                                if "loss" in h]}
        done.write_text(json.dumps(trial, indent=1), encoding="utf-8")
        trials.append(trial)
        if not keep_weights:
            shutil.rmtree(out, ignore_errors=True)
    return {"space": grid, "n_trials": len(trials), "selected_by": list(SELECT_BY),
            "split": "validation pool", "pool_chunks": len(pool),
            "trial_pairs": max_pairs, "best": best_trial(trials), "trials": trials}

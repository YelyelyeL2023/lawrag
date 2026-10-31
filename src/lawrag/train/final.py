"""The final experiment: model comparison, tuning, fine-tuning, transfer
learning and QLoRA, all on one split, in one resumable command.

``lawrag final-experiment`` runs these stages; each writes its result under
``<out>/stages/`` and is skipped on the next run when that file exists:

1. **split** — train / validation / test *by article*. Test is the midterm's
   held-out split, unchanged; validation is carved out of the midterm's train
   articles. Every model below is trained on train only, selected on
   validation only, and scored on test once.
2. **BM25 grid search** (classical) — stem length x k1 x b on validation.
3. **fine-tuning grid search** — learning rate x trainable layers for the
   bge-m3 bi-encoder, low-fidelity trials on a fixed validation pool.
4. **final fine-tune** — the best configuration on all train pairs.
5. **dense and lexical systems on test** — BM25 default / tuned, bge-m3
   zero-shot, bge-m3 fine-tuned with the midterm configuration, and with the
   tuned one.
6. **learned hybrid** (classical ML) — gradient boosting and logistic
   regression over BM25 + dense features, default vs tuned, fitted and
   tuned on validation.
7. **LLM reranker** — Qwen3-0.6B reranks the best first stage's top 20:
   zero-shot, then after LoRA; bge-reranker-v2-m3 as a reference.
8. **report** — one results table, paired bootstrap tests, grids, figures,
   the environment (library versions, GPU, seeds, git commit).
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import Config
from ..logging_setup import get_logger
from .evaluate import EvalResult, encode, encode_corpus, evaluate, load_model
from .experiment import _hit, _rr10, bootstrap_ci, error_analysis, number_slice
from .pairs import Pair, carve_validation, read_jsonl, read_pairs

log = get_logger(__name__)

SEED = 13
RERANK_DEPTH = 20


# ---------------------------------------------------------------- systems --
# Every system that appears in the results table: display name, family (the
# figure colour), what it learned from, and how it was tuned.
SYSTEMS = {
    "bm25-default": ("BM25, midterm parameters", "lexical", "corpus statistics", "—"),
    "bm25-tuned": ("BM25, grid-searched", "lexical", "corpus statistics",
                   "grid, 216 points (validation)"),
    "dense-zeroshot": ("bge-m3 zero-shot", "dense", "nothing (transfer only)", "—"),
    "dense-midterm": ("bge-m3 fine-tuned, midterm config", "dense", "train pairs",
                      "— (lr 2e-5, top 8 layers)"),
    "dense-final": ("bge-m3 fine-tuned, tuned config", "dense", "train pairs",
                    "grid, 9 points (validation)"),
    "hybrid-logreg-default": ("hybrid: logistic regression, default", "hybrid",
                              "validation candidates", "—"),
    "hybrid-logreg-tuned": ("hybrid: logistic regression, tuned", "hybrid",
                            "validation candidates", "grid over C, 5-fold group CV"),
    "hybrid-hgb-default": ("hybrid: gradient boosting, default", "hybrid",
                           "validation candidates", "—"),
    "hybrid-hgb-tuned": ("hybrid: gradient boosting, tuned", "hybrid",
                         "validation candidates", "random search, 30 points, 5-fold group CV"),
    "rerank-bge": ("+ bge-reranker-v2-m3 zero-shot (reference)", "reranker",
                   "nothing (transfer only)", "—"),
    "rerank-qwen-zeroshot": ("+ Qwen3-0.6B reranker zero-shot", "reranker",
                             "nothing (transfer only)", "—"),
    "rerank-qwen-lora": ("+ Qwen3-0.6B reranker, LoRA", "reranker", "train pairs (groups)",
                         "—"),
}


def canonical(pairs: list[Pair]) -> list[Pair]:
    """Queries in the order ``rank_pairs`` reports them (by kind, then as
    given), so per-query results of every system line up for paired tests."""
    return [p for kind in sorted({p.kind for p in pairs}) for p in pairs if p.kind == kind]


class Stages:
    """JSON results of finished stages under ``<out>/stages``."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def get(self, name: str):
        p = self.path(name)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, name: str, value) -> None:
        # Write-then-rename: a crash mid-write (this machine's GPU driver has
        # taken the whole system down once) leaves the old file or none, never
        # a truncated one the resume would choke on.
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    def cached(self, name: str, compute: Callable[[], object]):
        value = self.get(name)
        if value is None:
            log.info("stage %s", name)
            value = compute()
            self.put(name, value)
        else:
            log.info("stage %s: cached", name)
        return value

    def system(self, name: str, compute: Callable[[], EvalResult]) -> EvalResult:
        stored = self.cached(f"systems/{name}",
                             lambda: {"name": name, "queries": compute().queries})
        return EvalResult.from_queries(name, stored["queries"])


# --------------------------------------------------------------- stay awake --

class stay_awake:
    """Ask Windows not to idle-sleep while the experiment runs (the run is
    hours long and usually left unattended). A per-process request: no power
    setting changes, and it ends with the process. A no-op elsewhere."""

    _CONTINUOUS, _SYSTEM_REQUIRED = 0x80000000, 0x00000001

    def __enter__(self):
        if os.name == "nt":
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(self._CONTINUOUS | self._SYSTEM_REQUIRED)
        return self

    def __exit__(self, *exc):
        if os.name == "nt":
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(self._CONTINUOUS)
        return False


# ------------------------------------------------------------ environment --

LIBRARIES = ("torch", "transformers", "sentence-transformers", "peft", "bitsandbytes",
             "accelerate", "datasets", "scikit-learn", "numpy", "scipy", "matplotlib")


def environment() -> dict:
    versions = {}
    for lib in LIBRARIES:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = None
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = {"name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                   "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}
    except ImportError:
        pass

    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:  # noqa: BLE001
            return ""

    return {"python": platform.python_version(), "platform": platform.platform(),
            "gpu": gpu, "libraries": versions, "git_commit": git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")), "seed": SEED}


# --------------------------------------------------------------- scorers --

class DenseScorer:
    """Cosine scores of queries against the whole corpus for one embedder."""

    def __init__(self, model_path: Path, corpus: list[dict], cache_dir: Path,
                 device: str | None):
        self.model = load_model(model_path, device)
        self.doc = encode_corpus(self.model, model_path, [c["text"] for c in corpus], cache_dir)

    def __call__(self, queries: list[str]) -> np.ndarray:
        return encode(self.model, queries, is_query=True) @ self.doc.T

    def close(self) -> None:
        del self.model
        _free_gpu()


def _free_gpu() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _from_ranks(stage1: EvalResult, ranks, top1, name: str) -> EvalResult:
    queries = [{k: v for k, v in q.items() if k not in ("top", "top_scores")}
               | {"rank": int(r), "top1": t}
               for q, r, t in zip(stage1.queries, ranks, top1)]
    return EvalResult.from_queries(name, queries)


# ------------------------------------------------------------- statistics --

def overall(res: EvalResult) -> dict:
    r = np.array([q["rank"] for q in res.queries])
    return {m: [float(f(r).mean()), *bootstrap_ci(f(r))]
            for m, f in (("recall@1", lambda x: _hit(x, 1)), ("recall@5", lambda x: _hit(x, 5)),
                         ("recall@10", lambda x: _hit(x, 10)), ("mrr@10", _rr10))}


def paired(a: EvalResult, b: EvalResult) -> dict:
    """``a`` minus ``b`` on the same queries, with bootstrap intervals."""
    ra = np.array([q["rank"] for q in a.queries])
    rb = np.array([q["rank"] for q in b.queries])
    assert [q["query"] for q in a.queries] == [q["query"] for q in b.queries], "unaligned"
    d_mrr, d_r1 = _rr10(ra) - _rr10(rb), _hit(ra, 1) - _hit(rb, 1)
    d_r10 = _hit(ra, 10) - _hit(rb, 10)
    return {"mrr@10": [float(d_mrr.mean()), *bootstrap_ci(d_mrr)],
            "recall@1": [float(d_r1.mean()), *bootstrap_ci(d_r1)],
            "recall@10": [float(d_r10.mean()), *bootstrap_ci(d_r10)],
            "better": int((_rr10(ra) > _rr10(rb)).sum()),
            "worse": int((_rr10(ra) < _rr10(rb)).sum())}


# What each improvement step is compared with: (system, its predecessor).
STEPS = [
    ("bm25-tuned", "bm25-default", "BM25 parameter tuning"),
    ("dense-zeroshot", "bm25-tuned", "transfer: pre-trained multilingual encoder vs tuned BM25"),
    ("dense-midterm", "dense-zeroshot", "fine-tuning (midterm configuration)"),
    ("dense-final", "dense-midterm", "fine-tuning hyper-parameter search"),
    ("hybrid-hgb-tuned", "hybrid-hgb-default", "gradient-boosting hyper-parameter search"),
    ("hybrid-logreg-tuned", "hybrid-logreg-default", "logistic-regression C search"),
    ("@stage1", "dense-final", "learned hybrid vs dense alone"),
    ("rerank-qwen-zeroshot", "@stage1", "zero-shot LLM reranker vs its input"),
    ("rerank-qwen-lora", "rerank-qwen-zeroshot", "LoRA adaptation of the LLM reranker"),
    ("rerank-qwen-lora", "@stage1", "LoRA LLM reranker vs its input"),
    ("rerank-qwen-lora", "rerank-bge", "LoRA Qwen3-0.6B vs bge-reranker-v2-m3"),
    ("@best", "dense-zeroshot", "whole pipeline vs zero-shot bge-m3"),
]


# ----------------------------------------------------------------- driver --

def run_final(cfg: Config, *, pairs_dir: Path, model_dir: Path, out_dir: Path,
              base_name: str = "bge-m3", midterm_name: str = "bge-m3-lawrag-midterm",
              final_name: str = "bge-m3-lawrag-final", llm_name: str = "Qwen3-0.6B",
              adapter_name: str = "qwen3-0.6b-lawrag-lora", qlora: bool = False,
              cross_name: str = "bge-reranker-v2-m3", device: str | None = None,
              train: bool = True, ft_grid: bool = True, rerank: bool = True,
              rerank_queries: int = 4000, ft_grid_spec: dict | None = None,
              bm25_grid_spec: dict | None = None, keep_trial_weights: bool = False,
              ft_trial_pairs: int | None = None, final_max_pairs: int | None = None) -> dict:
    from . import tuning
    from .finetune import TrainSettings, finetune

    random.seed(SEED)
    np.random.seed(SEED)
    stages = Stages(out_dir / "stages")
    cache = pairs_dir / "cache"

    # 1. split
    train_pairs = read_pairs(pairs_dir / "pairs.train.jsonl")
    test = canonical(read_pairs(pairs_dir / "pairs.dev.jsonl"))
    corpus = read_jsonl(pairs_dir / "corpus.jsonl")
    fit, val = carve_validation(train_pairs)
    val = canonical(val)
    held_out = frozenset(p.article for p in val) | frozenset(p.article for p in test)
    data = split_summary(fit, val, test, corpus)
    if data["shared_articles"]:
        raise RuntimeError(f"splits share articles: {data['shared_articles']}")
    if not val or not test:
        raise RuntimeError("validation or test split is empty")

    systems: dict[str, EvalResult] = {}
    base, midterm, final = model_dir / base_name, model_dir / midterm_name, model_dir / final_name
    has = lambda p: (p / "config.json").exists()  # noqa: E731
    # A model this experiment trains counts as finished only once its
    # metadata — written after the weights — is there; a crash while saving
    # leaves config.json without it, and the stage is redone.
    trained = lambda p: (p / "lawrag_train.json").exists()  # noqa: E731

    # 2. BM25 grid search
    bm25_grid = stages.cached("bm25_grid", lambda: tuning.bm25_grid_search(
        val, corpus, bm25_grid_spec))
    bm25 = {"default": tuning.make_bm25(tuning.BM25_DEFAULT, corpus),
            "tuned": tuning.make_bm25(bm25_grid["best"], corpus)}
    for which, model in bm25.items():
        systems[f"bm25-{which}"] = stages.system(
            f"bm25-{which}", lambda m=model: evaluate("bm25", test, corpus, bm25=m))

    # 3. fine-tuning grid search on the validation pool
    ft = None
    if has(base):
        def pool() -> dict:
            tops = [q["top"] for q in evaluate(base, val, corpus, device=device, cache_dir=cache,
                                               keep_top=30).queries]
            tops += [q["top"] for q in evaluate("bm25", val, corpus, bm25=bm25["default"],
                                                keep_top=30).queries]
            return {"rows": tuning.validation_pool(val, corpus, tops)}
        pool_rows = stages.cached("val_pool", pool)["rows"]
        if ft_grid and train:
            def grid() -> dict:
                res = tuning.ft_grid_search(
                    base, fit, val, corpus, pool=pool_rows, held_out=held_out,
                    trial_dir=model_dir / ".ft-grid", stage_dir=stages.root / "ft_grid",
                    cache_dir=cache, device=device, grid=ft_grid_spec,
                    max_pairs=ft_trial_pairs or tuning.FT_TRIAL_PAIRS,
                    keep_weights=keep_trial_weights)
                sub = [corpus[i] for i in pool_rows]
                res["reference"] = {
                    name: tuning.metrics(evaluate(path, val, sub, device=device, cache_dir=cache))
                    for name, path in (("zero-shot", base), ("midterm", midterm)) if has(path)}
                return res
            ft = stages.cached("ft_grid", grid)
        else:
            ft = stages.get("ft_grid")

    # 4. final fine-tune with the selected configuration
    if train and has(base) and not trained(final) and ft:
        best = ft["best"]
        log.info("final fine-tune: lr %g, top %d layers", best["learning_rate"],
                 best["train_layers"])
        finetune(TrainSettings(base=str(base), out=str(final), device=device,
                               learning_rate=best["learning_rate"],
                               train_layers=best["train_layers"], max_pairs=final_max_pairs),
                 fit, corpus, cache_dir=cache, exclude_articles=held_out)
        _free_gpu()

    # 5. dense systems on test
    for name, path in (("dense-zeroshot", base), ("dense-midterm", midterm),
                       ("dense-final", final)):
        if has(path) and (path != final or trained(final)):
            systems[name] = stages.system(name, lambda p=path: evaluate(
                p, test, corpus, device=device, cache_dir=cache,
                keep_top=RERANK_DEPTH if p == final else 0))
    dense_val = None
    if trained(final):
        dense_val = stages.cached("dense-final-val", lambda: tuning.metrics(
            evaluate(final, val, corpus, device=device, cache_dir=cache)))

    # 6. learned hybrid (classical ML) over BM25-tuned + dense-final
    hybrid = None
    if trained(final):
        hybrid = stages.get("hybrid")
        if hybrid is None:
            hybrid = run_hybrid(final, val, test, corpus, bm25["tuned"], cache, device)
            stages.put("hybrid", hybrid)
        for key, ranks in hybrid["test_ranks"].items():
            systems[f"hybrid-{key}"] = _from_ranks(systems["dense-final"], ranks,
                                                   hybrid["test_top1"][key], f"hybrid-{key}")

    # 7. LLM reranker over the best first stage's top 20
    stage1, stage1_tops = None, None
    if "dense-final" in systems:
        stage1 = "dense-final"
        stage1_tops = [q["top"] for q in stages.get("systems/dense-final")["queries"]]
        if hybrid and hybrid["best"]["cv_mrr@10"] > dense_val["mrr@10"]:
            stage1 = f"hybrid-{hybrid['best']['key']}"
            stage1_tops = hybrid["best"]["test_top"]
    rerank_meta = None
    llm = model_dir / llm_name
    if rerank and stage1 and has(llm):
        rerank_meta = run_rerankers(
            stages, systems, stage1, stage1_tops, corpus, fit, held_out, final=final,
            llm=llm, adapter=model_dir / adapter_name, cross=model_dir / cross_name,
            cache=cache, device=device, n_queries=rerank_queries, train=train, qlora=qlora)

    # 8. report
    from .final_report import write_report

    train_meta = None
    if has(final) and (final / "lawrag_train.json").exists():
        train_meta = json.loads((final / "lawrag_train.json").read_text(encoding="utf-8"))
    best_name = max(systems, key=lambda n: overall(systems[n])["mrr@10"][0])
    resolve = {"@stage1": stage1, "@best": best_name}
    steps = []
    for a, b, label in STEPS:
        a, b = resolve.get(a, a), resolve.get(b, b)
        if a in systems and b in systems and a != b:
            steps.append({"system": a, "baseline": b, "label": label,
                          **paired(systems[a], systems[b])})
    summary = {
        "data": data,
        "environment": environment(),
        "systems": {n: {"label": SYSTEMS[n][0], "family": SYSTEMS[n][1],
                        "learned_from": SYSTEMS[n][2], "tuning": SYSTEMS[n][3],
                        **r.to_dict(), "overall": overall(r)} for n, r in systems.items()},
        "steps": steps,
        "stage1": stage1,
        "best_system": best_name,
        "bm25_grid": bm25_grid,
        "ft_grid": ft,
        "dense_final_val": dense_val,
        "hybrid": {k: v for k, v in (hybrid or {}).items()
                   if k not in ("test_ranks", "test_top1")} or None,
        "finetune": _portable(train_meta) if train_meta else None,
        "reranker": rerank_meta,
        "errors": {n: error_analysis(systems[n], corpus) for n in ("dense-final", best_name)
                   if n in systems},
        "number_shortcut": number_slice({n: systems[n] for n in
                                         ("dense-zeroshot", "dense-midterm", "dense-final",
                                          stage1, best_name) if n in systems}),
    }
    if summary["hybrid"] and "best" in summary["hybrid"]:
        summary["hybrid"]["best"] = {k: v for k, v in summary["hybrid"]["best"].items()
                                     if k != "test_top"}
    write_report(summary, systems, out_dir, train_meta)
    return summary


def split_summary(fit: list[Pair], val: list[Pair], test: list[Pair], corpus: list[dict]) -> dict:
    def one(ps: list[Pair]) -> dict:
        kinds: dict[str, int] = {}
        for p in ps:
            kinds[p.kind] = kinds.get(p.kind, 0) + 1
        return {"pairs": len(ps), "articles": len({p.article for p in ps}), "kinds": kinds}

    arts = [{p.article for p in s} for s in (fit, val, test)]
    return {"train": one(fit), "validation": one(val), "test": one(test),
            "chunks": len(corpus),
            "shared_articles": len(arts[0] & arts[1]) + len(arts[0] & arts[2])
            + len(arts[1] & arts[2])}


def _portable(meta: dict) -> dict:
    out = {k: v for k, v in meta.items() if k != "log_history"}
    out["base"] = Path(str(out.get("base", ""))).name
    if "settings" in out:
        out["settings"] = {**out["settings"],
                           "base": Path(str(out["settings"].get("base", ""))).name,
                           "out": Path(str(out["settings"].get("out", ""))).name}
    out["loss_curve"] = [[h["epoch"], h["loss"]] for h in meta.get("log_history", [])
                         if "loss" in h]
    return out


# ----------------------------------------------------------------- hybrid --

def run_hybrid(final: Path, val: list[Pair], test: list[Pair], corpus: list[dict], bm25,
               cache: Path, device: str | None) -> dict:
    from . import ltr

    dense = DenseScorer(final, corpus, cache, device)
    started = time.perf_counter()
    c_val = ltr.build_candidates(val, corpus, dense, bm25.score)
    c_test = ltr.build_candidates(test, corpus, dense, bm25.score)
    dense.close()
    log.info("hybrid candidates: %d val rows, %d test rows (%.0fs)", len(c_val.y),
             len(c_test.y), time.perf_counter() - started)
    search = ltr.search(c_val, seed=SEED)
    out: dict = {"features": list(ltr.FEATURES), "candidates_per_list": ltr.CANDIDATES_PER_LIST,
                 "val_rows": int(len(c_val.y)), "val_positive_rate": float(c_val.y.mean()),
                 "val_recall_of_candidates": float(np.mean(c_val.fallback_rank == 0)),
                 "test_recall_of_candidates": float(np.mean(c_test.fallback_rank == 0)),
                 "search": search, "test_ranks": {}, "test_top1": {}, "importance": {}}
    best = None
    for kind in ("logreg", "hgb"):
        for which in ("default", "best"):
            trial = search[kind][which]
            model = ltr.make_model(kind, trial["params"], SEED).fit(c_val.X, c_val.y)
            probs = model.predict_proba(c_test.X)[:, 1]
            key = f"{kind}-{'tuned' if which == 'best' else 'default'}"
            out["test_ranks"][key] = ltr.ranks_from_probs(c_test, probs).tolist()
            out["test_top1"][key] = [corpus[rows[0]]["article"]
                                     for rows in ltr.top_rows(c_test, probs, 1)]
            if which == "best":
                out["importance"][kind] = ltr.feature_importance(model, c_val, SEED)
                if kind == "logreg":
                    coef = model[-1].coef_[0]
                    out["logreg_coefficients"] = {f: round(float(v), 4)
                                                  for f, v in zip(ltr.FEATURES, coef)}
                if best is None or trial["cv_mrr@10"] > best["cv_mrr@10"]:
                    best = {"key": key, "cv_mrr@10": trial["cv_mrr@10"],
                            "test_top": ltr.top_rows(c_test, probs, RERANK_DEPTH)}
    out["best"] = best
    return out


# --------------------------------------------------------------- rerankers --

def run_rerankers(stages: Stages, systems: dict[str, EvalResult], stage1: str,
                  stage1_tops: list[list[int]], corpus: list[dict], fit: list[Pair],
                  held_out: frozenset[str], *, final: Path, llm: Path, adapter: Path,
                  cross: Path, cache: Path, device: str | None, n_queries: int,
                  train: bool, qlora: bool = False) -> dict:
    from .llm_rerank import (RerankSettings, load_reranker, make_groups, rerank,
                             train_reranker)

    base_res = systems[stage1]
    queries = base_res.queries
    cands = [list(t[:RERANK_DEPTH]) for t in stage1_tops]

    def run_llm(adapter_path: Path | None) -> EvalResult:
        model, tok = load_reranker(str(llm), str(adapter_path) if adapter_path else None,
                                   device=device, load_in_4bit=qlora)
        ranks, top1, _ = rerank(queries, cands, corpus, model, tok)
        del model
        _free_gpu()
        return _from_ranks(base_res, ranks, top1, "llm")

    def run_cross() -> EvalResult:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(str(cross), device=device, max_length=512)
        if device is None or str(device).startswith("cuda"):
            model.model.half()
        flat = [(q["query"], corpus[j]["text"]) for q, c in zip(queries, cands) for j in c]
        scores = np.asarray(model.predict(flat, batch_size=32, show_progress_bar=False))
        ranks, top1, at = [], [], 0
        for q, c in zip(queries, cands):
            s = scores[at:at + len(c)]
            at += len(c)
            order = [c[i] for i in np.argsort(-s, kind="stable")]
            hit = next((r for r, j in enumerate(order, 1) if corpus[j]["article"] == q["article"]
                        and corpus[j]["lang"] == q["pos_lang"]), None)
            ranks.append(hit if hit is not None else q["rank"])
            top1.append(corpus[order[0]]["article"])
        del model
        _free_gpu()
        return _from_ranks(base_res, ranks, top1, "bge-reranker")

    systems["rerank-qwen-zeroshot"] = stages.system("rerank-qwen-zeroshot", lambda: run_llm(None))
    if (cross / "config.json").exists():
        systems["rerank-bge"] = stages.system("rerank-bge", run_cross)

    settings = RerankSettings(base=str(llm), out=str(adapter), device=device,
                              n_queries=n_queries, load_in_4bit=qlora)
    if train and not (adapter / "lawrag_rerank.json").exists():
        rng = random.Random(settings.seed)
        pool = [p for p in fit if p.article not in held_out]
        sample = canonical(rng.sample(pool, min(n_queries, len(pool))))
        # The fine-tuned bi-encoder's own confusions on training queries are
        # the negatives: what the reranker will be asked to sort out.
        tops = [q["top"] for q in evaluate(final, sample, corpus, device=device, cache_dir=cache,
                                           keep_top=settings.negative_pool).queries]
        _free_gpu()
        groups = make_groups(sample, tops, corpus, settings, held_out)
        log.info("reranker training groups: %d", len(groups))
        train_reranker(settings, groups)
    meta = None
    if (adapter / "lawrag_rerank.json").exists():
        systems["rerank-qwen-lora"] = stages.system("rerank-qwen-lora", lambda: run_llm(adapter))
        meta = json.loads((adapter / "lawrag_rerank.json").read_text(encoding="utf-8"))
        meta["settings"] = {**meta["settings"], "base": llm.name, "out": adapter.name}
    return {"stage1": stage1, "depth": RERANK_DEPTH, "training": meta}

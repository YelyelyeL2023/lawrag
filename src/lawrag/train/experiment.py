"""The baseline experiment, end to end, in one reproducible command.

``lawrag experiment`` runs every stage that produces the reported numbers, and
skips a stage whose output already exists:

1. **data** — mine (query, article) pairs from the corpus and split them into
   train and dev *by article* (``pairs.py``); dev articles are never seen in
   training, not even as negatives;
2. **train** — fine-tune bge-m3 on the train pairs (``finetune.py``);
3. **evaluate** — rank the whole corpus for every dev query with three
   systems: BM25 (lexical, nothing learned), stock bge-m3 (zero-shot dense)
   and the fine-tuned bge-m3 (``evaluate.py``);
4. **report** — tables, paired-bootstrap confidence intervals, an error
   analysis and figures, written to ``reports/midterm/`` (or ``--out``).

Every path is relative to the working directory or comes from an option /
``LAWRAG_MODEL_DIR``; nothing here knows which machine it runs on.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

from ..config import Config
from ..logging_setup import get_logger
from .evaluate import EvalResult, evaluate
from .lexical import BM25_MODEL
from .pairs import Pair, read_jsonl, read_pairs

log = get_logger(__name__)

# Display names and fixed colours per system: a system keeps its colour in
# every figure (categorical slots 1-3 of the validated reference palette).
LABELS = {"bm25": "BM25 (lexical)", "base": "bge-m3 (zero-shot)", "tuned": "bge-m3 fine-tuned"}
COLORS = {"tuned": "#2a78d6", "base": "#eb6834", "bm25": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

KIND_TITLES = {
    "heading": "heading — article title as query",
    "xref": "xref — citing sentence, citation removed",
    "xlang": "xlang — same norm, other language",
}


def default_model_dir() -> Path:
    """Where the weights live: ``LAWRAG_MODEL_DIR``, else ``MODEL_DIR`` (the
    name scripts/download_models.py and docker/.env use), else ``models/``."""
    return Path(os.environ.get("LAWRAG_MODEL_DIR") or os.environ.get("MODEL_DIR") or "models")


# ----------------------------------------------------------------- stages --

def ensure_pairs(cfg: Config, corpus_dir: Path, pairs_dir: Path,
                 dev_fraction: float = 0.05) -> None:
    """Stage 1: mine and split the pairs unless they are already there."""
    from .pairs import mine_directory, split_pairs, write_jsonl

    if (pairs_dir / "pairs.train.jsonl").exists() and (pairs_dir / "corpus.jsonl").exists():
        log.info("pairs: using %s", pairs_dir)
        return
    log.info("pairs: mining %s", corpus_dir)
    pairs, corpus = mine_directory(cfg, corpus_dir, query_log=False, cache_dir=pairs_dir)
    train, dev = split_pairs(pairs, dev_fraction=dev_fraction)
    write_jsonl(pairs_dir / "pairs.train.jsonl", (p.to_dict() for p in train))
    write_jsonl(pairs_dir / "pairs.dev.jsonl", (p.to_dict() for p in dev))
    write_jsonl(pairs_dir / "corpus.jsonl", corpus.to_records())


def ensure_model(base: Path, tuned: Path, pairs_dir: Path, train: list[Pair],
                 dev: list[Pair], corpus: list[dict], device: str | None) -> bool:
    """Stage 2: fine-tune unless the tuned model already exists. Returns
    whether a tuned model is available afterwards."""
    from .finetune import TrainSettings, finetune

    if (tuned / "config.json").exists():
        log.info("model: using %s", tuned)
        return True
    if not (base / "config.json").exists():
        raise FileNotFoundError(
            f"base model not found at {base}; download it (scripts/download_models.py) "
            "or point --model-dir / LAWRAG_MODEL_DIR at the weights")
    settings = TrainSettings(base=str(base), out=str(tuned), device=device)
    finetune(settings, train, corpus, cache_dir=pairs_dir / "cache",
             exclude_articles=frozenset(p.article for p in dev))
    return True


# ------------------------------------------------------------- statistics --

def _rr10(ranks: np.ndarray) -> np.ndarray:
    return np.where((ranks > 0) & (ranks <= 10), 1.0 / np.maximum(ranks, 1), 0.0)


def _mean(values: np.ndarray) -> float:
    """Mean that is 0.0, not NaN, for an empty slice."""
    return float(values.mean()) if len(values) else 0.0


def _hit(ranks: np.ndarray, k: int) -> np.ndarray:
    return ((ranks > 0) & (ranks <= k)).astype(float)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 13) -> tuple[float, float]:
    """95% percentile interval of the mean, resampling queries."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ranks_of(res: EvalResult) -> np.ndarray:
    return np.array([q["rank"] for q in res.queries])


def metric_table(results: dict[str, EvalResult]) -> dict:
    """Overall metrics with CIs per system, and paired differences against
    the fine-tuned model (same queries, so the pairing removes query
    difficulty from the comparison)."""
    out: dict = {"systems": {}, "paired": {}}
    for name, res in results.items():
        r = ranks_of(res)
        out["systems"][name] = {
            "recall@1": [float(_hit(r, 1).mean()), *bootstrap_ci(_hit(r, 1))],
            "recall@10": [float(_hit(r, 10).mean()), *bootstrap_ci(_hit(r, 10))],
            "mrr@10": [float(_rr10(r).mean()), *bootstrap_ci(_rr10(r))],
        }
    if "tuned" in results:
        t = ranks_of(results["tuned"])
        for other in (n for n in results if n != "tuned"):
            o = ranks_of(results[other])
            d_mrr = _rr10(t) - _rr10(o)
            d_r10 = _hit(t, 10) - _hit(o, 10)
            out["paired"][f"tuned-{other}"] = {
                "mrr@10": [float(d_mrr.mean()), *bootstrap_ci(d_mrr)],
                "recall@10": [float(d_r10.mean()), *bootstrap_ci(d_r10)],
                "better": int((_rr10(t) > _rr10(o)).sum()),
                "worse": int((_rr10(t) < _rr10(o)).sum()),
            }
    return out


# --------------------------------------------------------- error analysis --

_NUMBER = re.compile(r"\d{2,}")


def number_slice(results: dict[str, EvalResult]) -> dict:
    """xref MRR@10 per system, split by whether the query still names some
    *other* article number (the miner cuts out only the target citation).
    A model that matches numbers instead of meaning does worse on the first
    slice and is lured to the article the number names."""
    out: dict[str, dict] = {}
    for name, res in results.items():
        xref = [q for q in res.queries if q["kind"] == "xref"]
        with_num = np.array([q["rank"] for q in xref if _NUMBER.search(q["query"])])
        without = np.array([q["rank"] for q in xref if not _NUMBER.search(q["query"])])
        misses = [q for q in xref if q["rank"] != 1]
        lured = [q for q in misses if (n := _num(q["top1"])) is not None
                 and re.search(rf"(?<!\d){n}(?!\d)", q["query"])]
        out[name] = {"with_number": {"n": int(len(with_num)), "mrr@10": _mean(_rr10(with_num))},
                     "without_number": {"n": int(len(without)), "mrr@10": _mean(_rr10(without))},
                     "lured_share_of_misses": len(lured) / max(1, len(misses))}
    return out

_ART_NUM = re.compile(r"/st(\d+)")
# The corpus holds the current Tax Code (nk) and the one it replaced
# (nk2017); many of their articles state the same norm.
SUPERSEDED = {frozenset({"nk", "nk2017"})}

# Two weak-supervision artefacts the error analysis surfaced in xref pairs:
# an amendment footnote mined as if it were a sentence about the article,
# and a Kazakh citation cut out without its case suffix ("Кодекстің ына").
AMENDMENT_NOTE = re.compile(r"(Закон\w* РК от \d|ҚР \d{1,2}\.\d{2}\.\d{4}|вводится в действие с|"
                            r"қолданысқа енгізіледі|утрачивает силу|күшін жояды)")
ORPHAN_SUFFIX = re.compile(r"(Кодекстің|Заңның|баптың)\s+(ына|ыне|ина|іне|сына|сіне|на|не|ында|інде|ымен)\b")


def query_flag(q: dict) -> str:
    if AMENDMENT_NOTE.search(q["query"]):
        return "amendment footnote"
    if q["lang"] == "kk" and ORPHAN_SUFFIX.search(q["query"]):
        return "kk orphan suffix"
    return "clean"


def _act(article: str) -> str:
    return article.split("/", 1)[0]


def _num(article: str) -> int | None:
    m = _ART_NUM.search(article)
    return int(m.group(1)) if m else None


def classify_miss(q: dict) -> str:
    """Where the top-1 hit landed when it was not the right article."""
    want, got = q["article"], q["top1"]
    if got == want:
        return "right article, other language"
    if frozenset({_act(got), _act(want)}) in SUPERSEDED:
        return "superseded Tax Code (nk ↔ nk2017)"
    if _act(got) != _act(want):
        return "different code"
    a, b = _num(want), _num(got)
    if a is not None and b is not None and abs(a - b) <= 5:
        return "same code, article within ±5"
    return "same code, distant article"


def headings(corpus: list[dict]) -> dict[tuple[str, str], str]:
    """(article, lang) -> its heading line, from the chunk text the index embeds."""
    out: dict[tuple[str, str], str] = {}
    for c in corpus:
        key = (c["article"], c["lang"])
        if key not in out:
            lines = c["text"].split("\n")
            out[key] = lines[1] if len(lines) > 1 else lines[0]
    return out


def error_analysis(res: EvalResult, corpus: list[dict], n_examples: int = 8,
                   seed: int = 13) -> dict:
    misses = [q for q in res.queries if q["rank"] != 1]
    by_cat: dict[str, int] = {}
    for q in misses:
        by_cat[classify_miss(q)] = by_cat.get(classify_miss(q), 0) + 1
    by_kind: dict[str, dict[str, int]] = {}
    for q in misses:
        by_kind.setdefault(q["kind"], {}).setdefault(classify_miss(q), 0)
        by_kind[q["kind"]][classify_miss(q)] += 1
    heads = headings(corpus)
    rng = np.random.default_rng(seed)
    hard = [q for q in misses if q["kind"] == "xref" and q["rank"] > 10]
    pick = [hard[i] for i in rng.choice(len(hard), size=min(n_examples, len(hard)),
                                        replace=False)] if hard else []
    examples = [{
        "query": q["query"], "lang": q["lang"], "rank": q["rank"],
        "expected": f"{q['article']} — {heads.get((q['article'], q['lang']), '')}",
        "retrieved": f"{q['top1']} — {heads.get((q['top1'], q['lang']), '')}",
        "category": classify_miss(q),
    } for q in pick]
    ranks = ranks_of(res)
    # How the mining artefacts affect the xref scores.
    quality: dict[str, dict] = {}
    xref = [q for q in res.queries if q["kind"] == "xref"]
    for flag in ("clean", "kk orphan suffix", "amendment footnote"):
        r = np.array([q["rank"] for q in xref if query_flag(q) == flag])
        if len(r):
            quality[flag] = {"n": int(len(r)), "recall@10": float(_hit(r, 10).mean()),
                             "mrr@10": float(_rr10(r).mean())}
    return {"misses_at_1": len(misses), "queries": len(res.queries),
            "xref_quality": quality,
            "categories": dict(sorted(by_cat.items(), key=lambda kv: -kv[1])),
            "categories_by_kind": by_kind,
            "median_rank_of_misses": float(np.median(ranks[ranks != 1])) if misses else 0.0,
            "examples": examples}


# ---------------------------------------------------------------- figures --

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _spread(values: list[float], gap: float = 0.07) -> list[float]:
    """Label positions for sorted ``values``, at least ``gap`` apart, kept
    as close to the values as the gap allows."""
    out: list[float] = []
    for v in values:
        out.append(max(v, out[-1] + gap) if out else v)
    shift = max(0.0, out[-1] - 1.0) if out else 0.0   # keep the top label on the plot
    return [y - shift for y in out]


def plot_recall_at_k(results: dict[str, EvalResult], path: Path, max_k: int = 50) -> None:
    """Recall@k for k = 1..50, one panel per query kind plus all queries."""
    import matplotlib.pyplot as plt

    kinds = ["all"] + sorted({q["kind"] for q in next(iter(results.values())).queries})
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.3 * len(kinds), 3.4), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    ks = np.arange(1, max_k + 1)
    for ax, kind in zip(axes, kinds):
        _style(ax)
        at10: list[tuple[float, str]] = []
        for name in ("bm25", "base", "tuned"):
            if name not in results:
                continue
            r = np.array([q["rank"] for q in results[name].queries
                          if kind == "all" or q["kind"] == kind])
            curve = [_hit(r, k).mean() for k in ks]
            ax.plot(ks, curve, color=COLORS[name], linewidth=2, label=LABELS[name])
            at10.append((curve[9], name))
        # Recall@10 values, spread apart so close values do not overprint,
        # each tied to its point by a thin leader; text stays in ink.
        for (y, name), y_label in zip(sorted(at10), _spread([y for y, _ in sorted(at10)])):
            ax.plot([10], [y], "o", color=COLORS[name], markersize=4, zorder=3)
            ax.annotate(f"{y:.2f}", (10, y), xytext=(2.4, y_label), textcoords="data",
                        fontsize=8, color=INK_2, va="center",
                        bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none"},
                        arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.8})
        ax.axvline(10, color=GRID, linewidth=1, linestyle="--")
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 50], ["1", "5", "10", "50"])
        ax.set_title("all dev queries" if kind == "all" else KIND_TITLES.get(kind, kind),
                     fontsize=9, color=INK, loc="left")
        ax.set_xlabel("k", color=INK_2, fontsize=9)
    axes[0].set_ylabel("Recall@k", color=INK_2, fontsize=9)
    axes[0].set_ylim(0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_per_kind(results: dict[str, EvalResult], path: Path) -> None:
    """MRR@10 per query kind and language, grouped by system."""
    import matplotlib.pyplot as plt

    names = [n for n in ("bm25", "base", "tuned") if n in results]
    groups = ([k.kind for k in results[names[0]].per_kind]
              + [k.kind for k in results[names[0]].per_lang] + ["all"])
    fig, ax = plt.subplots(figsize=(8.5, 3.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)
    width = 0.8 / len(names)
    x = np.arange(len(groups))
    for i, name in enumerate(names):
        res = results[name]
        vals = {k.kind: k.mrr10 for k in res.per_kind + res.per_lang + [res.overall]}
        ys = [vals.get(g, 0.0) for g in groups]
        # 2px surface gap between neighbouring bars.
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * width, ys, width * 0.92,
                      color=COLORS[name], label=LABELS[name], edgecolor=SURFACE, linewidth=1)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + 0.015, f"{y:.2f}", ha="center",
                    fontsize=7, color=INK_2)
    ax.set_xticks(x, ["RU queries" if g == "ru" else "KK queries" if g == "kk" else g
                      for g in groups])
    ax.set_ylabel("MRR@10", color=INK_2, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.18), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_train_loss(meta: dict, path: Path) -> bool:
    import matplotlib.pyplot as plt

    hist = [h for h in meta.get("log_history", []) if "loss" in h and "epoch" in h]
    if not hist:
        return False
    epoch = np.array([float(h["epoch"]) for h in hist])
    loss = np.array([float(h["loss"]) for h in hist])
    fig, ax = plt.subplots(figsize=(6, 3), facecolor=SURFACE)
    _style(ax)
    ax.plot(epoch, loss, color=COLORS["tuned"], linewidth=1, alpha=0.35)
    # Moving average over 5 logging steps (100 optimiser steps) for the trend.
    if len(loss) >= 5:
        smooth = np.convolve(loss, np.ones(5) / 5, mode="valid")
        ax.plot(epoch[2:-2], smooth, color=COLORS["tuned"], linewidth=2)
    ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    ax.set_ylabel("training loss (MNRL)", color=INK_2, fontsize=9)
    ax.set_title("Fine-tuning loss — logged every 20 steps, 5-point moving average",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


# ----------------------------------------------------------------- report --

def _pct(v: float) -> str:
    return f"{v:.3f}"


def results_markdown(results: dict[str, EvalResult], stats: dict, errors: dict | None,
                     data: dict, train_meta: dict | None, numbers: dict | None = None) -> str:
    """The generated half of the report: every number comes from this run."""
    lines = ["# Baseline experiment — generated results", "",
             f"_Generated by `lawrag experiment` on {time.strftime('%Y-%m-%d %H:%M')}. "
             "Do not edit by hand; re-run the command._", "",
             "## Data", "",
             "| split | pairs | articles | heading | xref | xlang | definition |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split in ("train", "dev"):
        d = data[split]
        lines.append(f"| {split} | {d['pairs']:,} | {d['articles']:,} | "
                     + " | ".join(f"{d['kinds'].get(k, 0):,}"
                                  for k in ("heading", "xref", "xlang", "definition")) + " |")
    lines += ["", f"Corpus: {data['chunks']:,} chunks. Train and dev share "
              f"{data['shared_articles']} articles.", ""]

    if train_meta:
        s = train_meta.get("settings", {})
        lines += ["## Training configuration", "", "| parameter | value |", "| --- | --- |"]
        rows = [("base model", Path(str(train_meta.get("base", ""))).name),
                ("training pairs", f"{train_meta.get('pairs', 0):,}"),
                ("epochs", s.get("epochs")), ("batch size (in-batch negatives)", s.get("batch_size")),
                ("mini-batch (GradCache)", s.get("mini_batch_size")),
                ("learning rate", s.get("learning_rate")), ("warm-up", "10% linear"),
                ("max sequence length", s.get("max_seq_length")),
                ("trainable layers (top)", s.get("train_layers")),
                ("trainable / frozen parameters",
                 f"{train_meta.get('trainable_parameters', 0) / 1e6:.1f}M / "
                 f"{train_meta.get('frozen_parameters', 0) / 1e6:.1f}M"
                 if train_meta.get("frozen_parameters") else
                 f"{train_meta.get('trainable_parameters', 0) / 1e6:.1f}M trainable"),
                ("hard negatives", f"1 per pair, top-{s.get('hard_negative_top_k')} under base model"),
                ("dev articles held out of negatives", train_meta.get("held_out_articles", "no")),
                ("seed", s.get("seed")),
                ("training time", f"{train_meta.get('train_seconds', 0) / 60:.0f} min")]
        lines += [f"| {k} | {v} |" for k, v in rows] + [""]

    lines += ["## Overall results (dev, all queries)", "",
              "95% bootstrap confidence intervals over queries (2,000 resamples).", "",
              "| system | Recall@1 | Recall@10 | MRR@10 |", "| --- | --- | --- | --- |"]
    for name, m in stats["systems"].items():
        cells = [f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
                 for v in (m["recall@1"], m["recall@10"], m["mrr@10"])]
        lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + " |")
    if stats["paired"]:
        lines += ["", "Paired differences (fine-tuned minus other system, same queries):", "",
                  "| comparison | Δ MRR@10 | Δ Recall@10 | queries better | queries worse |",
                  "| --- | --- | --- | ---: | ---: |"]
        for key, m in stats["paired"].items():
            other = LABELS[key.split("-", 1)[1]]
            lines.append(
                f"| fine-tuned vs {other} | {m['mrr@10'][0]:+.3f} [{m['mrr@10'][1]:+.3f}, "
                f"{m['mrr@10'][2]:+.3f}] | {m['recall@10'][0]:+.3f} [{m['recall@10'][1]:+.3f}, "
                f"{m['recall@10'][2]:+.3f}] | {m['better']} | {m['worse']} |")

    lines += ["", "## By query kind and language", "",
              "| system | slice | n | Recall@1 | Recall@5 | Recall@10 | MRR@10 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, res in results.items():
        for k in res.per_kind + res.per_lang:
            lines.append(f"| {LABELS[name]} | {k.kind} | {k.n} | {_pct(k.r1)} | {_pct(k.r5)} | "
                         f"{_pct(k.r10)} | {_pct(k.mrr10)} |")

    if numbers:
        lines += ["", "## Shortcut check — leftover article numbers in `xref` queries", "",
                  "| system | MRR@10, query names another article number | MRR@10, no number | "
                  "misses whose top-1 is the article the query names |",
                  "| --- | --- | --- | ---: |"]
        for name, m in numbers.items():
            lines.append(f"| {LABELS[name]} | {m['with_number']['mrr@10']:.3f} "
                         f"(n={m['with_number']['n']}) | {m['without_number']['mrr@10']:.3f} "
                         f"(n={m['without_number']['n']}) | {m['lured_share_of_misses']:.0%} |")
    if errors:
        lines += ["", "## Error analysis — fine-tuned model", "",
                  f"{errors['misses_at_1']} of {errors['queries']} dev queries do not have the "
                  f"right article at rank 1 (median rank of those: "
                  f"{errors['median_rank_of_misses']:.0f}). Where the wrong top-1 hit came from:", "",
                  "| where the top-1 landed | queries | share of misses |", "| --- | ---: | ---: |"]
        for cat, n in errors["categories"].items():
            lines.append(f"| {cat} | {n} | {n / max(1, errors['misses_at_1']):.0%} |")
        lines += ["", "By query kind:", "", "| kind | " + " | ".join(errors["categories"]) + " |",
                  "| --- |" + " ---: |" * len(errors["categories"])]
        for kind, cats in sorted(errors["categories_by_kind"].items()):
            lines.append(f"| {kind} | " + " | ".join(str(cats.get(c, 0))
                                                    for c in errors["categories"]) + " |")
        if errors.get("xref_quality"):
            lines += ["", "`xref` queries by data quality (mining artefacts found during this "
                      "analysis):", "", "| xref query | n | Recall@10 | MRR@10 |",
                      "| --- | ---: | ---: | ---: |"]
            for flag, m in errors["xref_quality"].items():
                lines.append(f"| {flag} | {m['n']} | {m['recall@10']:.3f} | {m['mrr@10']:.3f} |")
        lines += ["", "Random `xref` queries the fine-tuned model misses (right article not in "
                  "the top 10):", ""]
        for i, ex in enumerate(errors["examples"], 1):
            q = ex["query"] if len(ex["query"]) <= 220 else ex["query"][:217] + "…"
            lines += [f"{i}. **[{ex['lang']}]** {q}",
                      f"   - expected: `{ex['expected']}` (rank {ex['rank']})",
                      f"   - top-1: `{ex['retrieved']}` — *{ex['category']}*", ""]
    return "\n".join(lines) + "\n"


def _portable(meta: dict) -> dict:
    """Training metadata without the loss history and with model paths cut
    to their directory names."""
    out = {k: v for k, v in meta.items() if k != "log_history"}
    out["base"] = Path(str(out.get("base", ""))).name
    if "settings" in out:
        out["settings"] = {**out["settings"],
                           "base": Path(str(out["settings"].get("base", ""))).name,
                           "out": Path(str(out["settings"].get("out", ""))).name}
    return out


def data_summary(train: list[Pair], dev: list[Pair], corpus: list[dict]) -> dict:
    def one(ps: list[Pair]) -> dict:
        kinds: dict[str, int] = {}
        for p in ps:
            kinds[p.kind] = kinds.get(p.kind, 0) + 1
        return {"pairs": len(ps), "articles": len({p.article for p in ps}), "kinds": kinds}

    return {"train": one(train), "dev": one(dev), "chunks": len(corpus),
            "shared_articles": len({p.article for p in train} & {p.article for p in dev})}


# ----------------------------------------------------------------- driver --

def run_experiment(cfg: Config, *, corpus_dir: Path, pairs_dir: Path, model_dir: Path,
                   base_name: str, tuned_name: str, out_dir: Path, train: bool = True,
                   device: str | None = None, dev_fraction: float = 0.05) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. data
    ensure_pairs(cfg, corpus_dir, pairs_dir, dev_fraction)
    train_pairs = read_pairs(pairs_dir / "pairs.train.jsonl")
    dev_pairs = read_pairs(pairs_dir / "pairs.dev.jsonl")
    corpus = read_jsonl(pairs_dir / "corpus.jsonl")
    data = data_summary(train_pairs, dev_pairs, corpus)
    if not dev_pairs:
        raise RuntimeError(f"the dev split in {pairs_dir} is empty: the corpus is too small "
                           "for the dev fraction")
    if data["shared_articles"]:
        raise RuntimeError(f"{data['shared_articles']} articles are in both train and dev")

    # 2. model
    base, tuned = model_dir / base_name, model_dir / tuned_name
    has_tuned = (tuned / "config.json").exists()
    if train and not has_tuned:
        has_tuned = ensure_model(base, tuned, pairs_dir, train_pairs, dev_pairs, corpus, device)
    elif not has_tuned:
        log.warning("no fine-tuned model at %s and --no-train: evaluating baselines only", tuned)

    # 3. evaluate — every dev query, no sampling. A missing weights directory
    # drops that system (with a warning) instead of reaching for the network.
    systems: dict[str, Path | str] = {"bm25": BM25_MODEL}
    if (base / "config.json").exists():
        systems["base"] = base
    else:
        log.warning("no base model at %s: zero-shot dense baseline skipped", base)
    if has_tuned:
        systems["tuned"] = tuned
    results: dict[str, EvalResult] = {}
    for name, model in systems.items():
        log.info("evaluating %s", name)
        results[name] = evaluate(model, dev_pairs, corpus, device=device,
                                 cache_dir=pairs_dir / "cache", embedding_cfg=cfg.embedding)

    # 4. report
    stats = metric_table(results)
    errors = error_analysis(results["tuned"], corpus) if has_tuned else None
    train_meta = None
    if has_tuned and (tuned / "lawrag_train.json").exists():
        train_meta = json.loads((tuned / "lawrag_train.json").read_text(encoding="utf-8"))

    figures = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        plot_recall_at_k(results, out_dir / "fig_recall_at_k.png")
        plot_per_kind(results, out_dir / "fig_mrr_by_slice.png")
        figures = ["fig_recall_at_k.png", "fig_mrr_by_slice.png"]
        if train_meta and plot_train_loss(train_meta, out_dir / "fig_train_loss.png"):
            figures.append("fig_train_loss.png")
    except ImportError:
        log.warning("matplotlib not installed: figures skipped (pip install matplotlib)")

    numbers = number_slice(results)
    md = results_markdown(results, stats, errors, data, train_meta, numbers)
    if figures:
        md += "\n## Figures\n\n" + "\n".join(f"![{f}]({f})" for f in figures) + "\n"
    (out_dir / "results.md").write_text(md, encoding="utf-8")
    summary = {
        "data": data,
        # Directory names only: the report is committed and must not carry
        # this machine's paths.
        "systems": {n: Path(str(m)).name for n, m in systems.items()},
        "results": {n: {**r.to_dict(), "model": Path(r.model).name}
                    for n, r in results.items()},
        "stats": stats,
        "errors": errors,
        "number_shortcut": numbers,
        "training": (_portable(train_meta) if train_meta else None),
        "figures": figures,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    log.info("report written to %s", out_dir)
    return summary


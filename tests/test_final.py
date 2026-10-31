"""The final experiment's building blocks: split, searches, hybrid, LLM prompts.

Pure numpy / scikit-learn / a fake tokenizer — no weights and no GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lawrag.config import Config
from lawrag.train import ltr, tuning
from lawrag.train import pairs as P
from lawrag.train.evaluate import EvalResult, rank_pairs, ranks_from_scores, top_k_indices
from lawrag.train.final import Stages, canonical, paired
from lawrag.train.lexical import BM25
from lawrag.train.llm_rerank import PREFIX, SUFFIX, RerankSettings, build_ids, make_groups

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "synthetic"


@pytest.fixture(scope="module")
def mined():
    cfg = Config()
    cfg.db.dsn = "postgresql://nobody@127.0.0.1:1/none"
    cfg.db.connect_timeout_s = 1
    pairs, corpus = P.mine_directory(cfg, SAMPLE, query_log=False)
    return pairs, corpus.to_records()


def _naive_ranks(scores, keys, targets):
    out = np.zeros(len(targets), dtype=np.int64)
    for i, t in enumerate(targets):
        for rank, j in enumerate(np.argsort(-scores[i], kind="stable"), 1):
            if keys[j] == t:
                out[i] = rank
                break
    return out


def test_fast_ranks_equal_a_stable_sort_including_ties():
    rng = np.random.default_rng(0)
    keys = [(f"a{rng.integers(6)}", "ru" if rng.random() < .5 else "kk") for _ in range(40)]
    scores = rng.integers(0, 4, size=(200, 40)).astype(np.float32)   # many ties
    scores[:5, :] = -np.inf                                            # all masked
    targets = [keys[rng.integers(40)] for _ in range(200)] + [("missing", "ru")]
    scores = np.vstack([scores, np.zeros((1, 40), dtype=np.float32)])
    assert (ranks_from_scores(scores, keys, targets) == _naive_ranks(scores, keys, targets)).all()


def test_top_k_is_best_first():
    s = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert top_k_indices(s, 3).tolist() == [[1, 3, 2]]


def test_validation_is_disjoint_and_leaves_test_untouched(mined):
    pairs, _ = mined
    train, test = P.split_pairs(pairs, dev_fraction=0.2)
    fit, val = P.carve_validation(train, fraction=0.3)
    arts = [{p.article for p in s} for s in (fit, val, test)]
    assert not arts[0] & arts[1] and not arts[0] & arts[2] and not arts[1] & arts[2]
    assert {p.kind for p in val} <= P.VALIDATION_KINDS
    assert P.carve_validation(train, fraction=0.3) == (fit, val)        # deterministic
    assert P.split_pairs(pairs, dev_fraction=0.2)[1] == test


def test_bm25_reweight_equals_a_fresh_fit(mined):
    _, corpus = mined
    texts = [c["text"] for c in corpus]
    queries = [c["text"][:60] for c in corpus[:5]]
    a = BM25(k1=1.2, b=0.75, prefix=5).fit(texts).reweight(2.0, 0.3)
    b = BM25(k1=2.0, b=0.3, prefix=5).fit(texts)
    assert np.allclose(a.score(queries), b.score(queries))


def test_bm25_grid_reports_default_and_best(mined):
    pairs, corpus = mined
    val = [p for p in pairs if p.kind in P.VALIDATION_KINDS]
    grid = {"prefix": [5, 6], "k1": [1.2, 2.0], "b": [0.5, 0.75]}
    res = tuning.bm25_grid_search(val, corpus, grid)
    assert res["n_trials"] == 8
    assert res["default"]["prefix"] == 6 and res["default"]["k1"] == 1.2
    assert res["best"]["mrr@10"] == max(t["mrr@10"] for t in res["trials"])
    assert res["best"]["mrr@10"] >= res["default"]["mrr@10"]


def test_result_round_trips_through_its_queries(mined):
    pairs, corpus = mined
    val = canonical([p for p in pairs if p.kind in P.VALIDATION_KINDS])
    bm25 = BM25().fit([c["text"] for c in corpus])
    res = rank_pairs(bm25.score, val, corpus, name="bm25")
    again = EvalResult.from_queries("bm25", res.queries)
    assert again.to_dict()["overall"] == res.to_dict()["overall"]
    assert [q["query"] for q in res.queries] == [p.query for p in val]   # canonical order
    d = paired(res, again)
    assert d["mrr@10"][0] == 0 and d["better"] == d["worse"] == 0


def test_hybrid_ranks_follow_the_probabilities(mined):
    pairs, corpus = mined
    val = canonical([p for p in pairs if p.kind in P.VALIDATION_KINDS])[:20]
    bm25 = BM25().fit([c["text"] for c in corpus])
    rng = np.random.default_rng(0)
    noise = rng.random((len(val), len(corpus))).astype(np.float32)
    lookup = {p.query: i for i, p in enumerate(val)}
    dense = lambda qs: noise[[lookup[q] for q in qs]]           # noqa: E731
    c = ltr.build_candidates(val, corpus, dense, bm25.score, k=3)
    assert c.X.shape == (len(c.y), len(ltr.FEATURES))
    # An oracle puts the right article first wherever it is a candidate...
    ranks = ltr.ranks_from_probs(c, c.y.astype(float))
    has = np.array([c.y[c.query_ids == q].any() for q in range(len(val))])
    assert (ranks[has] == 1).all()
    # ...and the rest are reported below every candidate.
    assert (ranks[~has & (c.fallback_rank > 0)] > c.n_candidates[~has & (c.fallback_rank > 0)]).all()


def test_hybrid_search_space_is_sampled_reproducibly():
    a = ltr.sample_space(ltr.HGB_SPACE, 5, seed=1)
    assert a == ltr.sample_space(ltr.HGB_SPACE, 5, seed=1)
    for point in a:
        assert 0.02 <= point["learning_rate"] <= 0.3
        assert point["max_leaf_nodes"] in ltr.HGB_SPACE["max_leaf_nodes"]


class _CharTok:
    """One token per character: enough to test the prompt arithmetic."""

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_prompt_keeps_its_frame_when_the_document_is_cut():
    tok = _CharTok()
    ids = build_ids(tok, "q" * 500, "d" * 5000, max_len=900, max_query_tokens=50)
    text = tok.decode(ids)
    assert text.startswith(PREFIX) and text.endswith(SUFFIX)
    assert len(ids) <= 900
    assert "q" * 50 in text and "q" * 51 not in text


def test_reranker_groups_exclude_held_out_and_duplicate_text():
    corpus = [{"id": f"c{i}", "article": f"x/st{i}", "lang": "ru", "text": f"text {i}"}
              for i in range(12)]
    corpus.append({"id": "dup", "article": "x/st99", "lang": "ru", "text": "text 0"})
    pair = P.Pair(query="q", pos="c0", article="x/st0", lang="ru", pos_lang="ru", kind="xref")
    settings = RerankSettings(base="", out="", group_size=4, hard_negatives=2, negative_pool=30)
    top = [12, 0, 1, 2, 3, 4, 5, 6]
    (g,) = make_groups([pair], [top], corpus, settings, held_out=frozenset({"x/st1"}))
    assert g["positive"] == "text 0" and len(g["negatives"]) == 3
    assert "text 0" not in g["negatives"] and "text 1" not in g["negatives"]
    assert g["negatives"][:2] == ["text 2", "text 3"]                 # hardest first


def test_stages_cache(tmp_path):
    st = Stages(tmp_path)
    calls = []
    assert st.cached("x", lambda: calls.append(1) or {"v": 1}) == {"v": 1}
    assert st.cached("x", lambda: calls.append(1) or {"v": 2}) == {"v": 1}
    assert calls == [1]

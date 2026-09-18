"""Training-pair mining and the database-free evaluator.

Runs on the synthetic fixture and needs neither Postgres nor torch: the pair
miner is pure parsing, and the ranking metric is plain numpy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lawrag.config import Config
from lawrag.train import pairs as P
from lawrag.train.evaluate import ranks_from_scores, summarise

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "synthetic"


@pytest.fixture(scope="module")
def mined():
    cfg = Config()
    cfg.db.dsn = "postgresql://nobody@127.0.0.1:1/none"   # query log must just skip
    cfg.db.connect_timeout_s = 1
    return P.mine_directory(cfg, SAMPLE, query_log=True)


def test_every_pair_points_at_a_real_chunk(mined):
    pairs, corpus = mined
    assert pairs
    for p in pairs:
        chunk = corpus.chunks[p.pos]
        assert chunk.article_node_id == p.article
        assert chunk.lang.value == p.pos_lang
        assert not chunk.repealed, "a repeal stub is not a positive"


def test_the_corpus_has_every_signal(mined):
    pairs, _ = mined
    kinds = {p.kind for p in pairs}
    assert {"heading", "xref", "xlang", "definition"} <= kinds
    # Nothing was generated or logged, so nothing may claim to be.
    assert not kinds & {"doc2query", "querylog"}


def test_citation_queries_do_not_contain_the_cited_number(mined):
    pairs, _ = mined
    xrefs = [p for p in pairs if p.kind == "xref"]
    assert xrefs
    for p in xrefs:
        number = p.article.split("/st")[1]
        assert f"стать{number}" not in p.query.replace(" ", "").lower()
        assert f"{number}-бап" not in p.query
        assert P._MIN_QUERY <= len(p.query) <= P._MAX_QUERY


def test_cross_language_pairs_go_both_ways(mined):
    pairs, _ = mined
    xl = [p for p in pairs if p.kind == "xlang"]
    assert {(p.lang, p.pos_lang) for p in xl} == {("ru", "kk"), ("kk", "ru")}


def test_definitions_ask_for_the_term(mined):
    pairs, corpus = mined
    defs = [p for p in pairs if p.kind == "definition"]
    assert defs
    for p in defs:
        term = p.query.strip("«»?").replace("что означает термин", "").replace("деген не", "").strip(" «»?")
        assert term.lower() in corpus.chunks[p.pos].display_text.lower()


def test_split_holds_out_whole_articles(mined):
    pairs, _ = mined
    train, dev = P.split_pairs(pairs, dev_fraction=0.3, seed=1)
    assert train and dev
    assert not {p.article for p in train} & {p.article for p in dev}
    # Deterministic: the same seed gives the same split.
    again_train, _ = P.split_pairs(pairs, dev_fraction=0.3, seed=1)
    assert [p.pos for p in again_train] == [p.pos for p in train]


def test_cap_per_kind_keeps_the_others_intact(mined):
    pairs, _ = mined
    n_heading = sum(1 for p in pairs if p.kind == "heading")
    capped = P.cap_per_kind(pairs, {"xlang": 2})
    assert sum(1 for p in capped if p.kind == "xlang") == 2
    assert sum(1 for p in capped if p.kind == "heading") == n_heading


def test_pairs_round_trip_through_jsonl(mined, tmp_path):
    pairs, corpus = mined
    path = tmp_path / "pairs.jsonl"
    P.write_jsonl(path, (p.to_dict() for p in pairs[:5]))
    assert P.read_pairs(path) == pairs[:5]
    P.write_jsonl(tmp_path / "corpus.jsonl", corpus.to_records())
    rec = P.read_jsonl(tmp_path / "corpus.jsonl")[0]
    assert {"id", "article", "lang", "text"} <= rec.keys()


def test_ranking_metric():
    keys = [("a/st1", "ru"), ("a/st2", "ru"), ("a/st2", "kk"), ("a/st3", "ru")]
    scores = np.array([
        [0.1, 0.9, 0.2, 0.3],   # target a/st2 ru at rank 1
        [0.9, 0.1, 0.2, 0.8],   # target a/st3 ru at rank 2
        [0.9, 0.8, 0.7, 0.6],   # target a/st2 kk at rank 3
        [0.5, 0.5, 0.5, 0.5],   # target not in corpus
    ])
    targets = [("a/st2", "ru"), ("a/st3", "ru"), ("a/st2", "kk"), ("zz/st9", "ru")]
    ranks = ranks_from_scores(scores, keys, targets)
    assert ranks.tolist() == [1, 2, 3, 0]
    r = summarise("t", ranks)
    assert r.n == 4 and r.r1 == 0.25 and r.r5 == 0.75 and r.r10 == 0.75
    assert r.mrr10 == pytest.approx((1 + 0.5 + 1 / 3) / 4)


def test_fetcher_refuses_the_javascript_shell():
    from lawrag.ingest.fetch_adilet import is_app_shell

    shell = ("<html><head><title>НЦПС «Әділет»</title></head><body>"
             "<p>Для работы с поиском и текстами актов включите JavaScript.</p></body></html>")
    assert is_app_shell(shell)
    page = "<html><body><p>Статья 1. Предмет регулирования</p><p>1. Настоящий Кодекс регулирует.</p></body></html>"
    assert not is_app_shell(page)

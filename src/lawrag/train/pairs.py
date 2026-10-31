"""Training pairs mined from the corpus, with no labelled data.

Every pair is (query, positive chunk). The positive is a *chunk* — the exact
unit the index embeds, built by the same ``Chunker`` — so what the model is
trained to match is what it will be asked to match at query time.

The split into train and dev is by *article*, not by pair: a dev query must
point at an article the model never saw as a positive during training, or the
dev numbers say nothing about generalisation.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from ..chunking.chunker import Chunker
from ..config import ChunkConfig, Config
from ..ids import node_article_id
from ..ingest import refs as R
from ..ingest import structure as S
from ..ingest.parser import parse_act_source, parse_adilet
from ..logging_setup import get_logger
from ..models import Chunk, Lang, Node, NodeKind

log = get_logger(__name__)

KINDS = ("heading", "xref", "xlang", "definition", "doc2query", "querylog")


@dataclass(slots=True)
class Pair:
    query: str
    pos: str            # chunk id
    article: str        # article node id of the positive — the split unit
    lang: str           # language of the query
    pos_lang: str       # language of the positive chunk
    kind: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Corpus:
    """Every chunk of a corpus directory, as the index would hold it."""

    chunks: dict[str, Chunk]

    def by_article(self, article: str, lang: str) -> list[Chunk]:
        return sorted(
            (c for c in self.chunks.values()
             if c.article_node_id == article and c.lang.value == lang),
            key=lambda c: (c.valid_from, c.id),
        )

    def first_chunk(self, article: str, lang: str) -> Chunk | None:
        """The article-level chunk, or the first point chunk of a split article.

        Prefers the newest redaction: that is the text the current index
        answers from, and the one the query log's questions were asked about.
        """
        chunks = self.by_article(article, lang)
        if not chunks:
            return None
        newest = max(c.valid_from for c in chunks)
        chunks = [c for c in chunks if c.valid_from == newest]
        whole = [c for c in chunks if c.node_id == article]
        return whole[0] if whole else chunks[0]

    def to_records(self) -> list[dict]:
        return [
            {"id": c.id, "article": c.article_node_id, "node": c.node_id,
             "lang": c.lang.value, "text": c.embed_text, "repealed": c.repealed}
            for c in self.chunks.values()
        ]


# ---------------------------------------------------------------- loading --

def load_trees(path: Path) -> list[tuple[Node, "SourceDoc"]]:  # noqa: F821
    """Parse every act source in ``path``, exactly as ``ingest_directory`` does."""
    out = []
    for source in sorted(path.glob("*.act.txt")):
        _act, root, doc = parse_act_source(source.read_text(encoding="utf-8"))
        out.append((root, doc))
    index = path / "documents.json"
    if index.exists():
        for entry in json.loads(index.read_text(encoding="utf-8")):
            html_path = path / entry["file"]
            if not html_path.exists():
                continue
            _act, root, doc = parse_adilet(html_path.read_text(encoding="utf-8"), entry)
            out.append((root, doc))
    if not out:
        raise FileNotFoundError(f"no *.act.txt or adilet documents.json found in {path}")
    return out


def build_corpus(trees: Iterable[tuple[Node, "SourceDoc"]],  # noqa: F821
                 chunk_cfg: ChunkConfig | None = None) -> Corpus:
    chunker = Chunker(chunk_cfg or ChunkConfig())
    chunks: dict[str, Chunk] = {}
    for root, doc in trees:
        for chunk in chunker.chunk_act(root, doc.lang):
            chunks[chunk.id] = chunk
    return Corpus(chunks)


# ----------------------------------------------------------------- mining --

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.;!?])\s+|\n+")
_MIN_QUERY = 40
_MAX_QUERY = 600


def _sentence_around(text: str, start: int, end: int) -> str:
    """The sentence of ``text`` that spans [start, end)."""
    left = 0
    for m in _SENTENCE_BOUNDARY.finditer(text, 0, start):
        left = m.end()
    right = len(text)
    m = _SENTENCE_BOUNDARY.search(text, end)
    if m:
        right = m.start()
    return text[left:right].strip()


def _usable(text: str) -> bool:
    return bool(text) and not S.is_repealed(text)


def mine_headings(trees, corpus: Corpus) -> Iterator[Pair]:
    """Article title -> the article. Short, plentiful, and exactly the kind of
    terse query a compliance officer types."""
    for root, doc in trees:
        lang = doc.lang.value
        for article in root.articles():
            text = article.texts.get(doc.lang)
            if text is None or not _usable(text.body) or len(text.heading) < 8:
                continue
            chunk = corpus.first_chunk(article.id, lang)
            if chunk is None or chunk.repealed:
                continue
            yield Pair(text.heading.strip(), chunk.id, article.id, lang, lang, "heading")


def mine_xrefs(trees, corpus: Corpus, per_node: int = 3) -> Iterator[Pair]:
    """The sentence that cites article B, minus the citation, -> article B.

    "в порядке, установленном статьей 76 Налогового кодекса" becomes "в
    порядке, установленном Налогового кодекса" pointing at nk/st76: the model
    must learn that this sentence is *about* registration deadlines, since the
    number it could have matched on is gone.
    """
    for root, doc in trees:
        lang = doc.lang.value
        for node in root.walk():
            if node.kind not in (NodeKind.ARTICLE, NodeKind.POINT, NodeKind.NOTE):
                continue
            text = node.texts.get(doc.lang)
            if text is None or not text.body:
                continue
            body = text.body
            emitted = 0
            for ref in R.extract_refs(node.id, doc.act_code, body, doc.lang):
                if ref.dst_node_id is None or emitted >= per_node:
                    continue
                pos = corpus.first_chunk(ref.dst_node_id, lang)
                if pos is None or pos.repealed:
                    continue
                at = body.find(ref.raw)
                if at < 0:
                    continue
                sentence = _sentence_around(body, at, at + len(ref.raw))
                query = " ".join(sentence.replace(ref.raw, " ").split())
                if not (_MIN_QUERY <= len(query) <= _MAX_QUERY):
                    continue
                emitted += 1
                yield Pair(query, pos.id, ref.dst_node_id, lang, lang, "xref")


def mine_xlang(corpus: Corpus, max_chars: int = 1200) -> Iterator[Pair]:
    """Russian text of a node <-> its Kazakh text. Both directions.

    The two are legally the same norm and share almost no tokens; this is the
    only signal that pulls the two languages' legal vocabularies together in
    the embedding space, and it is what makes a Kazakh question find the
    Russian article and vice versa.
    """
    by_node: dict[tuple[str, str], Chunk] = {}
    for c in corpus.chunks.values():
        if c.repealed:
            continue
        key = (c.node_id, c.lang.value)
        prev = by_node.get(key)
        if prev is None or c.valid_from > prev.valid_from:
            by_node[key] = c
    for (node_id, lang), chunk in by_node.items():
        if lang != "ru":
            continue
        other = by_node.get((node_id, "kk"))
        if other is None:
            continue
        ru = chunk.display_text[:max_chars]
        kk = other.display_text[:max_chars]
        if len(ru) < _MIN_QUERY or len(kk) < _MIN_QUERY:
            continue
        yield Pair(ru, other.id, other.article_node_id, "ru", "kk", "xlang")
        yield Pair(kk, chunk.id, chunk.article_node_id, "kk", "ru", "xlang")


_DEF_TEMPLATES = {
    "ru": ("{term}", "что означает термин «{term}»?"),
    "kk": ("{term}", "«{term}» деген не?"),
}


def mine_definitions(trees, corpus: Corpus) -> Iterator[Pair]:
    for root, doc in trees:
        lang = doc.lang.value
        for article in root.articles():
            text = article.texts.get(doc.lang)
            if text is None or not R.is_definition_article(text.heading):
                continue
            chunks = corpus.by_article(article.id, lang)
            if not chunks:
                continue
            for term, definition in R.extract_definitions(article.id, doc.act_code, text.body):
                probe = definition[:60]
                pos = next((c for c in chunks if probe in c.display_text), None)
                if pos is None:
                    continue
                for template in _DEF_TEMPLATES[lang]:
                    yield Pair(template.format(term=term), pos.id, article.id, lang, lang,
                               "definition")


# -------------------------------------------------------------- doc2query --

_D2Q_PROMPT = {
    "ru": ("Ниже приведён фрагмент закона Республики Казахстан. Напишите {n} разных "
           "вопроса, которые мог бы задать сотрудник по комплаенсу без юридического "
           "образования и на которые этот фрагмент отвечает. Не называйте номер статьи "
           "и кодекс. По одному вопросу в строке, без нумерации и пояснений.\n\n{text}"),
    "kk": ("Төменде Қазақстан Республикасы заңының үзіндісі берілген. Заң білімі жоқ "
           "комплаенс қызметкері қоюы мүмкін және осы үзінді жауап беретін {n} түрлі "
           "сұрақ жазыңыз. Бап нөмірін және кодексті атамаңыз. Әр сұрақ жеке жолда, "
           "нөмірсіз және түсіндірмесіз.\n\n{text}"),
}


def _content_key(chunk: Chunk) -> str:
    return hashlib.sha256(chunk.embed_text.encode("utf-8")).hexdigest()[:16]


def mine_doc2query(cfg: Config, corpus: Corpus, cache: Path, n: int = 3,
                   max_chunks: int | None = None) -> Iterator[Pair]:
    """Questions written by the local generator, one call per chunk, cached.

    Cached by content hash so that a re-run after a corpus update only asks
    about the chunks that changed. Skipped outright on the echo backend: it
    would cache refusals as questions.
    """
    from ..generate.llm import build_llm

    llm = build_llm(cfg.llm)
    if getattr(llm, "name", "") == "echo":
        log.warning("doc2query skipped: llm.backend is 'echo' (no generator configured)")
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, list[str]] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = rec["questions"]
    todo = [c for c in corpus.chunks.values() if not c.repealed and _content_key(c) not in done]
    if max_chunks is not None:
        todo = todo[:max_chunks]
    log.info("doc2query: %d chunks cached, %d to generate", len(done), len(todo))
    with cache.open("a", encoding="utf-8") as sink:
        for i, chunk in enumerate(todo, 1):
            prompt = _D2Q_PROMPT[chunk.lang.value].format(n=n, text=chunk.display_text[:4000])
            try:
                raw = llm.complete(prompt, max_tokens=400, temperature=0.7)
            except Exception as exc:  # noqa: BLE001 — one failed call must not lose the run
                log.warning("doc2query failed for %s: %s", chunk.id, exc)
                continue
            questions = [q.strip(" -•*0123456789.)") for q in raw.splitlines()]
            questions = [q for q in questions if _MIN_QUERY // 4 <= len(q) <= _MAX_QUERY][:n]
            key = _content_key(chunk)
            done[key] = questions
            sink.write(json.dumps({"key": key, "chunk": chunk.id, "questions": questions},
                                  ensure_ascii=False) + "\n")
            if i % 50 == 0:
                log.info("doc2query: %d/%d", i, len(todo))
    for chunk in corpus.chunks.values():
        for q in done.get(_content_key(chunk), []):
            yield Pair(q, chunk.id, chunk.article_node_id, chunk.lang.value,
                       chunk.lang.value, "doc2query")


# -------------------------------------------------------------- query log --

_VERIFIED = re.compile(r"^(\d+)/(\d+) citations verified$")


def mine_query_log(cfg: Config, corpus: Corpus) -> Iterator[Pair]:
    """Real questions whose every citation passed mechanical verification.

    The most valuable pairs there are — actual users, actual phrasing — and
    the only ones that reflect what people ask rather than what the text
    says. Empty until the system has been used; silently empty when there is
    no database to read.
    """
    from ..index.db import Database

    try:
        with Database(cfg).connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT question, lang, cited, verdict, abstained FROM query_log")
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        log.warning("query log not read (%s); skipping querylog pairs", exc)
        return
    for row in rows:
        m = _VERIFIED.match(row["verdict"] or "")
        if row["abstained"] or not m or m.group(1) != m.group(2) or m.group(2) == "0":
            continue
        for cite in row["cited"] or []:
            node = cite.get("node_id") if isinstance(cite, dict) else None
            if not node and isinstance(cite, dict) and cite.get("act_code") and cite.get("article"):
                node = f"{cite['act_code']}/st{cite['article']}"
            if not node:
                continue
            article = node_article_id(node)
            pos = corpus.first_chunk(article, row["lang"])
            if pos is None:
                continue
            yield Pair(row["question"], pos.id, article, row["lang"], pos.lang.value, "querylog")


# ---------------------------------------------------------------- driver --

def _dev_article(article: str, dev_fraction: float, seed: int) -> bool:
    h = hashlib.blake2b(f"{seed}:{article}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "big") / 2**32 < dev_fraction


def split_pairs(pairs: list[Pair], dev_fraction: float = 0.05,
                seed: int = 13) -> tuple[list[Pair], list[Pair]]:
    """Hold out whole articles: every pair whose positive is in a dev article
    goes to dev, so nothing the dev set asks about was a training positive."""
    train: list[Pair] = []
    dev: list[Pair] = []
    for p in pairs:
        (dev if _dev_article(p.article, dev_fraction, seed) else train).append(p)
    return train, dev


def carve_validation(train: list[Pair], fraction: float = 0.05, seed: int = 29,
                     ) -> tuple[list[Pair], list[Pair]]:
    """Split the train pairs once more, by article, into (fit, validation).

    The held-out test split stays exactly what ``split_pairs`` made it, so
    final numbers remain comparable with every earlier report; validation is
    a second, independent hash bucket of the *train* articles, used for every
    model-selection decision (hyper-parameters, fusion weights). ``fraction``
    is of the train articles. Validation keeps only the query kinds the test
    split has, so the two measure the same thing.
    """
    fit: list[Pair] = []
    val: list[Pair] = []
    for p in train:
        (val if _dev_article(p.article, fraction, seed) else fit).append(p)
    return fit, [p for p in val if p.kind in VALIDATION_KINDS]


VALIDATION_KINDS = frozenset({"heading", "xref", "xlang"})


def dedupe(pairs: Iterable[Pair]) -> list[Pair]:
    seen: set[tuple[str, str]] = set()
    out: list[Pair] = []
    for p in pairs:
        key = (p.query, p.pos)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def cap_per_kind(pairs: list[Pair], caps: dict[str, int], seed: int = 13) -> list[Pair]:
    """Keep the mix balanced: 40k cross-language pairs would otherwise drown
    the 3k citation sentences that look most like real questions."""
    rng = random.Random(seed)
    by_kind: dict[str, list[Pair]] = {}
    for p in pairs:
        by_kind.setdefault(p.kind, []).append(p)
    out: list[Pair] = []
    for kind, items in by_kind.items():
        cap = caps.get(kind)
        if cap is not None and len(items) > cap:
            items = rng.sample(items, cap)
        out.extend(items)
    return out


DEFAULT_CAPS = {"xlang": 8000, "heading": 8000}


def mine_directory(cfg: Config, path: Path, *, doc2query: bool = False,
                   query_log: bool = True, cache_dir: Path | None = None,
                   caps: dict[str, int] | None = None) -> tuple[list[Pair], Corpus]:
    trees = load_trees(path)
    corpus = build_corpus(trees, cfg.chunking)
    log.info("corpus: %d documents, %d chunks", len(trees), len(corpus.chunks))

    pairs: list[Pair] = []
    pairs.extend(mine_headings(trees, corpus))
    pairs.extend(mine_xrefs(trees, corpus))
    pairs.extend(mine_xlang(corpus))
    pairs.extend(mine_definitions(trees, corpus))
    if doc2query:
        cache = (cache_dir or Path(cfg.data_dir) / "train") / "doc2query.jsonl"
        pairs.extend(mine_doc2query(cfg, corpus, cache))
    if query_log:
        pairs.extend(mine_query_log(cfg, corpus))
    pairs = cap_per_kind(dedupe(pairs), DEFAULT_CAPS if caps is None else caps)
    counts = {k: sum(1 for p in pairs if p.kind == k) for k in KINDS}
    log.info("pairs: %s", counts)
    return pairs, corpus


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_pairs(path: Path) -> list[Pair]:
    return [Pair(**rec) for rec in read_jsonl(path)]

"""Source text -> canonical tree.

Two adapters feed one line-based parser: a plain-text corpus format (used by the
sample data and by anything you export yourself) and an adilet HTML adapter. The
HTML adapter's only job is to produce clean lines; all structural understanding
lives in :func:`parse_lines`, so adding a third source format never touches the
part that decides what an article is.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import yaml

from ..ids import article_id, point_id, subpoint_id
from ..logging_setup import get_logger
from ..models import Act, ActKind, Lang, Node, NodeKind, NodeText, OPEN_ENDED
from . import structure as S

log = get_logger(__name__)


@dataclass(slots=True)
class SourceDoc:
    """One act, in one language, as one redaction."""

    act_code: str
    kind: ActKind
    official_id: str
    lang: Lang
    title: str
    body: str
    valid_from: date
    valid_to: date = OPEN_ENDED
    adopted_on: date | None = None
    url: str | None = None


# ---------------------------------------------------------------------------
# Plain-text corpus format
# ---------------------------------------------------------------------------

def parse_source_file(raw: str) -> SourceDoc:
    """Parse a ``.act.txt`` file: YAML front matter, then the act body."""
    if not raw.lstrip().startswith("---"):
        raise ValueError("source file must start with YAML front matter delimited by ---")
    stripped = raw.lstrip()
    _, fm, body = stripped.split("---", 2)
    meta = yaml.safe_load(fm) or {}

    def as_date(key: str, default: date | None = None) -> date | None:
        v = meta.get(key)
        if v is None:
            return default
        return v if isinstance(v, date) else date.fromisoformat(str(v))

    return SourceDoc(
        act_code=meta["code"],
        kind=ActKind(meta.get("kind", "code")),
        official_id=meta.get("official_id", ""),
        lang=Lang(meta.get("lang") or S.detect_lang(body).value),
        title=meta.get("title", ""),
        body=body.strip("\n"),
        valid_from=as_date("valid_from") or date(1900, 1, 1),
        valid_to=as_date("valid_to") or OPEN_ENDED,
        adopted_on=as_date("adopted_on"),
        url=meta.get("url"),
    )


# ---------------------------------------------------------------------------
# adilet HTML adapter
# ---------------------------------------------------------------------------

def adilet_html_to_lines(html: str) -> list[str]:
    """Reduce an adilet document page to clean text lines.

    Kept deliberately conservative: block-level tags become line breaks, tables
    are flattened with tab separators so the table detector downstream can still
    recognise them, and everything else is dropped.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    root = (
        soup.select_one("#docHtml")
        or soup.select_one(".doc-content")
        or soup.select_one("article")
        or soup.body
        or soup
    )

    lines: list[str] = []
    for row in root.find_all("tr"):
        cells = [" ".join(c.get_text(" ", strip=True).split()) for c in row.find_all(["td", "th"])]
        if any(cells):
            row.replace_with("\n" + "\t".join(cells) + "\n")

    text = root.get_text("\n")
    for raw_line in text.split("\n"):
        line = " ".join(raw_line.replace("\xa0", " ").split())
        if line:
            lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# The line-based structural parser
# ---------------------------------------------------------------------------

_CONTAINER_ORDER = {
    NodeKind.SECTION: 0,
    NodeKind.CHAPTER: 1,
    NodeKind.PARAGRAPH: 2,
}


def parse_lines(doc: SourceDoc, lines: Iterable[str]) -> tuple[Act, Node]:
    act = Act(
        code=doc.act_code,
        kind=doc.kind,
        official_id=doc.official_id,
        title={doc.lang.value: doc.title},
        adopted_on=doc.adopted_on,
        url=doc.url,
    )

    root = Node(id=doc.act_code, act_code=doc.act_code, kind=NodeKind.ACT,
                number=None, ordinal=0)
    root.texts[doc.lang] = NodeText(lang=doc.lang, heading=doc.title, body="",
                                    valid_from=doc.valid_from, valid_to=doc.valid_to)

    # Stack of open container nodes (act -> section -> chapter -> paragraph).
    stack: list[Node] = [root]
    current_article: Node | None = None
    buffer: list[str] = []
    counters: dict[str, int] = {}
    # Text that belongs to a container rather than to an article: a preamble, a
    # section's operative lead paragraph, transitional provisions. Discarding it
    # silently loses real law and leaves no trace in the logs.
    container_buffer: list[str] = []
    orphan_lines = 0

    def next_ordinal(parent_id: str) -> int:
        counters[parent_id] = counters.get(parent_id, 0) + 1
        return counters[parent_id]

    def flush_article() -> None:
        nonlocal buffer, current_article
        if current_article is None:
            buffer = []
            return
        body, provenance = S.strip_amendment_notes("\n".join(buffer))
        txt = current_article.texts[doc.lang]
        # The article's own text keeps its substantive notes — they are part of
        # what a user must read. Only the amending-law provenance is removed.
        txt.body = body
        if provenance:
            _attach_provenance(current_article, provenance)
        point_body, notes = S.split_notes(body)
        for note in notes:
            _attach_note(current_article, note, doc)
        _build_points(current_article, doc, point_body)
        buffer = []
        current_article = None

    def flush_container() -> None:
        nonlocal container_buffer
        text = "\n".join(container_buffer).strip()
        if text:
            node = stack[-1]
            existing = node.texts.get(doc.lang)
            if existing is None:
                node.texts[doc.lang] = NodeText(
                    lang=doc.lang, heading="", body=text,
                    valid_from=doc.valid_from, valid_to=doc.valid_to)
            else:
                existing.body = (existing.body + "\n" + text).strip()
        container_buffer = []

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            if current_article is not None:
                buffer.append("")
            continue

        heading = S.match_heading(line, doc.lang)
        if heading is None:
            if current_article is not None:
                buffer.append(line)
            else:
                container_buffer.append(line)
                orphan_lines += 1
            continue

        if heading.kind is NodeKind.ARTICLE:
            flush_article()
            flush_container()
            parent = stack[-1]
            nid = article_id(doc.act_code, heading.number)
            node = Node(id=nid, act_code=doc.act_code, kind=NodeKind.ARTICLE,
                        number=heading.number, ordinal=next_ordinal(parent.id),
                        parent_id=parent.id)
            node.texts[doc.lang] = NodeText(lang=doc.lang, heading=heading.title, body="",
                                            valid_from=doc.valid_from, valid_to=doc.valid_to)
            parent.children.append(node)
            current_article = node
            continue

        # A container heading closes any open article and rewinds the stack to
        # the correct depth.
        flush_article()
        flush_container()
        depth = _CONTAINER_ORDER[heading.kind]
        while len(stack) > 1 and _CONTAINER_ORDER.get(stack[-1].kind, -1) >= depth:
            stack.pop()
        parent = stack[-1]
        nid = f"{parent.id}/{heading.kind.value[:3]}{heading.number}"
        node = Node(id=nid, act_code=doc.act_code, kind=heading.kind,
                    number=heading.number, ordinal=next_ordinal(parent.id),
                    parent_id=parent.id)
        node.texts[doc.lang] = NodeText(lang=doc.lang, heading=heading.title, body="",
                                        valid_from=doc.valid_from, valid_to=doc.valid_to)
        parent.children.append(node)
        stack.append(node)

    flush_article()
    flush_container()
    n_articles = sum(1 for _ in root.articles())
    log.info("parsed %s [%s]: %d articles, %d lines attached to containers "
             "(preamble / transitional provisions)",
             doc.act_code, doc.lang.value, n_articles, orphan_lines)
    if n_articles == 0:
        log.warning(
            "%s [%s] produced zero articles — check that the heading dialect "
            "matches the document language", doc.act_code, doc.lang.value
        )
    return act, root


def _attach_note(article: Node, note: str, doc: "SourceDoc") -> None:
    """A substantive Примечание / Ескерту becomes its own retrievable node."""
    existing = sum(1 for c in article.children if c.kind is NodeKind.NOTE
                   and c.id.endswith(tuple(f"/note{i}" for i in range(1, 20))))
    nid = f"{article.id}/note{existing + 1}"
    node = Node(id=nid, act_code=article.act_code, kind=NodeKind.NOTE,
                number=str(existing + 1), ordinal=900 + existing,
                parent_id=article.id)
    node.texts[doc.lang] = NodeText(lang=doc.lang, heading="", body=note,
                                    valid_from=doc.valid_from, valid_to=doc.valid_to)
    article.children.append(node)


def _attach_provenance(article: Node, notes: list[str]) -> None:
    note_id = f"{article.id}/amend"
    node = Node(id=note_id, act_code=article.act_code, kind=NodeKind.NOTE,
                number=None, ordinal=999, parent_id=article.id)
    lang = next(iter(article.texts))
    node.texts[lang] = NodeText(lang=lang, heading="amendment-history",
                                body="\n".join(notes),
                                valid_from=article.texts[lang].valid_from,
                                valid_to=article.texts[lang].valid_to)
    article.children.append(node)


def _build_points(article: Node, doc: SourceDoc, body: str | None = None) -> None:
    """Materialise пункт-level children so long articles can be indexed finely.

    Points are always created, even for short articles — the chunker decides
    whether to cut at this level, and it needs the structure to be there to make
    that decision.
    """
    txt = article.texts[doc.lang]
    parts = S.split_points(body if body is not None else txt.body)
    if len(parts) == 1 and parts[0][0] is None:
        return
    for i, (number, body) in enumerate(parts, start=1):
        if number is None:
            # Text before the first numbered point (a lead-in sentence) stays on
            # the article itself rather than becoming a phantom point.
            continue
        nid = point_id(article.id, number)
        node = Node(id=nid, act_code=article.act_code, kind=NodeKind.POINT,
                    number=number, ordinal=i, parent_id=article.id)
        node.texts[doc.lang] = NodeText(lang=doc.lang, heading="", body=body,
                                        valid_from=txt.valid_from, valid_to=txt.valid_to)
        article.children.append(node)


def parse_act_source(raw: str) -> tuple[Act, Node, SourceDoc]:
    doc = parse_source_file(raw)
    act, root = parse_lines(doc, doc.body.splitlines())
    return act, root, doc


def parse_adilet(html: str, meta: dict) -> tuple[Act, Node, SourceDoc]:
    lines = adilet_html_to_lines(html)
    body = "\n".join(lines)
    doc = SourceDoc(
        act_code=meta["code"],
        kind=ActKind(meta.get("kind", "code")),
        official_id=meta.get("official_id", ""),
        lang=Lang(meta.get("lang") or S.detect_lang(body).value),
        title=meta.get("title", ""),
        body=body,
        valid_from=meta.get("valid_from") or date(1900, 1, 1),
        valid_to=meta.get("valid_to") or OPEN_ENDED,
        url=meta.get("url"),
    )
    act, root = parse_lines(doc, lines)
    return act, root, doc

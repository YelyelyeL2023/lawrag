"""Source text -> canonical tree.

Two adapters feed one line-based parser: a plain-text corpus format (used by the
sample data and by anything you export yourself) and an adilet HTML adapter. The
HTML adapter's only job is to produce clean lines; all structural understanding
lives in :func:`parse_lines`, so adding a third source format never touches the
part that decides what an article is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import yaml

from ..ids import article_id, point_id, subpoint_id
from ..logging_setup import get_logger
from ..models import Act, ActKind, Lang, Node, NodeKind, NodeText, OPEN_ENDED
from . import dating as D
from . import structure as S

log = get_logger(__name__)

# Stored when nothing at all is known about when a text started to apply. It is
# not a date in the law; the console counts rows sitting on it.
UNDATED = date(1900, 1, 1)


@dataclass(slots=True)
class SourceDoc:
    """One act, in one language, as one snapshot.

    ``valid_from`` is the *snapshot* date: the day the whole document is known
    to be in this state. Individual nodes may have started to apply earlier —
    see ``derive_dates``.
    """

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
    # True when each article's start date is read from its own amendment
    # footnotes (adilet snapshots). False when the source declares one redaction
    # date for the whole document (``valid_from`` in .act.txt front matter),
    # which then applies to every node.
    derive_dates: bool = False
    # Filled in by the parser: how many node texts got a date from a footnote,
    # how many of those were resolved relative to publication, and how many
    # fell back to the adoption date.
    dating: dict = field(default_factory=dict)

    @property
    def base_from(self) -> date:
        """Start date for a node with no amendment footnote of its own."""
        if not self.derive_dates:
            return self.valid_from
        return self.adopted_on or UNDATED


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
        # A declared redaction date applies to the whole file. Without one, the
        # file is treated like an adilet page: dated per article from its
        # footnotes, falling back to adopted_on.
        valid_from=as_date("valid_from") or as_date("adopted_on") or UNDATED,
        valid_to=as_date("valid_to") or OPEN_ENDED,
        adopted_on=as_date("adopted_on"),
        url=meta.get("url"),
        derive_dates=meta.get("valid_from") is None,
    )


# ---------------------------------------------------------------------------
# adilet HTML adapter
# ---------------------------------------------------------------------------

_BLOCK_TAGS = ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol",
               "table", "article", "section", "blockquote", "pre")


def _is_red(tag) -> bool:
    return (tag.get("color") or "").strip().lower() in ("#ff0000", "red")


def adilet_html_to_lines(html: str) -> list[str]:
    """Reduce an adilet document page to clean text lines.

    Line breaks come only from block structure — paragraphs, headings, ``<br>``
    and table rows. adilet wraps every cited article and every amending law's
    number in an inline ``<a>``, and splitting on those turned one sentence
    into five lines: "вносящих изменения в" / "Уголовный кодекс" / "Республики
    Казахстан," — which broke point detection, left footnote fragments such as
    "№ 158-VII" in the article body, and made any quote spanning a link
    unverifiable.

    Tables are flattened with tab separators so the table detector downstream
    can still recognise them. Consecutive red editorial blocks ("Примечание
    ИЗПИ!" followed by its text) are joined into one line so the structural
    parser can recognise and set aside the whole note.
    """
    from bs4 import BeautifulSoup, NavigableString

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

    for row in root.find_all("tr"):
        cells = [" ".join(c.get_text(" ", strip=True).split()) for c in row.find_all(["td", "th"])]
        if any(cells):
            row.replace_with("\n" + "\t".join(cells) + "\n")

    # Editorial notes: gather each run of red <font> siblings separated only by
    # <br> or whitespace, and replace the run with a single line.
    for font in root.find_all("font"):
        if font.parent is None or not _is_red(font):
            continue
        parts = [font.get_text(" ", strip=True)]
        run = []
        sib = font.next_sibling
        while sib is not None:
            if isinstance(sib, NavigableString) and not sib.strip():
                run.append(sib)
            elif getattr(sib, "name", None) == "br":
                run.append(sib)
            elif getattr(sib, "name", None) == "font" and _is_red(sib):
                parts.append(sib.get_text(" ", strip=True))
                run.append(sib)
            else:
                break
            sib = sib.next_sibling
        # Trailing breaks/whitespace after the last red block are not part of it.
        while run and getattr(run[-1], "name", None) != "font":
            run.pop()
        for extra in run:
            extra.extract()
        text = " ".join(p for p in parts if p)
        if text and not S.is_editorial_line(text):
            text = "[ИЗПИ] " + text
        font.replace_with("\n" + text + "\n")

    for br in root.find_all("br"):
        # Inside a bold article title a <br> is typography, not a new line of
        # law. Section headings (<h3>) are left alone: adilet stacks "РАЗДЕЛ",
        # "Глава" and "Параграф" in one <h3> separated by <br>, and each has to
        # stay on its own line to be recognised.
        in_heading = br.find_parent("b") is not None
        br.replace_with(" " if in_heading else "\n")
    for tag in root.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.append("\n")

    text = root.get_text("")
    lines: list[str] = []
    for raw_line in text.split("\n"):
        if "\t" in raw_line:
            cells = [" ".join(c.replace("\xa0", " ").split()) for c in raw_line.split("\t")]
            line = "\t".join(cells).strip("\t")
        else:
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
                                    valid_from=doc.base_from, valid_to=doc.valid_to)

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
        if doc.derive_dates:
            _date_article(txt, provenance, doc)
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
        text, provenance = S.strip_amendment_notes("\n".join(container_buffer))
        node = stack[-1]
        if provenance and doc.lang in node.texts:
            # "Сноска. Заголовок главы 8 в редакции ..." and the publisher's
            # notes on the act as a whole are provenance, not law.
            _attach_provenance(node, provenance)
        if text:
            existing = node.texts.get(doc.lang)
            if existing is None:
                node.texts[doc.lang] = NodeText(
                    lang=doc.lang, heading="", body=text,
                    valid_from=doc.base_from, valid_to=doc.valid_to)
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
                                            valid_from=doc.base_from, valid_to=doc.valid_to)
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
                                        valid_from=doc.base_from, valid_to=doc.valid_to)
        parent.children.append(node)
        stack.append(node)

    flush_article()
    flush_container()
    n_articles = sum(1 for _ in root.articles())
    log.info("parsed %s [%s]: %d articles, %d lines attached to containers "
             "(preamble / transitional provisions)",
             doc.act_code, doc.lang.value, n_articles, orphan_lines)
    if doc.derive_dates:
        d = doc.dating
        log.info("dated %s [%s] as of %s: %d articles from footnotes (%d approximate, "
                 "resolved from the law's signing date), %d from adoption %s, %d undated",
                 doc.act_code, doc.lang.value, doc.valid_from, d.get("footnote", 0),
                 d.get("approximate", 0), d.get("adoption", 0), doc.adopted_on,
                 d.get("undated", 0))
        if d.get("undated"):
            log.warning("%s [%s]: no adoption date known, so %d unamended articles are "
                        "stored as undated (%s). Set adopted_on in documents.json.",
                        doc.act_code, doc.lang.value, d["undated"], UNDATED)
    if n_articles == 0:
        log.warning(
            "%s [%s] produced zero articles — check that the heading dialect "
            "matches the document language", doc.act_code, doc.lang.value
        )
    return act, root


def _date_article(txt: NodeText, provenance: list[str], doc: "SourceDoc") -> None:
    """Start the article's text on the latest commencement its footnotes cite.

    Editorial notes are excluded: they announce amendments that have *not*
    been applied to the text on the page. A repeal stub carries its law in the
    heading ("Статья 12. Исключена Законом РК от ..."), so the heading is read
    too.
    """
    footnotes = [line for line in provenance if S.is_footnote_line(line)]
    if S.REPEALED_RE.search(txt.heading):
        footnotes.append(txt.heading)
    stats = doc.dating
    # For an adilet snapshot nothing on the page can have commenced after the
    # day it was fetched. A hand-written source with no declared date has no
    # such bound.
    ceiling = doc.valid_from if doc.valid_from > doc.base_from else None
    when, approximate = D.latest_effective(footnotes, not_after=ceiling)
    if when is None or when <= doc.base_from:
        key = "adoption" if doc.adopted_on else "undated"
        stats[key] = stats.get(key, 0) + 1
        return
    txt.valid_from = when
    stats["footnote"] = stats.get("footnote", 0) + 1
    if approximate:
        stats["approximate"] = stats.get("approximate", 0) + 1


def _attach_note(article: Node, note: str, doc: "SourceDoc") -> None:
    """A substantive Примечание / Ескерту becomes its own retrievable node."""
    existing = sum(1 for c in article.children if c.kind is NodeKind.NOTE
                   and c.id.endswith(tuple(f"/note{i}" for i in range(1, 20))))
    nid = f"{article.id}/note{existing + 1}"
    node = Node(id=nid, act_code=article.act_code, kind=NodeKind.NOTE,
                number=str(existing + 1), ordinal=900 + existing,
                parent_id=article.id)
    parent = article.texts[doc.lang]
    node.texts[doc.lang] = NodeText(lang=doc.lang, heading="", body=note,
                                    valid_from=parent.valid_from, valid_to=parent.valid_to)
    article.children.append(node)


def _attach_provenance(article: Node, notes: list[str]) -> None:
    note_id = f"{article.id}/amend"
    lang = next(iter(article.texts))
    existing = next((c for c in article.children if c.id == note_id), None)
    if existing is not None and lang in existing.texts:
        existing.texts[lang].body += "\n" + "\n".join(notes)
        return
    node = Node(id=note_id, act_code=article.act_code, kind=NodeKind.NOTE,
                number=None, ordinal=999, parent_id=article.id)
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
    by_id: dict[str, Node] = {}
    for i, (number, body) in enumerate(parts, start=1):
        if number is None:
            # Text before the first numbered point (a lead-in sentence) stays on
            # the article itself rather than becoming a phantom point.
            continue
        nid = point_id(article.id, number)
        if nid in by_id:
            # The same number twice in one article. Two nodes with one id become
            # two chunks with one chunk id, and the second silently overwrites
            # the first in the index — the text of the real point disappears.
            # Keep both texts on the one node and say so.
            log.warning("%s [%s]: point %s occurs twice; texts merged",
                        article.id, doc.lang.value, number)
            prev = by_id[nid].texts[doc.lang]
            prev.body = f"{prev.body}\n{body}"
            continue
        node = Node(id=nid, act_code=article.act_code, kind=NodeKind.POINT,
                    number=number, ordinal=i, parent_id=article.id)
        node.texts[doc.lang] = NodeText(lang=doc.lang, heading="", body=body,
                                        valid_from=txt.valid_from, valid_to=txt.valid_to)
        article.children.append(node)
        by_id[nid] = node


def parse_act_source(raw: str) -> tuple[Act, Node, SourceDoc]:
    doc = parse_source_file(raw)
    act, root = parse_lines(doc, doc.body.splitlines())
    return act, root, doc


def parse_adilet(html: str, meta: dict) -> tuple[Act, Node, SourceDoc]:
    """Parse one adilet page.

    Dates, in order of authority:

    * ``valid_from`` in documents.json — an operator's explicit statement that
      the whole document is one redaction from that day. Disables per-article
      dating.
    * otherwise the snapshot date is ``as_of`` (a point-in-time fetch) or the
      day it was fetched, and each article starts on the commencement date of
      its latest amendment footnote, or on the act's adoption date (from
      documents.json ``adopted_on``, else the page header).
    """
    lines = adilet_html_to_lines(html)
    body = "\n".join(lines)
    header = D.adilet_header(html)

    def as_date(value) -> date | None:
        if value in (None, ""):
            return None
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10])

    declared = as_date(meta.get("valid_from"))
    snapshot = declared or as_date(meta.get("as_of")) or as_date(meta.get("fetched_at"))
    if snapshot is None:
        log.warning("%s [%s]: documents.json has no valid_from, as_of or fetched_at; "
                    "the snapshot cannot be dated", meta.get("code"), meta.get("lang"))
    if header.status and not header.status.lower().startswith(("обновл", "жаңарт", "действ")):
        log.warning("%s [%s]: adilet marks this document as %r", meta.get("code"),
                    meta.get("lang"), header.status)
    doc = SourceDoc(
        act_code=meta["code"],
        kind=ActKind(meta.get("kind", "code")),
        official_id=meta.get("official_id", ""),
        lang=Lang(meta.get("lang") or S.detect_lang(body).value),
        title=meta.get("title", ""),
        body=body,
        valid_from=snapshot or UNDATED,
        valid_to=as_date(meta.get("valid_to")) or OPEN_ENDED,
        adopted_on=as_date(meta.get("adopted_on")) or header.adopted_on,
        url=meta.get("url"),
        derive_dates=declared is None and snapshot is not None,
    )
    act, root = parse_lines(doc, lines)
    return act, root, doc

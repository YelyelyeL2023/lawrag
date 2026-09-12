"""The canonical legal document model.

Codes are parsed into a tree, not a blob. Everything downstream — chunking,
citation, version diffing, bilingual alignment — keys off this structure, so the
parser is the single place where "what is an article" is decided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Iterator


class Lang(str, Enum):
    RU = "ru"
    KK = "kk"


class NodeKind(str, Enum):
    ACT = "act"           # the code / law as a whole
    SECTION = "section"   # Раздел / Бөлім
    CHAPTER = "chapter"   # Глава / Тарау
    PARAGRAPH = "paragraph"  # Параграф (occasional level between chapter and article)
    ARTICLE = "article"   # Статья / Бап   <- the atomic legal unit
    POINT = "point"       # пункт / тармақ
    SUBPOINT = "subpoint" # подпункт / тармақша
    NOTE = "note"         # Примечание — legally operative, always kept with its article


class ActKind(str, Enum):
    CONSTITUTION = "constitution"
    CONSTITUTIONAL_LAW = "constitutional_law"
    CODE = "code"
    LAW = "law"
    PRESIDENTIAL_DECREE = "presidential_decree"
    GOVERNMENT_RESOLUTION = "government_resolution"
    MINISTERIAL_ORDER = "ministerial_order"


# Hierarchy of legal force, per the Law "О правовых актах" (No. 480-V of 6 April
# 2016). Lower rank number wins a conflict. Used both to order retrieval results
# and to let the answer layer flag when a lower act appears to contradict a
# higher one.
NORM_RANK: dict[ActKind, int] = {
    ActKind.CONSTITUTION: 1,
    ActKind.CONSTITUTIONAL_LAW: 2,
    ActKind.CODE: 3,
    ActKind.LAW: 4,
    ActKind.PRESIDENTIAL_DECREE: 5,
    ActKind.GOVERNMENT_RESOLUTION: 6,
    ActKind.MINISTERIAL_ORDER: 7,
}

# Open-ended validity is stored as this sentinel rather than NULL so that range
# comparisons in SQL stay simple and index-friendly.
OPEN_ENDED = date(9999, 12, 31)


@dataclass(slots=True)
class Act:
    code: str                      # short stable slug, e.g. "koap"
    kind: ActKind
    official_id: str               # e.g. "K1400000235" — the adilet document id
    title: dict[str, str]          # lang -> title
    adopted_on: date | None = None
    url: str | None = None

    @property
    def norm_rank(self) -> int:
        return NORM_RANK[self.kind]


@dataclass(slots=True)
class NodeText:
    """The text of one structural node, in one language, in one redaction."""

    lang: Lang
    heading: str
    body: str
    valid_from: date
    valid_to: date = OPEN_ENDED

    @property
    def is_current(self) -> bool:
        return self.valid_to == OPEN_ENDED


@dataclass(slots=True)
class Node:
    id: str
    act_code: str
    kind: NodeKind
    number: str | None                     # "278", "2", "1-1", None for act root
    ordinal: int                           # position among siblings, for stable ordering
    parent_id: str | None = None
    texts: dict[Lang, NodeText] = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)

    def walk(self) -> Iterator["Node"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def articles(self) -> Iterator["Node"]:
        for node in self.walk():
            if node.kind is NodeKind.ARTICLE:
                yield node

    def text(self, lang: Lang) -> NodeText | None:
        return self.texts.get(lang)


@dataclass(slots=True)
class XRef:
    """A cross-reference extracted from article text.

    ``dst_node_id`` is None when the reference could not be resolved against the
    corpus (a reference to an act we have not ingested, or a malformed cite).
    Unresolved references are kept deliberately — their count is a corpus health
    metric, and hiding them would make the graph look complete when it is not.
    """

    src_node_id: str
    raw: str
    dst_act_code: str | None
    dst_article: str | None
    dst_node_id: str | None
    kind: str = "reference"     # reference | definition | amendment


@dataclass(slots=True)
class Chunk:
    """One indexed unit.

    Three distinct texts live here on purpose:

    ``display_text``  clean legal text, quoted verbatim to the user
    ``embed_text``    display text + hierarchy header + generated context line;
                      what the embedding and BM25 indexes actually see
    ``context_sentence``  one generated line situating the chunk in its act
    """

    id: str
    node_id: str
    article_node_id: str
    act_code: str
    lang: Lang
    valid_from: date
    valid_to: date
    display_text: str
    embed_text: str
    hierarchy_path: str
    context_sentence: str = ""
    lead_in: str = ""
    repealed: bool = False
    heading: str = ""
    token_estimate: int = 0
    aligned_chunk_id: str | None = None
    release_id: str = ""


@dataclass(slots=True)
class CorpusRelease:
    id: str
    source: str
    fetched_at: datetime
    manifest_sha256: str
    note: str = ""

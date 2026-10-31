"""BM25 over the same chunk corpus: the lexical baseline for the embedders.

The production retriever's lexical branch is Postgres full-text search, which
needs a database. This is the textbook Okapi BM25 in numpy/scipy, so the
lexical baseline is measured with the same harness, on the same queries and
the same chunks, as the dense models — the comparison is then about the
*model*, not the plumbing.

Russian and Kazakh are both heavily inflected (Kazakh is agglutinative), so a
whole-word index would treat "штраф", "штрафа" and "штрафом" as unrelated.
Tokens are cut to a fixed prefix instead — crude, language-agnostic stemming
that needs no per-language dictionary on an air-gapped machine.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

import numpy as np

BM25_MODEL = "bm25"
_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str, prefix: int = 6) -> list[str]:
    """Lowercased word tokens, each cut to its first ``prefix`` characters
    (``prefix`` 0 keeps whole words)."""
    tokens = _TOKEN.findall(text.lower())
    return [t[:prefix] for t in tokens] if prefix else tokens


class BM25:
    """Okapi BM25 with the usual defaults (k1 = 1.2, b = 0.75).

    ``fit`` builds the document-term matrix once with every BM25 term weight
    precomputed, so scoring a batch of queries is one sparse matrix product.
    ``reweight`` changes k1 and b without re-tokenising the corpus: the raw
    term counts are kept, which is what makes a grid search over k1 x b cheap.
    """

    name = BM25_MODEL

    def __init__(self, k1: float = 1.2, b: float = 0.75, prefix: int = 6):
        self.k1, self.b, self.prefix = k1, b, prefix
        self.vocab: dict[str, int] = {}
        self._counts = None    # (n_docs x n_terms) CSR raw term frequencies
        self._weights = None   # (n_terms x n_docs) CSR, BM25 weight per (term, doc)

    def params(self) -> dict:
        return {"k1": self.k1, "b": self.b, "prefix": self.prefix}

    def fit(self, texts: Sequence[str]) -> "BM25":
        from scipy.sparse import csr_matrix

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for d, text in enumerate(texts):
            for term, tf in Counter(tokenize(text, self.prefix)).items():
                rows.append(d)
                cols.append(self.vocab.setdefault(term, len(self.vocab)))
                vals.append(tf)
        self._counts = csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)),
                                  shape=(len(texts), len(self.vocab)))
        return self.reweight(self.k1, self.b)

    def reweight(self, k1: float, b: float) -> "BM25":
        tf = self._counts.copy()
        self.k1, self.b = k1, b
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = float(doc_len.mean()) if n_docs else 1.0
        df = np.bincount(tf.indices, minlength=len(self.vocab))
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # Term-frequency saturation with document-length normalisation, then
        # idf: the standard Okapi weight, computed per nonzero entry.
        row_of = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        norm = k1 * (1.0 - b + b * doc_len[row_of] / avg_len)
        tf.data = (tf.data * (k1 + 1.0) / (tf.data + norm) * idf[tf.indices]).astype(np.float32)
        self._weights = tf.T.tocsr()
        return self

    def query_matrix(self, queries: Sequence[str]):
        """(n_queries x n_terms) binary query-term matrix: a term counts once."""
        from scipy.sparse import csr_matrix

        rows: list[int] = []
        cols: list[int] = []
        for q, text in enumerate(queries):
            for term in set(tokenize(text, self.prefix)):
                j = self.vocab.get(term)
                if j is not None:
                    rows.append(q)
                    cols.append(j)
        return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(len(queries), len(self.vocab)))

    def score(self, queries: Sequence[str]) -> np.ndarray:
        """(n_queries x n_docs) BM25 scores."""
        return np.asarray((self.query_matrix(queries) @ self._weights).todense(),
                          dtype=np.float32)

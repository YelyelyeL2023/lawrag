# SYNTHETIC TEST FIXTURE — NOT REAL LAW

Every file in this directory is **invented**. The article numbers, texts, fines
and deadlines here do not reproduce the Codes of the Republic of Kazakhstan and
must never be used to answer a real legal or compliance question.

The fixture exists so that `lawrag ingest` → `index` → `ask` runs end-to-end on a
clean clone, and so the parser, chunker, cross-reference extractor, versioning
and bilingual alignment all have something to bite on. It deliberately contains:

- a three-level hierarchy (раздел → глава → статья) in both language dialects
- a definitions article, to exercise glossary extraction
- articles that cite other acts, to exercise the citation graph
- one article with two redactions, to exercise point-in-time retrieval
- one repealed article, to exercise repeal handling
- a rate table that must survive chunking intact

It lives under `tests/` and nowhere else. It used to sit in `data/sample/` next
to a real adilet snapshot, where both were ingested together under the same act
codes and the invented "статья 278 КоАП" replaced the real one as current law.
The real corpus is in `data/adilet/`; never copy these files into it.

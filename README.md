# lawrag

Retrieval-augmented lookup over the Constitution and Codes of the Republic of
Kazakhstan, for compliance staff without legal training, on a fully air-gapped
machine.

Everything runs locally. The corpus arrives as signed offline snapshot bundles;
no query-path code makes an outbound network call.

> **Two corpora, kept apart.** `data/adilet/` is a real snapshot of adilet.zan.kz
> (fetched 2026-09-08). `tests/fixtures/synthetic/` is an **invented** fixture the
> test suite runs on. They used to share `data/sample/`, where the fixture's
> "статья 278 КоАП" closed the real article 278 and answered as current law;
> `ingest` now refuses a directory that holds two different documents under one
> act code.

---

## Quick start

Docker is the only prerequisite. There is nothing to install on the host — no
Python, no virtualenv, no `pip install -e`.

```bash
docker/scripts/bootstrap.sh          # Windows: docker\scripts\bootstrap.ps1
```

or the same thing by hand. Every `docker compose` command in this README runs
from `docker/`, where all Docker specifics live — the compose files, the
Dockerfile and its `.env`. Nothing outside that directory knows it is running
in a container.

```bash
cd docker
docker compose build app
docker compose up -d --wait db                                   # Postgres 16 + pgvector
docker compose run --rm cli ingest data/adilet --reset --no-verify
docker compose run --rm cli stats
docker compose up -d app                                         # API + console on :8000
```

```bash
docker compose run --rm cli search "В какой срок нужно встать на регистрационный учет?"
docker compose run --rm cli search "Какой штраф за просрочку?" --as-of 2025-03-15   # point-in-time
docker compose run --rm cli calibrate                            # relevance floor for this model
docker compose run --rm cli search "Тіркеу есебіне қою мерзімі қандай?"             # Kazakh
docker compose run --rm cli ask    "Продлевается ли срок для малого бизнеса?"
docker compose run --rm tests                                    # the test suite
```

`cli` is the same image the API runs, against the same config and database, so a
question answered from the shell and the same question answered in the browser
go through identical code.

---

## Containers

| service | profile | what it is |
| --- | --- | --- |
| `db` | — | Postgres 16 + pgvector, published on `127.0.0.1:5433` |
| `app` | — | the HTTP API and the browser console, on `127.0.0.1:8000` |
| `cli` | `cli` | one-shot `lawrag` commands: ingest, search, ask, stats, eval |
| `tests` | `tests` | pytest, against a live `db` |
| `llm` | `llm` | llama-server with the local generator |
| `download-models` | `download` | fetches the weights — connected side only |

Two image targets, because the weights stack dwarfs everything else: `runtime`
(~390 MB — CLI, API, ingest, retrieval, a llama.cpp client) and `models`
(~10 GB, adding torch + FlagEmbedding for bge-m3 and the reranker).
`LAWRAG_TARGET` in `docker/.env` picks one; it defaults to `models`, because the
shipped config asks for bge-m3.

What stays outside the image is deliberate: the corpus, the weights and the
config are mounted. An image with a snapshot of the law baked in would go stale
with no symptom, which is the failure this project works hardest to prevent.

**GPU.** The base compose file runs on the CPU so a machine without a card still
works; a device reservation that cannot be satisfied is a hard startup failure,
so it lives in an overlay instead of a flag:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Set `COMPOSE_FILE` in `docker/.env` to make that the default. Needs the NVIDIA
Container Toolkit (Docker Desktop provides it on WSL2 with a current driver).

**Air gap.** Move images, not wheels. On the connected side:

```bash
docker compose build app
docker save lawrag:models pgvector/pgvector:pg16 ghcr.io/ggml-org/llama.cpp:server-cuda -o lawrag-images.tar
```

Carry that tarball and the `MODEL_DIR` tree across; `docker load -i
lawrag-images.tar` on the target, then `docker compose up -d`. No pip index, no
wheelhouse to assemble, and no host interpreter whose version has to match.

**Configuration.** `config/default.yaml` is where a setting is reasoned about;
anything that differs between deployments is an environment variable that
overrides it, spelled `LAWRAG_<SECTION>_<FIELD>` — `LAWRAG_DB_DSN`,
`LAWRAG_EMBEDDING_DEVICE`, `LAWRAG_LLM_BASE_URL`. The YAML carries
local-development defaults and no container hostnames or paths;
`docker/docker-compose.yml` sets exactly those variables for the containers.

`config/local.yaml` is the same pipeline on the placeholder models, for running
`lawrag` from a host interpreter against the compose database:
`LAWRAG_CONFIG=config/local.yaml lawrag stats`. The test suite pins the same
placeholders. Both name the database as `127.0.0.1`, not `localhost`: compose
publishes the port on IPv4 loopback only, and a Windows host resolves
`localhost` to `::1` first, so each connection would wait out a timeout before
falling back. `config/host-gpu.yaml` is the same again with the fine-tuned
embedder on the host's card.

---

## Console

The `app` service serves a browser console at `http://127.0.0.1:8000/`, over the
same endpoints the CLI uses:

| view | what it is for |
| --- | --- |
| **Вопрос** | ask a question; the answer arrives with its citations, the verification verdict, and every source that went into the packet |
| **Поиск** | retrieval only — per-branch scores (dense / lexical / trigram), the fused rank, and the exact context the model would have been handed |
| **Корпус** | acts and articles as they stood on a chosen date, every redaction of an article, and its citation graph in both directions |
| **Импорт** | upload a snapshot and watch `ingest_directory` run, with the pipeline's own log |
| **Состояние** | corpus counts, unresolved references, query volume, latency percentiles, verification outcomes, and the audit log |

It is static HTML, CSS and JavaScript with no build step and no external
requests — no CDN, no web fonts, no npm — because the machine it runs on has no
network, and a console that degrades when a font server is unreachable is not a
console. Upload is JSON-with-text rather than multipart so that no extra wheel
(`python-multipart`) has to be carried across the air gap.

Every node id the console shows is a link into **Корпус** — a search hit, a
packet source, a citation in an answer, a chunk in the query log — which opens
that article in that language on that redaction date. Search results
highlight the query's word stems, so a Kazakh case ending or a Russian
declension still lights up, and hits from one article say which point matched.
Import history rows open the job's log.

The extra endpoints live in `api/console.py` under `/api`, and every one of them
is read-only except `POST /api/ingest`. `api/main.py` is unchanged in substance:
delete `api/static/` and the API is exactly what it was.

### Access

| setting | effect |
| --- | --- |
| `LAWRAG_API_TOKEN` | when set, every request except `/health` and the console's static files needs `Authorization: Bearer <token>`; the console asks for it once per tab |
| neither set | `POST /api/ingest` (which can drop the schema) is accepted only from a loopback client; everything else is open |
| `LAWRAG_ALLOW_REMOTE_WRITES=1` | lifts the loopback rule without a token. Compose sets it, because the port is published on the host's `127.0.0.1` but requests reach the container from the Docker bridge |

`lawrag serve --host 0.0.0.0` without a token prints a warning.

Import history is written to `data/uploads/<job>/job.json`, next to the staged
files, so it survives a restart (not the database: an import with `reset` would
delete the record of itself). A job that was running when the server stopped is
shown as interrupted.

Degraded state is rendered rather than hidden. Placeholder backends raise a
banner on every view, an unreachable database is reported as such instead of an
empty page, and a superseded or repealed article is never shown with the same
badge as one in force.

There are also **placeholder models** — a deterministic hash embedder and an echo
LLM — so the pipeline is runnable and testable on a machine with no weights on
disk (`embedding.backend: hash`, `llm.backend: echo`). Retrieval is fully
functional in that state; `ask` abstains with `no_model_loaded` rather than
pretending, and `GET /health` reports `production_ready: false`. The console
raises a banner on every view rather than letting placeholder output pass for an
answer.

---

## Why it is built this way

Four properties of this corpus drive nearly every design decision.

**It is bilingual, and the two languages are not translations of each other for
parsing purposes.** Russian marks structure as `Статья 278` / `Глава 14`; Kazakh
attaches the number as an ordinal prefix, `278-бап` / `14-тарау`. A parser
written against Russian alone yields one flat blob per Kazakh act, which then
looks like a retrieval-quality problem three layers downstream. Both dialects
are handled in `ingest/structure.py`, and both languages are parsed onto the
*same* structural skeleton — so aligning them is a primary-key lookup, not a
similarity match.

**It changes constantly, and "what was the rule then" is most of the hard
questions.** Storage is bitemporal: every text row carries the range over which
the norm was in force and the corpus snapshot it came from. Superseded text is
closed, never deleted — including text a new snapshot simply *drops*, which is
how a repeal actually looks and which nothing in the per-article update path can
see. The citation graph and the glossary are versioned too, so a question asked
as-of 2024 is not hydrated through the 2026 graph. Overlapping validity ranges
are rejected by the database (`EXCLUDE USING gist`) rather than prevented by
ingest discipline: two redactions both reading as current law is a wrong answer
with no symptom. The as-of date is a `WHERE` clause in every retrieval branch —
never a ranking signal.

**The users cannot spot a bad citation.** So a citation is not trusted because
it looks right: `verify/citations.py` checks mechanically that each cited
article exists, was actually in the retrieved packet, was in force on the as-of
date, and that the quote appears verbatim. Any failure downgrades the answer to
"here are the relevant provisions" instead of letting a wrong conclusion through
with a plausible footnote.

**There is no internet.** The connected-side fetcher writes a bundle with a
SHA-256 manifest; import verifies every file before a byte is parsed. The
release id is derived from the manifest digest, so every indexed chunk and every
logged answer traces back to an exact, reproducible snapshot.

---

## Pipeline

```
adilet / reference-bank export        [connected side]
        │  lawrag fetch → lawrag pack
        ▼
   signed bundle  ═══ air gap ═══▶  lawrag verify-bundle → lawrag ingest
                                            │
        ┌───────────────────────────────────┴────────────────────────┐
        ▼                                                            ▼
  structural parse                                          cross-references
  (RU + KK dialects)                                        + defined terms
        │                                                            │
        ▼                                                            ▼
  structural chunking  ──▶  embed_text (hierarchy + lead-in)   citation graph
        │                                                       + glossary
        ▼                                                            │
   Postgres: pgvector + tsvector + trigram  ◀────────────────────────┘
        │
        ▼
  query analysis (lang · as-of · explicit citations)
        │
        ▼
  hybrid retrieve → RRF → relevance floor → rerank → dedupe languages
        │
        ▼
  hydrate: parent article · referenced norms · definitions   (1 hop)
        │
        ▼
  context packet  ──▶  local LLM  ──▶  citation verification  ──▶  answer + audit log
```

---

## Chunking

Three questions, answered separately — collapsing them is the usual cause of
"the retriever found the right article but the answer was still wrong".

| question | answered by | where |
|---|---|---|
| What is the atomic unit? | a structural node, never a token window | `chunking/chunker.py` |
| What text gets embedded? | hierarchy header + context line + lead-in + body | `build_embed_text` |
| What text reaches the model? | assembled at query time from the graph | `retrieve/packet.py` |

Rules the chunker follows:

- Article-level by default; **пункт**-level only when the article exceeds
  ~1000 tokens. Every boundary coincides with one the legislator drew.
- Short articles are never merged with their neighbours to hit a size target —
  that would destroy pinpoint citation, which is the whole product.
- Rate tables and enumerated lists of grounds are never split. An oversized
  correct chunk beats two mutilated ones.
- **Point chunks carry the article's lead-in.** In KZ codes the clause before
  point 1 states the conduct; the numbered points attach sanctions to it. A
  point indexed without it describes a penalty for nothing. It is stored as its
  own column so that jobs rebuilding `embed_text` cannot quietly drop it.
- **Repeal stubs stay indexed but are flagged.** People do ask about repealed
  articles. An article whose whole body is «Исключена Законом РК от…» is marked
  `repealed`, and the packet says so out loud instead of presenting an empty
  article as an ordinary norm.
- Substantive `Примечание` / `Ескерту` become their own retrievable nodes.
  Amendment provenance (`Сноска. Статья 278 в редакции Закона РК от…`) is
  stripped from indexed text and kept as metadata — otherwise every article in
  a heavily amended code shares the same high-frequency boilerplate.

`docker compose run --rm cli context` runs the optional contextual-retrieval batch job: a small local
model writes one sentence per chunk situating it in its act, cached by content
hash. Largest retrieval gain per unit of effort; costs nothing at query time.

---

## Retrieval

Lexical matters more here than in most RAG systems — legal queries carry exact
article numbers and fixed statutory terminology, and a dense model will happily
rank article 277 next to 278. Three branches are fused with reciprocal rank:

- **dense** — pgvector cosine over `embed_text`
- **lexical** — Postgres FTS, `russian` stemmer for RU, `simple` + trigram
  fallback for KK
- **citation** — direct lookup when the query names an article, weighted above
  the other two because an explicit citation is an instruction, not a hint

Then: a **relevance floor** (absolute, not relative — rank fusion always returns
*something*), reranking, language dedupe, and one-hop hydration.

Two known weak points, stated rather than hidden:

- Kazakh has no Postgres text-search dictionary. `simple` plus a
  `word_similarity` trigram fallback is a workaround for an agglutinative
  language, and only a partial one. Wiring in a morphological analyser
  (`apertium-kaz`) is the single highest-value corpus improvement available.
- The relevance floor has three thresholds on three different scales:
  `min_dense_score` is a cosine and **must be re-tuned** when you switch from
  the hash placeholder to BGE-M3; `min_lexical_score` is a `ts_rank_cd` value
  that depends on the corpus, not the model; `min_trigram_score` is a
  `word_similarity` on 0..1. `lawrag calibrate` measures them: it runs the
  question set with the floor off and proposes each threshold in the gap between
  the weakest answerable question and the strongest out-of-corpus one — or says
  that a signal cannot separate them.

### Redaction dates

adilet serves the current text, and records every change in a footnote under the
article it touched ("Сноска. Статья 8 с изменениями, внесенными законами РК от
05.11.2022 № 158-VII (вводится в действие с 01.01.2023); …"). `ingest/dating.py`
reads those: an article starts on the commencement date of its latest
amendment, and an unamended article on the act's adoption date from the page
header. Commencement stated relative to publication ("по истечении десяти
календарных дней…") is resolved from the signing date and logged as approximate.
Publisher's notes ("Примечание ИЗПИ!") announce amendments not yet applied; they
are set aside, never attached as an operative note, and never used as a date.

This dates the text in the corpus; it does not recreate earlier wording. A
question about a date before an article's last amendment finds no text for it,
and the system says so (`cited_text_not_in_force_on_date`) rather than quoting
today's wording as if it had applied then. For history, fetch a snapshot per date
(`lawrag fetch <dir> --as-of YYYY-MM-DD`) and ingest each; a page that cites an
amending law signed after the requested date is rejected, since that means the
server ignored the date.

---

## Models

Hardware assumed: 8 GB VRAM, 32 GB RAM, high-core CPU.

The backends in `config/default.yaml` are already the production ones — bge-m3,
the bge reranker, and llama.cpp. What they need is the weights, in `MODEL_DIR`
on the host, mounted at `/models` in every container:

```bash
docker/scripts/download_models.sh        # connected side; Windows: docker\scripts\download_models.ps1
```

That lands `bge-m3/` and `bge-reranker-v2-m3/` under `MODEL_DIR`; put the
`.gguf` generator file beside them. The download runs in the same `models` image
that will later load the weights, so what is fetched is what is loaded.

```bash
docker/scripts/serve_llm.sh           # profile B (the default)
docker/scripts/serve_llm.sh A         # Qwen3.5-9B Q4_K_M, full GPU offload, ~50 tok/s
docker/scripts/serve_llm.sh B         # Qwen3.6-35B-A3B, MoE experts in RAM, ~30 tok/s
docker/scripts/serve_llm.sh B --cpu   # no card: CPU image, all layers on the CPU
```

Both scripts are wrappers around `docker compose --profile llm up -d llm`: the
profile differs only in `LLM_MODEL_FILE` and `LLM_N_CPU_MOE`, so switching
models is a variable, not a rebuild. Nothing in the generator path is installed
on the host — there is no `llama-server.exe` to put on `PATH`.

Profile B is the default — in the script, and in `docker-compose.yml` when no
`LLM_MODEL_FILE` is set — and the quality choice. It needs ~20 GB of free RAM
for weights — tight on 32 GB once Postgres is also running, and profile A is the
fallback when RAM is short; 64 GB is the highest-ROI upgrade for this box. For
MoE offload, **memory bandwidth beats core count** — verify your DDR5 is running
at its rated speed before blaming the software. The 50-series is Blackwell
(sm_120): pin a current `ghcr.io/ggml-org/llama.cpp:server-cuda` tag
(`LLAMA_IMAGE`) or you will silently fall back to slow paths.

A machine with no weights at all can still run everything except `ask`: set
`embedding.backend: hash` and `llm.backend: echo`, build with
`LAWRAG_TARGET=runtime`, and retrieval stays fully functional. `GET /health`
reports `production_ready: false` in that state.

After switching, run `docker compose run --rm cli calibrate`, put the proposed
`retrieval.min_*` values in `config/default.yaml`, and re-run
`docker compose run --rm cli eval`.

### Kazakh answers

The model never composes legal Kazakh. Retrieval and reasoning run over both
languages; the Kazakh answer quotes the authoritative Kazakh text **verbatim**
from the corpus, and only the plain-language wrapper is generated. The statutory
wording is itself the legal fact, and a paraphrase of it is not the law. The
prompt in `generate/prompts.py` states this as a hard rule.

---

## Adapting the models to the corpus

bge-m3 and the bge reranker were trained on general web text. Kazakh
statutory language is a long way from that, and the lever that moves
retrieval quality here is not a bigger model but the same model fine-tuned on
this corpus. There is no labelled data, so the corpus is its own supervision
(`train/pairs.py`):

| pair kind | query | positive | what it teaches |
| --- | --- | --- | --- |
| `heading` | the article's title | the article | terse queries, the way people actually type |
| `xref` | the sentence that cites article B, **with the citation cut out** | article B | meaning, not numbers: "в порядке, установленном ___" must find the registration article without "статьей 76" to match on |
| `xlang` | the Russian text of a node | its Kazakh text (and back) | the two legal vocabularies land in one space, so a Kazakh question finds the Russian article |
| `definition` | a defined term, bare and as a question | the definition | glossary lookups |
| `doc2query` | questions the local LLM writes per chunk (`--doc2query`) | the chunk | realistic phrasing; cached by content hash |
| `querylog` | real questions whose every citation passed verification | the cited article | what users ask — the only kind that is not derived from the text itself |

```bash
docker compose run --rm cli train-pairs data/adilet             # -> data/train/
docker compose run --rm cli train-eval /models/bge-m3           # baseline, no database needed
docker compose run --rm cli train --base /models/bge-m3 --out /models/bge-m3-lawrag
docker compose run --rm cli train-eval /models/bge-m3 /models/bge-m3-lawrag
```

`train` is contrastive (`MultipleNegativesRankingLoss`): every other positive
in the batch is a negative, plus one **hard negative** per query — the
best-ranked *wrong* article under the base model, which is what teaches the
difference between article 277 and article 278. It is sized for an 8 GB card:
the cached loss encodes in mini-batches, only the top eight transformer layers
are updated (the lower ones hold the multilingual knowledge the model came
with; freezing them cuts optimiser memory by two thirds and guards against
forgetting on a corpus this small), and gradient checkpointing is on. One
epoch over the adilet snapshot is under an hour on a laptop GPU.

`train-eval` holds out whole articles (5% by default), so no dev query points
at text that was a training positive, and ranks every chunk in the corpus by
cosine similarity — the dense branch alone, no lexical fusion, floor or
reranker. Absolute numbers are therefore below end-to-end recall; the
difference between two models on the same queries is what it is for. `xref`
is the kind that looks most like a real question.

Measured on the 2026-09-08 adilet snapshot (41 documents, 43,857 chunks,
31,246 training pairs, one epoch, 78 minutes on an 8 GB laptop GPU), on 1,620
held-out queries:

| kind | queries | recall@1 base → tuned | recall@10 base → tuned | MRR@10 base → tuned |
| --- | ---: | ---: | ---: | ---: |
| `xref` — citation sentences, closest to real questions | 793 | 0.12 → **0.24** | 0.42 → **0.52** | 0.21 → **0.32** |
| `heading` — article titles | 392 | 0.86 → 0.88 | 0.97 → 0.98 | 0.91 → 0.93 |
| `xlang` — RU text finds the KK article and back | 435 | 0.86 → 0.79 | 0.98 → 0.98 | 0.91 → 0.86 |
| all | 1,620 | 0.50 → 0.54 | 0.70 → 0.76 | 0.57 → 0.61 |

The realistic kind nearly doubles at rank 1. The one regression is honest
and tunable: cross-language matching at rank 1 gave up some ground, because
the run capped `xlang` pairs at 8,000 to keep citation sentences from being
drowned out; raise the cap (`DEFAULT_CAPS` in `train/pairs.py`) or train a
second epoch if bilingual recall matters more than monolingual precision for
your users. The trained directory is `models/bge-m3-lawrag`, selected with
`EMBEDDING_MODEL=bge-m3-lawrag` in `docker/.env`; `config/host-gpu.yaml`
runs it from a host interpreter on the card.

After switching: point `LAWRAG_EMBEDDING_MODEL_PATH` at the new directory
(it loads through both the `bge-m3` and `sentence-transformers` backends —
the base model's sparse and ColBERT heads are copied alongside), re-ingest so
every chunk is re-embedded, and re-run `calibrate`: `min_dense_score` is on
the model's own scale and moves with it.

Two limits, stated plainly. The pairs are derived from the text, so they
measure how well the model reads the law, not how well it reads people; the
query log closes that gap only once the system is in use, and `doc2query`
approximates it before then. And the corpus cannot currently be enlarged from
adilet: since September 2026 the site serves act text only to browsers running
its JavaScript and states in its robots policy that corpus-collecting crawlers
are closed while search and citation stay open. The fetcher now refuses that
shell instead of saving it as a document. New text has to come from a
reference bank export: put the files in a directory, `lawrag pack` it into a
bundle, and `lawrag ingest` the bundle. (There is no `--source-dir` flag,
although the fetcher's log message and docstring still name one.)

---

## Evaluation

```bash
docker compose run --rm cli eval src/lawrag/evalkit/questions.sample.yaml
```

Measures whether the correct **set** of articles reached the model, not whether
the top-1 vector hit was right — legal answers routinely span a penalty article,
a definition in another code, and an exception referring out to a third.

The shipped question set is written against the **synthetic fixture** in
`tests/fixtures/synthetic/`, which is what `docker compose run --rm tests`
ingests; the numbers below come from it. Run against `data/adilet` it measures
nothing, because its article ids name invented texts. Write a question set for
the real corpus before reading anything into `eval` on it.

```
recall@chunk  0.86
recall@packet 1.00   ← the gap is what hydration is buying you
MRR           0.86
abstain acc.  1.00
pass rate     1.00
```

Build the real question set from actual compliance questions, verified by a
lawyer. Include out-of-corpus questions: abstention accuracy is as important as
recall.

---

## Operations

- `docker compose run --rm cli stats` — row counts plus **unresolved
  references**, i.e. which acts are cited by your corpus but missing from it.
  Unresolved refs are kept on purpose; hiding them would make the graph look
  complete when it is not.
- The corpus lives in the `lawrag-pgdata` volume, not in an image. `docker
  compose down` keeps it; `down -v` destroys it and the corpus has to be
  re-ingested from the snapshots.
- Every answer is written to `query_log`: question, as-of date, snapshot,
  retrieved chunk ids, citations, verification verdict, latency.
- On each new snapshot, diff which previously-cited articles changed. That both
  invalidates cached answers and tells you which internal policies need review.
- `verify_signature()` in `ingest/bundle.py` is a hook, not an implementation.
  Wire it to your key infrastructure; with `--require-signature` it fails closed.

---

## Reference

### CLI

Every command runs as `docker compose run --rm cli <command>` from `docker/`,
or as `lawrag <command>` from a host interpreter. All of them take `--config`.

| command | what it does |
| --- | --- |
| `init-db` | create the schema |
| `stats` | row counts and unresolved references |
| `fetch <dir> [--as-of] [--codes] [--langs]` | connected side: download an adilet snapshot (see the note on adilet above) |
| `pack <src> <out>` | build a bundle with a SHA-256 manifest |
| `verify-bundle <dir> [--require-signature]` | check a bundle before import; exit 2 on rejection |
| `ingest <dir> [--reset] [--no-verify] [--no-build-index]` | parse, chunk, extract, embed and index a corpus directory |
| `context` | optional contextual-retrieval sentences per chunk, cached by content hash |
| `search "<q>" [--as-of]` | retrieval only, with per-branch scores |
| `ask "<q>" [--as-of]` | full answer: retrieve, generate, verify citations |
| `eval [questions.yaml]` | recall@chunk, recall@packet, MRR, abstain accuracy |
| `calibrate [questions.yaml]` | propose the `retrieval.min_*` relevance floors |
| `inspect <file> --code --valid-from` | parse one document and print the result; no database |
| `serve [--host] [--port]` | run the API and console |
| `train-pairs <dir> [--doc2query]` | mine training pairs from the corpus into `data/train/` |
| `train --base --out` | contrastive fine-tune of the embedder |
| `train-eval <model>...` | held-out dense retrieval comparison; no database |

`scripts/resume_ingest.py` finishes an interrupted ingest: `--embed` fills in
missing embeddings, `--link` aligns RU↔KK chunks, `--index` builds the HNSW
index; with no flags it only reports.

### HTTP API

Interactive docs at `http://127.0.0.1:8000/docs`.

| endpoint | what it does |
| --- | --- |
| `GET /health` | corpus counts, backends, `production_ready` |
| `POST /ask` | `{question, as_of?, lang?}` → answer with verified citations |
| `POST /search` | same body plus `include_packet` → ranked hits, optionally the context packet |
| `GET /article/{act_code}/{number}?as_of=` | point-in-time article lookup, no search |
| `GET /api/stats`, `/api/metrics` | console dashboards: counts, latency, verification outcomes |
| `GET /api/releases`, `/api/acts`, `/api/acts/{code}/toc`, `/api/nodes/{id}` | corpus browsing |
| `GET /api/queries`, `/api/queries/{id}` | the query log |
| `POST /api/ingest`, `GET /api/ingest/jobs[/{id}]` | console import and its history — the only write |

Access rules are under **Console → Access** above.

---

## Status

As of the last commit (`7e31571`, branch `corpus-split-and-embedder-adaptation`):

- The relevance floors in `config/default.yaml` (0.30 / 0.60 / 0.45) have not
  been calibrated for bge-m3 on the real corpus — `calibrate` needs a real
  question set first.
- There is no question set for `data/adilet` yet; the shipped one targets the
  synthetic fixture.
- Only current redactions are stored. Historical text needs one
  `fetch --as-of` snapshot per date, and adilet no longer serves text to the
  fetcher.
- BGE-M3's sparse and ColBERT outputs are unused; retrieval uses the dense
  vector only. A design choice, not a bug.
- `verify_signature()` is a hook without an implementation.
- `LAWRAG_API_TOKEN` is one shared secret; there are no user accounts.

---

## Layout

```
docker/                     everything Docker-specific, and nothing else is
  Dockerfile                two targets: runtime (no weights stack) and models
  Dockerfile.dockerignore   build-context hygiene (BuildKit per-Dockerfile ignore)
  docker-compose.yml        db, api, cli, tests, llm, model download
  docker-compose.gpu.yml    GPU overlay — device reservations, the CUDA llama.cpp
  .env.example              this machine: weights dir, ports, image target
  scripts/
    bootstrap.*             clean clone to a working system, in containers
    download_models.*       connected-side weight fetch, in a container
    serve_llm.*             generator profiles A/B as compose variables
config/default.yaml         every knob, with the reasoning next to it
config/local.yaml           host interpreter, placeholder models
config/host-gpu.yaml        host interpreter, fine-tuned embedder on CUDA
scripts/
  download_models.py        the weight fetch itself
  resume_ingest.py          finish an interrupted ingest
src/lawrag/
  sql/001_schema.sql        bitemporal schema + overlap constraints, versioned
                            citation graph, glossary, audit log
  models.py                 canonical legal document model, norm hierarchy
  ingest/
    fetch_adilet.py         connected-side snapshot fetcher
    dating.py               redaction dates from amendment footnotes
    bundle.py               manifest, verification, the air-gap boundary
    structure.py            RU/KK heading dialects, notes vs amendments
    parser.py               source text and adilet HTML -> tree
    refs.py                 cross-references and defined terms
    acts.py                 act registry and citation-name resolution
    pipeline.py             ingest orchestration
  chunking/
    chunker.py              structural chunking, embed vs display text
    context.py              contextual-retrieval batch job
  index/
    db.py  embed.py  indexer.py
  retrieve/
    query.py hybrid.py rerank.py hydrate.py packet.py engine.py
  generate/
    llm.py prompts.py answer.py
  verify/citations.py       mechanical citation checking
  evalkit/harness.py        retrieval eval
  evalkit/calibrate.py      relevance-floor calibration
  train/
    pairs.py                training pairs mined from the corpus itself
    finetune.py             contrastive fine-tune of the embedder, 8 GB-sized
    evaluate.py             held-out dense retrieval eval, no database
  api/
    main.py                 the CLI-mirroring HTTP surface
    console.py              read-only endpoints for the browser console
    security.py             token / loopback access control
    static/                 the console itself: no build step, no CDN
  cli.py
data/adilet/                real adilet snapshot: 41 documents (HTML) + documents.json
data/train/                 training pairs from `train-pairs` (git-ignored)
data/uploads/               console imports and their job.json history (git-ignored)
models/                     MODEL_DIR: bge-m3/, bge-m3-lawrag/, the reranker,
                            the .gguf generator (git-ignored, mounted at /models)
tests/                      107 tests; DB-backed ones skip without Postgres
  fixtures/synthetic/       the invented fixture the suite ingests
  test_regressions.py       one test per defect found in review, each named
                            after the failure it pins down
```

---

## What this is not

Not legal advice, and it does not try to be. It reports what the text of a norm
says, with a pinpoint citation, an in-force date and a verbatim quote, and it
escalates when the question calls for judgement. Every answer carries a
disclaimer. The Law «О правовых актах» hierarchy is encoded so that conflicts
between acts of different legal force can be surfaced — but resolving them is a
lawyer's job, not the system's.

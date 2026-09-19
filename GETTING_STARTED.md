# Getting started

How to get `lawrag` running on a fresh machine, in the order you actually need
it. `README.md` is the design rationale — *why* the system is built this way;
this file is the *how*.

Air-gapped RAG over the Codes of the Republic of Kazakhstan (RU + KK). The
corpus in `data/adilet/` is a real snapshot of adilet.zan.kz (41 documents,
21 codes, fetched 2026-09-08); `tests/fixtures/synthetic/` is invented fixture
data the test suite runs on. Keep them apart — `ingest` refuses a directory
holding two different documents under one act code.

---

## Prerequisites

Docker is the only requirement. Nothing is installed on the host — no Python,
no virtualenv, no `pip install -e`. Every command below runs inside the image.

Every `docker compose` command runs **from `docker/`**, where all Docker
specifics live. `docker/.env` is optional: compose has a default for every
variable it reads (`docker/docker-compose.yml`). Copy `docker/.env.example` to
`docker/.env` only when you need to change one.

Two image targets:

| target | size | what it carries |
| --- | --- | --- |
| `models` (default) | ~10 GB | adds torch + FlagEmbedding — needed by the bge-m3 backends in `config/default.yaml` |
| `runtime` | ~390 MB | ingest, retrieval over the hash embedder, the console, a llama.cpp HTTP client |

---

## Path A — the full system (with weights)

This is the real thing: bge-m3 embeddings, the reranker, a local generator,
and `ask` that actually answers.

```bash
# 1. weights — CONNECTED side only, several GB, the one step that uses network
docker/scripts/download_models.sh
#    lands models/bge-m3 and models/bge-reranker-v2-m3
#    put the generator's .gguf file beside them

# 2. everything else: build, database, ingest, console
docker/scripts/bootstrap.sh              # Windows: docker\scripts\bootstrap.ps1

# 3. the generator (needed only by `ask`)
docker/scripts/serve_llm.sh              # profile B, the default
docker/scripts/serve_llm.sh A            # smaller, when RAM is short
docker/scripts/serve_llm.sh B --cpu      # no GPU

# 4. re-tune the relevance floor for this embedder, then re-measure
cd docker
docker compose run --rm cli calibrate
```

`MODEL_DIR` decides where the weights live; it defaults to `../models`
relative to `docker/`, i.e. `lawrag/models/`. Set it in `docker/.env` to keep
them elsewhere.

Console: <http://127.0.0.1:8000/> · API docs: <http://127.0.0.1:8000/docs>

## Path B — no weights, retrieval only

Placeholder backends (deterministic hash embedder, echo LLM) keep the whole
pipeline runnable with nothing on disk. Retrieval is fully functional; `ask`
abstains with `no_model_loaded` and `GET /health` reports
`production_ready: false`. The console raises a banner on every view so
placeholder output is never mistaken for an answer.

```bash
cd docker
cp .env.example .env
sed -i '' 's/^LAWRAG_TARGET=models/LAWRAG_TARGET=runtime/' .env   # Linux: sed -i '...'
export LAWRAG_EMBEDDING_BACKEND=hash LAWRAG_LLM_BACKEND=echo LAWRAG_RERANK_BACKEND=none

docker compose build app
docker compose up -d --wait db
docker compose run --rm cli ingest data/adilet --reset --no-verify
docker compose up -d app
```

## Path C — host interpreter against the compose database

For editing and debugging from the IDE. Uses the untracked `config/local.yaml`
(hash embedder, echo LLM, database on `127.0.0.1:5433`).

```bash
cd docker && docker compose up -d --wait db && cd ..

LAWRAG_CONFIG=config/local.yaml .venv/bin/lawrag init-db
LAWRAG_CONFIG=config/local.yaml .venv/bin/lawrag ingest data/adilet --reset --no-verify
LAWRAG_CONFIG=config/local.yaml .venv/bin/lawrag search "Какой штраф за просрочку?"
```

The host `.venv` carries the base dependencies only. `train` and the bge-m3
backends need the `models` extra (`pip install -e ".[models]"`) — or just run
them in the container.

---

## Everyday commands

From `docker/`, as `docker compose run --rm cli <command>`; or as
`lawrag <command>` from a host interpreter. All take `--config`.

```bash
docker compose run --rm cli stats                                  # row counts, unresolved refs
docker compose run --rm cli search "В какой срок нужно встать на учет?"
docker compose run --rm cli search "Какой штраф за просрочку?" --as-of 2025-03-15
docker compose run --rm cli search "Тіркеу есебіне қою мерзімі қандай?"   # Kazakh
docker compose run --rm cli ask    "Продлевается ли срок для малого бизнеса?"
docker compose run --rm cli eval                                   # recall, MRR, abstain accuracy
docker compose run --rm cli calibrate                              # propose retrieval.min_* floors
docker compose run --rm tests                                      # pytest against a live db
docker compose run --rm tests tests/test_chunker.py::test_name -q   # one test
```

`cli` is the same image the API runs, against the same config and database, so
a question answered in the shell and the same question in the browser go
through identical code.

---

## Training the embedder

Only the **dense embedder** is trained. The reranker and the generator are used
as downloaded. There is no labelled data, so the corpus is its own supervision.

```bash
cd docker
docker compose run --rm cli train-pairs data/adilet          # 1. mine pairs -> data/train/
docker compose run --rm cli train-eval /models/bge-m3        # 2. baseline
docker compose run --rm cli train --base /models/bge-m3 --out /models/bge-m3-lawrag
docker compose run --rm cli train-eval /models/bge-m3 /models/bge-m3-lawrag   # 3. compare
```

1. **`train-pairs`** parses the corpus directly (no database, no weights) and
   writes `pairs.train.jsonl`, `pairs.dev.jsonl` and `corpus.jsonl`. The split
   is by *article*, not by pair, so a dev query never points at text the model
   saw as a positive. `--doc2query` adds LLM-written questions (skipped when
   `llm.backend` is `echo`); `--query-log` adds real verified questions.
2. **`train`** is contrastive with one mined hard negative per query — the
   best-ranked *wrong* article under the base model, which is what teaches the
   difference between article 277 and article 278. Sized for an 8 GB card:
   cached loss, only the top 8 transformer layers trainable, gradient
   checkpointing. Useful knobs: `--mini-batch-size` (VRAM), `--batch-size`
   (in-batch negatives), `--max-pairs` (time budget), `--epochs`.
3. **`train-eval`** ranks every chunk by cosine similarity — the dense branch
   alone, no fusion, floor or reranker — and reports recall@1/5/10 and MRR@10
   per pair kind. Absolute numbers sit below end-to-end recall by design; the
   base→tuned *difference* is what it measures. `xref` and `querylog` are the
   kinds closest to real questions.

**After training**, three things, in order: point
`LAWRAG_EMBEDDING_MODEL_PATH` at the new directory (or set
`EMBEDDING_MODEL=bge-m3-lawrag` in `docker/.env`), **re-ingest** so every chunk
is re-embedded, and re-run **`calibrate`** — `retrieval.min_dense_score` is on
the model's own cosine scale and moves with it.

Corpus embeddings are cached per (model, corpus content) under
`data/train/cache/`, so the baseline run also warms the cache that
hard-negative mining reuses.

---

## Configuration

Precedence, low to high:

```
dataclass defaults in config.py  →  YAML (config/default.yaml or $LAWRAG_CONFIG)
                                 →  .env  →  environment
```

Any scalar is overridable as `LAWRAG_<SECTION>_<FIELD>` — `LAWRAG_DB_DSN`,
`LAWRAG_EMBEDDING_DEVICE`, `LAWRAG_EMBEDDING_MODEL_PATH`. Unknown YAML keys
raise rather than being ignored. A new setting goes in the dataclass *and* in
`config/default.yaml`, with the reasoning as a comment.

| file | what it is for |
| --- | --- |
| `config/default.yaml` | every durable knob, with the reasoning next to it; names the production backends |
| `config/local.yaml` | host interpreter on placeholder models, database on 5433 |
| `config/host-gpu.yaml` | the fine-tuned embedder on the host's card |
| `docker/.env` | *this machine*: where the weights are, which ports are free, which image target |

---

## Troubleshooting

**`FileNotFoundError: Path /models/bge-m3 not found`** — the weights were never
downloaded. `/models` is the bind mount of `MODEL_DIR` (default `../models`
relative to `docker/`), and Docker creates that directory empty if it is
missing. Run `docker/scripts/download_models.sh`, or use Path B.

**`docker compose` hangs or says it cannot connect** — the Docker daemon is not
running. Start Docker Desktop first.

**Everything is slow and there is no GPU** — `train`, `train-eval` and ingest
with bge-m3 fall back to the CPU. Encoding the full corpus (~44k chunks) is
minutes on a laptop GPU and about an hour on a CPU; a training epoch is far
longer. `train-pairs` needs neither GPU nor weights.

**`eval` numbers look meaningless on `data/adilet`** — they are. The shipped
question set targets the synthetic fixture, and its article ids name invented
texts. Write a question set for the real corpus before reading anything into it.

**DB-backed tests all skip** — the `live_db` fixture skips silently unless
`LAWRAG_TEST_DSN` is set and reachable; it never falls back to `db.dsn`,
because the suite ingests with `reset=True`. A green run without it is not a
full run.

```bash
LAWRAG_TEST_DSN=postgresql://postgres:2006@localhost:5433/lawrag_test .venv/bin/pytest -q
```

**Schema changed but the database did not** — there is no migration tool. The
single schema file `src/lawrag/sql/001_schema.sql` is applied by `init-db` /
`ingest --reset`, and mounted into Postgres initdb, which runs **only on a
fresh volume**. On an existing volume: `init-db --reset` plus a re-ingest.

**Ingest was interrupted** — `scripts/resume_ingest.py --embed --index --link`
finishes it; with no flags it only reports what is missing.

**The corpus disappeared** — it lives in the `lawrag-pgdata` volume, not in an
image. `docker compose down` keeps it; `down -v` destroys it.

---

## Where things live

```
config/           default.yaml (durable knobs) · local.yaml · host-gpu.yaml
data/adilet/      the real corpus snapshot + documents.json manifest
data/train/       mined pairs and cached corpus embeddings (written by train-pairs)
docker/           Dockerfile, compose files, .env.example, wrapper scripts
scripts/          download_models.py, resume_ingest.py
src/lawrag/
  ingest/         bundle verify → parse (RU + KK dialects) → refs → dating
  chunking/       structural chunks; build_embed_text
  index/          Postgres schema and writers
  retrieve/       analyze → hybrid search → rerank → dedupe → hydrate → packet
  generate/       prompts and the LLM client
  verify/         mechanical citation verification
  train/          pairs.py · finetune.py · evaluate.py
  api/            main.py, console.py, static/ (no build step)
  sql/001_schema.sql
tests/            pytest; fixtures/synthetic/ is invented data
```

Read `README.md` before changing chunking, retrieval or verification — the
constraints those parts satisfy are not obvious from the code alone.

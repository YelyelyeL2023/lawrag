#!/usr/bin/env bash
# Stand up a working system on the sample corpus, from nothing.
#
# The only thing this needs on the host is Docker. No Python, no virtualenv,
# no pip install -e: every command below runs inside the lawrag image.
#
#   docker/scripts/bootstrap.sh          CPU
#   docker/scripts/bootstrap.sh --gpu    add the GPU overlay (embedder on the card)
set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE=(docker compose -f docker-compose.yml)
if [[ "${1:-}" == "--gpu" ]]; then
  COMPOSE+=(-f docker-compose.gpu.yml)
fi

echo "==> building the lawrag image"
"${COMPOSE[@]}" build app

echo "==> starting Postgres"
"${COMPOSE[@]}" up -d --wait db

echo "==> ingesting the SYNTHETIC sample corpus"
"${COMPOSE[@]}" run --rm cli ingest data/sample --reset --no-verify

echo "==> corpus"
"${COMPOSE[@]}" run --rm cli stats

echo "==> retrieval eval"
"${COMPOSE[@]}" run --rm cli eval

echo "==> starting the API and console"
"${COMPOSE[@]}" up -d app

cat <<'MSG'

Ready.  Console: http://127.0.0.1:8000/   API docs: http://127.0.0.1:8000/docs

    docker compose run --rm cli search "В какой срок нужно встать на регистрационный учет?"
    docker compose run --rm cli search "Какой штраф за просрочку?" --as-of 2025-03-15
    docker compose run --rm cli ask    "Продлевается ли срок для малого бизнеса?"

`ask` needs the generator running and the weights mounted:

    docker compose --profile llm up -d llm      (see README, "Models")
MSG

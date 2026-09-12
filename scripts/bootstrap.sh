#!/usr/bin/env bash
# Stand up a working system on the sample corpus, from nothing.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> starting Postgres"
docker compose up -d db
until docker compose exec -T db pg_isready -U lawrag -d lawrag >/dev/null 2>&1; do
  sleep 1
done

echo "==> installing lawrag"
pip install -e ".[dev]"

echo "==> ingesting the SYNTHETIC sample corpus"
lawrag ingest data/sample --reset --no-verify

echo "==> corpus"
lawrag stats

echo "==> retrieval eval"
lawrag eval

cat <<'MSG'

Ready. Try:
    lawrag search "В какой срок нужно встать на регистрационный учет?"
    lawrag search "Какой штраф за просрочку?" --as-of 2025-03-15
    lawrag ask    "Продлевается ли срок для малого бизнеса?"

Note: the shipped config uses placeholder models (hash embedder, echo LLM).
`search` is fully functional; `ask` will abstain until a real LLM is configured.
See README section "Switching on the real models".
MSG
